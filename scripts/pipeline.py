#!/usr/bin/env python3
"""
Pipeline Orchestrators
======================

Two pipelines, one shared backend (DocSearcher), one unified output shape
(PipelineResult). This is the core of the experiment: feed both the same query
on the same index with the same LLM, and compare.

    TraditionalPipeline : query -> search -> stuff top-k -> single LLM answer
                          (mirrors the v1 tool's search_and_answer path)

    AgenticPipeline     : query -> Planner -> Retriever (multi-pass)
                          -> Assembler -> Answerer -> Verifier
                          (each stage instrumented for latency + token cost)

Both return a PipelineResult so scripts/evaluate.py can score them identically.
"""

import re
import time
from typing import List, Optional

from schemas import (
    PipelineResult, StageTrace, LLMUsage, RetrievedChunk, Citation,
    AssembledContext,
)
from tools import ToolBox
from agents import (
    PlannerAgent, RetrieverAgent, AssemblerAgent, AnswererAgent, VerifierAgent,
)


# Parses footnote reference lines from the traditional pipeline's answer too,
# e.g.  [^1]: handbook, Page 27
_REF_LINE = re.compile(r"\[\^(\d+)\]:\s*([^,\n]+),\s*Page\s+(\w+)", re.IGNORECASE | re.MULTILINE)


class TraditionalPipeline:
    """Single-shot RAG — the v1 baseline, wrapped to emit a PipelineResult."""

    name = "traditional"

    def __init__(self, searcher, top_k: int = 5, verbose: bool = False):
        self.searcher = searcher
        self.top_k = top_k
        self.verbose = verbose

    def run(self, query: str) -> PipelineResult:
        usage = LLMUsage()
        stages: List[StageTrace] = []
        t0 = time.time()

        # Retrieval (HyDE may add LLM calls inside search; counted as one stage).
        t_r = time.time()
        raw_results = self.searcher.search(query, top_k=self.top_k, rerank=True)
        chunks = [RetrievedChunk.from_search_result(r, source_query=query) for r in raw_results]
        stages.append(StageTrace(name="retrieval",
                                 latency_ms=(time.time() - t_r) * 1000,
                                 detail={"chunks": len(chunks)}))

        # Single LLM answer via the shared backend's own generator.
        t_a = time.time()
        answer = self.searcher.generate_answer(query, raw_results, max_context_chunks=self.top_k) or ""
        # The traditional generator doesn't expose token counts; record one call.
        usage.record(prompt_tokens=0, completion_tokens=0)
        usage.calls = max(usage.calls, 1)
        stages.append(StageTrace(name="answer",
                                 latency_ms=(time.time() - t_a) * 1000,
                                 usage=LLMUsage(calls=1),
                                 detail={"answer_chars": len(answer)}))

        citations = _parse_traditional_citations(answer, chunks)

        return PipelineResult(
            query=query,
            answer=answer,
            pipeline=self.name,
            confidence="n/a",
            citations=citations,
            chunks=chunks,
            stages=stages,
            total_latency_ms=(time.time() - t0) * 1000,
            usage=usage,
        )


class AgenticPipeline:
    """Five-agent pipeline with verification, on the same shared backend."""

    name = "agentic"

    def __init__(
        self,
        searcher,
        top_k: int = 5,
        max_passes: int = 3,
        self_assess: bool = False,
        verify: bool = True,
        verbose: bool = False,
    ):
        self.searcher = searcher
        self.tools = ToolBox(searcher)
        self.verbose = verbose
        self.verify = verify

        self.planner = PlannerAgent(self.tools, verbose=verbose)
        self.retriever = RetrieverAgent(
            self.tools, verbose=verbose, top_k=top_k,
            max_passes=max_passes, self_assess=self_assess,
        )
        self.assembler = AssemblerAgent(self.tools, verbose=verbose)
        self.answerer = AnswererAgent(self.tools, verbose=verbose)
        self.verifier = VerifierAgent(self.tools, verbose=verbose)

    def run(self, query: str) -> PipelineResult:
        usage = LLMUsage()
        stages: List[StageTrace] = []
        t0 = time.time()

        if self.verbose:
            print(f"\n[AgenticPipeline] query: {query!r}")

        # 1) Plan
        plan, tr = self.planner.run(query)
        stages.append(tr); usage.add(tr.usage)

        # 2) Retrieve (multi-pass)
        retrieved, tr = self.retriever.run(plan)
        stages.append(tr); usage.add(tr.usage)

        # 3) Assemble (deterministic)
        context, tr = self.assembler.run(retrieved)
        stages.append(tr); usage.add(tr.usage)

        # 4) Answer
        answer_out, tr = self.answerer.run(query, context)
        stages.append(tr); usage.add(tr.usage)

        # 5) Verify (the zero-hallucination gate)
        final_answer = answer_out.answer_markdown
        confidence = "n/a"
        citations = answer_out.citations
        if self.verify:
            verified, tr = self.verifier.run(answer_out, context.ordered_chunks)
            stages.append(tr); usage.add(tr.usage)
            final_answer = verified.verified_answer
            confidence = verified.confidence

        return PipelineResult(
            query=query,
            answer=final_answer,
            pipeline=self.name,
            confidence=confidence,
            citations=citations,
            chunks=context.ordered_chunks,
            stages=stages,
            total_latency_ms=(time.time() - t0) * 1000,
            usage=usage,
        )


def _parse_traditional_citations(answer: str, chunks: List[RetrievedChunk]) -> List[Citation]:
    """Resolve the v1 answer's footnote references to source chunks by label."""
    by_label = {}
    for c in chunks:
        by_label.setdefault((c.doc_name, str(c.page)), c)

    citations: List[Citation] = []
    seen = set()
    for m in _REF_LINE.finditer(answer):
        marker = int(m.group(1))
        if marker in seen:
            continue
        doc = m.group(2).strip()
        page = m.group(3).strip()
        chunk = by_label.get((doc, page))
        if chunk is None:
            # loose match on doc name
            for c in chunks:
                if c.doc_name and (c.doc_name in doc or doc in c.doc_name):
                    chunk = c
                    break
        if chunk is not None:
            citations.append(Citation(marker=marker, chunk_id=chunk.chunk_id,
                                      doc_name=chunk.doc_name, page=chunk.page,
                                      section=chunk.section))
            seen.add(marker)
    return citations


def build_pipeline(kind: str, searcher, **kwargs):
    """Factory: 'traditional' or 'agentic'."""
    kind = (kind or "").lower().strip()
    if kind == "agentic":
        return AgenticPipeline(searcher, **kwargs)
    return TraditionalPipeline(
        searcher,
        top_k=kwargs.get("top_k", 5),
        verbose=kwargs.get("verbose", False),
    )
