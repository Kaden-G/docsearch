#!/usr/bin/env python3
"""
Retriever Agent
===============

The retrieval workhorse. For each sub-query it:

    1. Runs hybrid search (FAISS + BM25 + cross-encoder rerank) via the shared
       backend, with query-time HyDE disabled (the agent reformulates itself).
    2. Checks coverage with a deterministic gate — top cosine similarity vs
       `score_gate` — with NO LLM call.

By default this is a single deterministic pass per sub-query: fast, with zero
extra LLM calls. Higher first-pass recall from Smart Indexing makes that safe.

Opt-in (`self_assess=True`): when the deterministic gate flags weak coverage,
the agent spends ONE LLM call to reformulate the query (synonyms / expanded
acronyms) and searches again — up to `max_passes` times. All hits are merged
and de-duplicated by chunk_id.
"""

from typing import Dict, List

from .base import Agent
from schemas import PlannerOutput, RetrieverOutput, RetrievedChunk


_REFORMULATE_SYSTEM = """You improve a search query for an document knowledge base.

You are given a sub-query and short snippets of the chunks it retrieved, which were judged
insufficient. Propose ONE reformulated search query using different terminology (synonyms,
expanded acronyms, related component names) that might surface better chunks.

Respond with ONLY valid JSON:
{"reformulated_query": "..."}"""


class RetrieverAgent(Agent):
    name = "retriever"

    def __init__(
        self,
        toolbox,
        verbose: bool = False,
        top_k: int = 5,
        max_passes: int = 3,
        self_assess: bool = False,
        score_gate: float = 0.45,
    ):
        super().__init__(toolbox, verbose=verbose)
        self.top_k = top_k
        self.max_passes = max_passes
        self.self_assess = self_assess
        self.score_gate = score_gate

    def _run(self, plan: PlannerOutput) -> RetrieverOutput:
        merged: Dict[str, RetrievedChunk] = {}
        total_passes = 0
        coverage_notes: List[str] = []

        for sq in plan.sub_queries:
            query = sq.text
            for patt in range(1, self.max_passes + 1):
                total_passes += 1
                hits = self.tools.search_index(query, top_k=self.top_k, rerank=True)

                # Merge: keep the best score seen for each chunk.
                for h in hits:
                    existing = merged.get(h.chunk_id)
                    if existing is None or h.score > existing.score:
                        merged[h.chunk_id] = h

                # Deterministic coverage gate (no LLM): top cosine similarity.
                top_sim = max((h.similarity for h in hits), default=0.0)
                covered = bool(hits) and top_sim >= self.score_gate

                # Default path: a single deterministic pass — no LLM self-assessment.
                if not self.self_assess or not self.tools.searcher.llm_provider:
                    if not covered:
                        coverage_notes.append(f"Weak coverage (sim={top_sim:.2f}) for: {sq.text}")
                    break

                # Opt-in path: only spend an LLM reformulation call when the
                # deterministic gate says coverage is still weak.
                if covered or patt == self.max_passes:
                    if not covered and patt == self.max_passes:
                        coverage_notes.append(f"Partial coverage for: {sq.text}")
                    break

                reformulated = self._reformulate(sq.text, hits)
                if not reformulated:
                    break
                query = reformulated  # try again with a better query

        chunks = sorted(merged.values(), key=lambda c: c.score, reverse=True)

        self._detail = {
            "sub_queries": len(plan.sub_queries),
            "total_passes": total_passes,
            "unique_chunks": len(chunks),
            "self_assess": self.self_assess,
            "coverage_notes": coverage_notes,
        }
        return RetrieverOutput(
            chunks=chunks,
            passes=total_passes,
            coverage_notes="; ".join(coverage_notes),
        )

    def _reformulate(self, sub_query: str, hits: List[RetrievedChunk]) -> str:
        """Ask the LLM for ONE reformulated query. Used only on the opt-in
        self_assess path when the deterministic gate flags weak coverage.
        Returns "" if there is no usable reformulation (which ends the loop)."""
        snippets = "\n".join(
            f"- ({h.doc_name} p{h.page}) {h.text[:240]}" for h in hits[:5]
        )
        user = f"Sub-query: {sub_query}\n\nRetrieved snippets:\n{snippets}\n\nReformulation JSON:"
        raw = self.llm(_REFORMULATE_SYSTEM, user, max_tokens=120, temperature=0.0)
        parsed = self.parse_json(raw)
        if not parsed:
            return ""
        return (parsed.get("reformulated_query") or "").strip()
