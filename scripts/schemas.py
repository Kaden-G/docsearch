#!/usr/bin/env python3
"""
Agentic Pipeline Schemas
========================

Dependency-free dataclass I/O models for each agent in the agentic RAG
pipeline, plus instrumentation primitives used to compare the agentic
pipeline against the traditional single-shot RAG path.

Why dataclasses and not Pydantic? This project must stay airgap-friendly and
the agentic-vs-traditional comparison must avoid introducing dependency
confounds. The standard library is enough here: we parse LLM JSON with the
`json` module and validate in the agents themselves.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# Instrumentation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMUsage:
    """Tally of LLM work. The core scalability/cost axis for the comparison.

    Agentic pipelines make several LLM calls per query (planner, retriever
    self-assessment, answerer, verifier); the traditional path makes one. This
    is what lets us quantify the cost of the accuracy gain.
    """
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "LLMUsage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens

    def record(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Register a single LLM call and its token usage."""
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens


@dataclass
class StageTrace:
    """Per-agent timing + LLM accounting, collected by the orchestrator."""
    name: str
    latency_ms: float = 0.0
    usage: LLMUsage = field(default_factory=LLMUsage)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "latency_ms": round(self.latency_ms, 2),
            "llm_calls": self.usage.calls,
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "total_tokens": self.usage.total_tokens,
            "detail": self.detail,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shared retrieval types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single chunk with full provenance, normalized from a search result."""
    chunk_id: str
    doc_name: str
    page: Any
    section: str
    text: str
    score: float = 0.0
    similarity: float = 0.0
    source_query: str = ""  # which sub-query surfaced this chunk

    @classmethod
    def from_search_result(cls, r: Dict[str, Any], source_query: str = "") -> "RetrievedChunk":
        return cls(
            chunk_id=r.get("chunk_id", ""),
            doc_name=r.get("doc_name", ""),
            page=r.get("page", ""),
            section=r.get("section", ""),
            text=r.get("text", ""),
            score=float(r.get("score", 0.0) or 0.0),
            similarity=float(r.get("similarity", 0.0) or 0.0),
            source_query=source_query,
        )

    @property
    def citation_label(self) -> str:
        """Human-readable source label, e.g. 'handbook, Page 27'."""
        return f"{self.doc_name}, Page {self.page}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Agent I/O
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubQuery:
    """A focused sub-question produced by the Planner."""
    text: str
    intent: str = "general"  # e.g. prerequisites | procedure | verification | safety | rollback


@dataclass
class PlannerOutput:
    sub_queries: List[SubQuery]
    is_complex: bool
    reasoning: str = ""


@dataclass
class RetrieverOutput:
    chunks: List[RetrievedChunk]
    passes: int = 1
    coverage_notes: str = ""


@dataclass
class AssembledContext:
    """Ordered, de-duplicated context ready for the Answerer.

    `context_text` is the prompt-ready string with bracketed source tags;
    `ordered_chunks` preserves the chunks in the same order for auditing.
    """
    context_text: str
    ordered_chunks: List[RetrievedChunk]
    notes: str = ""


@dataclass
class Citation:
    """A footnote citation linking an answer claim to a source chunk."""
    marker: int                 # the N in [^N]
    chunk_id: str
    doc_name: str
    page: Any
    section: str = ""

    @property
    def label(self) -> str:
        return f"{self.doc_name}, Page {self.page}"


@dataclass
class AnswererOutput:
    answer_markdown: str
    citations: List[Citation] = field(default_factory=list)


@dataclass
class ClaimCheck:
    """Result of verifying one claim/step against the cited source."""
    claim: str
    supported: bool
    cited_chunk_id: str = ""
    reason: str = ""


@dataclass
class VerifierOutput:
    verified_answer: str
    confidence: str = "low"          # high | medium | low
    claim_checks: List[ClaimCheck] = field(default_factory=list)
    removed_claims: List[str] = field(default_factory=list)

    @property
    def supported_ratio(self) -> float:
        if not self.claim_checks:
            return 0.0
        supported = sum(1 for c in self.claim_checks if c.supported)
        return supported / len(self.claim_checks)


# ─────────────────────────────────────────────────────────────────────────────
# Final result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Unified output shape so the eval harness can score either pipeline.

    Both the traditional and agentic pipelines emit this. Fields that don't
    apply to a given pipeline (e.g. confidence for traditional) get sensible
    defaults.
    """
    query: str
    answer: str
    pipeline: str                                  # 'traditional' | 'agentic'
    confidence: str = "n/a"
    citations: List[Citation] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)
    stages: List[StageTrace] = field(default_factory=list)
    total_latency_ms: float = 0.0
    usage: LLMUsage = field(default_factory=LLMUsage)

    @property
    def retrieved_chunk_ids(self) -> List[str]:
        """For retrieval metrics (precision/recall/MRR/NDCG) in evaluate.py."""
        return [c.chunk_id for c in self.chunks]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "pipeline": self.pipeline,
            "answer": self.answer,
            "confidence": self.confidence,
            "citations": [
                {"marker": c.marker, "chunk_id": c.chunk_id,
                 "doc_name": c.doc_name, "page": c.page, "section": c.section}
                for c in self.citations
            ],
            "retrieved_chunks": self.retrieved_chunk_ids,
            "stages": [s.to_dict() for s in self.stages],
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_llm_calls": self.usage.calls,
            "total_tokens": self.usage.total_tokens,
        }
