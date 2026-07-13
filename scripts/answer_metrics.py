#!/usr/bin/env python3
"""
Answer-Level Metrics
====================

`evaluate.py` scores *retrieval* (did the right chunks come back?). This module
scores the *answer* — the thing the engineer actually reads — which is where
the agentic pipeline is supposed to pull ahead.

All metrics operate on a PipelineResult plus a labeled expectation, so they
work identically for the traditional and agentic pipelines. Nothing here calls
an LLM: the scoring is deterministic and cheap, so it can run offline in an
airgapped environment as often as you like.

Metrics
-------
fact_coverage         Fraction of `expected_facts` present in the answer text.
                      "Did it actually say the things a correct answer must say?"
must_cite_recall      Fraction of `must_cite` documents that appear in the
                      answer's citations. "Did it cite the right sources?"
citation_validity     Fraction of citations that point to a chunk actually in
                      the retrieved context. A citation to a chunk that was
                      never retrieved is a FABRICATED citation -> hallucination.
grounding             Fraction of exact value-tokens (numbers, paths, IPs) in
                      the answer that appear verbatim in the retrieved sources.
                      Low grounding is a strong hallucination signal.
abstained             True if the answer is an explicit "not in the docs".
                      Correct abstention on out-of-scope queries == resilience.
"""

import re
from typing import Dict, List, Any

from schemas import PipelineResult


_VALUE_PATTERNS = [
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),        # IPv4
    re.compile(r"[A-Za-z]:\\[^\s]+"),                    # Windows paths
    re.compile(r"(?:/[\w.-]+){2,}"),                     # unix-ish paths
    re.compile(r"\b\d+(?:\.\d+)?\b"),                    # numbers (whole tokens)
]

# Inline footnote markers like [^1] must be stripped before value extraction,
# otherwise their digits get mistaken for ungrounded values.
_FOOTNOTE_MARKER = re.compile(r"\[\^\d+\]")

_ABSTAIN_MARKERS = [
    "not in the", "could not find", "couldn't find", "no information",
    "does not contain", "doesn't contain", "not available in", "not found in",
    "unable to find", "insufficient information",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def fact_coverage(answer: str, expected_facts: List[str]) -> float:
    """Fraction of expected facts (substring/keyword) present in the answer."""
    if not expected_facts:
        return float("nan")  # not labeled -> excluded from averages
    norm = _normalize(answer)
    hit = sum(1 for fact in expected_facts if _normalize(fact) in norm)
    return hit / len(expected_facts)


def must_cite_recall(result: PipelineResult, must_cite: List[str]) -> float:
    """Fraction of required source documents that appear in the citations."""
    if not must_cite:
        return float("nan")
    cited_docs = {c.doc_name for c in result.citations}
    # also consider docs present in the answer's reference labels
    hit = 0
    for doc in must_cite:
        if any(doc in cd or cd in doc for cd in cited_docs):
            hit += 1
    return hit / len(must_cite)


def citation_validity(result: PipelineResult) -> float:
    """Fraction of citations that reference an actually-retrieved chunk.

    1.0 == every citation is real. < 1.0 == the answer cited something that was
    never in its context (a fabricated/misattributed citation).
    """
    if not result.citations:
        return float("nan")  # no citations to validate
    retrieved_ids = set(result.retrieved_chunk_ids)
    valid = sum(1 for c in result.citations if c.chunk_id in retrieved_ids)
    return valid / len(result.citations)


def grounding(result: PipelineResult) -> float:
    """Fraction of exact value-tokens in the answer that appear in the sources.

    This is the core hallucination signal: a number or path in the answer that
    does not appear anywhere in the retrieved chunks was invented.
    """
    body = (result.answer or "").split("References", 1)[0]
    body = _FOOTNOTE_MARKER.sub(" ", body)  # remove [^N] so their digits don't count
    source_norm = re.sub(r"\s+", " ", " ".join(c.text for c in result.chunks))

    values = set()
    for patt in _VALUE_PATTERNS:
        for m in patt.finditer(body):
            v = m.group(0).strip()
            if len(v) > 1:  # skip single-char noise (step/list markers)
                values.add(v)

    if not values:
        return float("nan")  # nothing to ground
    grounded = sum(1 for v in values if v in source_norm)
    return grounded / len(values)


def abstained(answer: str) -> bool:
    """Did the answer explicitly decline because the info isn't in the docs?"""
    norm = _normalize(answer)
    return any(marker in norm for marker in _ABSTAIN_MARKERS)


def score_answer(result: PipelineResult, label: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all answer-level metrics for one (result, label) pair.

    `label` may contain: expected_facts, must_cite, out_of_scope (bool).
    """
    expected_facts = label.get("expected_facts", []) or []
    must_cite = label.get("must_cite", []) or []
    out_of_scope = bool(label.get("out_of_scope", False))

    did_abstain = abstained(result.answer)

    metrics = {
        "fact_coverage": fact_coverage(result.answer, expected_facts),
        "must_cite_recall": must_cite_recall(result, must_cite),
        "citation_validity": citation_validity(result),
        "grounding": grounding(result),
        "abstained": did_abstain,
        "answer_chars": len(result.answer or ""),
        # cost / scalability axis (straight from instrumentation)
        "llm_calls": result.usage.calls,
        "total_tokens": result.usage.total_tokens,
        "latency_ms": result.total_latency_ms,
        "confidence": result.confidence,
    }

    # Resilience: on an out-of-scope query, the *correct* behavior is to abstain.
    if out_of_scope:
        metrics["resilience"] = 1.0 if did_abstain else 0.0

    return metrics
