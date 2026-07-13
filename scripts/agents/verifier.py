#!/usr/bin/env python3
"""
Verifier Agent
==============

The zero-hallucination gate. Nothing reaches the engineer until it has been
checked against the source chunks. This is the agent that most directly serves
the project's hard requirement: no hallucinations, citations required.

Hybrid verification (as planned):
    1. LLM claim-level audit  — break the answer into atomic claims, confirm
       each is DIRECTLY supported by the cited source, and rewrite the answer
       with unsupported claims removed.
    2. Deterministic value check — extract exact values (numbers, paths,
       commands, IPs) from the answer and confirm each appears verbatim in the
       source text. This catches the classic failure where an LLM paraphrases a
       number incorrectly, independent of the LLM's own judgment.

Confidence is the combination: high (all claims supported, all values found),
medium (minor gaps), low (major gaps — advise the engineer to consult sources).
"""

import re
from typing import Dict, List

from .base import Agent
from schemas import AnswererOutput, VerifierOutput, ClaimCheck, RetrievedChunk


_SYSTEM = """You are a strict fact-checker for document-based answers. You receive an ANSWER (with footnote
citations) and the SOURCE excerpts it cites. Your job is to ensure every claim is DIRECTLY supported
by the sources — no outside knowledge, no inference beyond what the text states.

For each atomic claim or step in the answer:
- Mark it supported ONLY if the cited source text directly states it.
- If a claim is unsupported, partially supported, or cites the wrong source, mark it unsupported.

Then produce a corrected answer that REMOVES unsupported claims (keep formatting and citations for the
supported parts). If you remove something, the remaining answer must still read coherently.

Respond with ONLY valid JSON:
{
  "claims": [
    {"claim": "short text of the claim", "supported": true, "cited_chunk_id": "id or ''", "reason": "why"}
  ],
  "verified_answer": "the corrected Markdown answer",
  "confidence": "high|medium|low"
}"""


# Exact-value patterns we insist must appear verbatim in the sources.
_VALUE_PATTERNS = [
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),          # IPv4
    re.compile(r"[A-Za-z]:\\[^\s]+"),                      # Windows paths
    re.compile(r"(?:/[\w.-]+){2,}"),                       # unix-ish paths
    re.compile(r"\b\d+(?:\.\d+)?\b"),                      # numbers (whole tokens)
]

# Inline footnote markers like [^1] are stripped before value extraction.
_FOOTNOTE_MARKER = re.compile(r"\[\^\d+\]")


class VerifierAgent(Agent):
    name = "verifier"

    def __init__(self, toolbox, verbose: bool = False, value_check: bool = True):
        super().__init__(toolbox, verbose=verbose)
        self.value_check = value_check

    def _run(self, answer: AnswererOutput, source_chunks: List[RetrievedChunk]) -> VerifierOutput:
        if not answer.answer_markdown or not source_chunks:
            self._detail = {"skipped": True}
            return VerifierOutput(
                verified_answer=answer.answer_markdown,
                confidence="low",
                claim_checks=[],
                removed_claims=[],
            )

        source_blob = "\n\n".join(
            f"[id: {c.chunk_id} | {c.doc_name} p{c.page}]\n{c.text}" for c in source_chunks
        )
        user = (
            f"ANSWER:\n{answer.answer_markdown}\n\n"
            f"SOURCES:\n{source_blob}\n\n"
            "Return the verification JSON."
        )
        raw = self.llm(_SYSTEM, user, max_tokens=2000, temperature=0.0)
        parsed = self.parse_json(raw)

        if not parsed:
            # If verification itself fails to parse, be conservative: keep the
            # answer but flag low confidence so the UI warns the user.
            self._detail = {"parse_failed": True}
            return VerifierOutput(
                verified_answer=answer.answer_markdown,
                confidence="low",
                claim_checks=[],
                removed_claims=[],
            )

        claim_checks: List[ClaimCheck] = []
        removed: List[str] = []
        for c in parsed.get("claims", []):
            check = ClaimCheck(
                claim=(c.get("claim") or "").strip(),
                supported=bool(c.get("supported", False)),
                cited_chunk_id=(c.get("cited_chunk_id") or "").strip(),
                reason=(c.get("reason") or "").strip(),
            )
            claim_checks.append(check)
            if not check.supported and check.claim:
                removed.append(check.claim)

        verified_answer = (parsed.get("verified_answer") or answer.answer_markdown).strip()
        llm_confidence = (parsed.get("confidence") or "low").strip().lower()

        # Deterministic exact-value cross-check against the sources.
        value_mismatches = []
        if self.value_check:
            value_mismatches = self._check_values(verified_answer, source_chunks)

        confidence = self._combine_confidence(llm_confidence, claim_checks, value_mismatches)

        self._detail = {
            "num_claims": len(claim_checks),
            "supported": sum(1 for c in claim_checks if c.supported),
            "removed": len(removed),
            "value_mismatches": value_mismatches[:10],
            "llm_confidence": llm_confidence,
            "final_confidence": confidence,
        }
        return VerifierOutput(
            verified_answer=verified_answer,
            confidence=confidence,
            claim_checks=claim_checks,
            removed_claims=removed,
        )

    def _check_values(self, answer: str, chunks: List[RetrievedChunk]) -> List[str]:
        """Return exact values present in the answer but absent from all sources."""
        source_text = " ".join(c.text for c in chunks)
        # Normalize whitespace for robust substring checks.
        source_norm = re.sub(r"\s+", " ", source_text)

        mismatches = []
        candidates = set()
        # Only check values inside the prose, not inside the References section.
        body = answer.split("References", 1)[0]
        body = _FOOTNOTE_MARKER.sub(" ", body)  # drop [^N] markers
        for patt in _VALUE_PATTERNS:
            for m in patt.finditer(body):
                val = m.group(0).strip()
                # Skip single-char noise (step/list markers like "1").
                if len(val) <= 1:
                    continue
                candidates.add(val)

        for val in candidates:
            if val not in source_norm:
                mismatches.append(val)
        return mismatches

    @staticmethod
    def _combine_confidence(llm_confidence: str, claim_checks: List[ClaimCheck],
                            value_mismatches: List[str]) -> str:
        """Blend the LLM's confidence with the deterministic checks.

        Deterministic value mismatches are treated as a strong negative signal —
        a wrong number in a procedure is exactly the failure mode we cannot
        tolerate.
        """
        if claim_checks:
            ratio = sum(1 for c in claim_checks if c.supported) / len(claim_checks)
        else:
            ratio = 0.0

        if value_mismatches:
            return "low"
        if ratio >= 0.95 and llm_confidence == "high":
            return "high"
        if ratio >= 0.75:
            return "medium"
        return "low"
