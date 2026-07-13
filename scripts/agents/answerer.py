#!/usr/bin/env python3
"""
Answerer Agent
==============

Generates the step-by-step response from the assembled context. The prompt is
intentionally aligned with the traditional pipeline's prompt (same Markdown +
footnote citation style) so that any quality difference in the comparison comes
from the *agentic orchestration*, not from prompt divergence.

Citations use numbered footnotes ([^1]) with a trailing References section, and
we parse those references back into Citation objects so the Verifier and the
eval harness can check them against the source chunks.
"""

import re
from typing import Dict, List

from .base import Agent
from schemas import AssembledContext, AnswererOutput, Citation, RetrievedChunk


_SYSTEM = """You answer questions using ONLY the provided document excerpts. Never use outside knowledge.

Formatting requirements:
- Respond in **Markdown**.
- For procedural questions, give complete, ordered **step-by-step instructions** with every detail needed.
- Include exact values, commands, paths, and settings verbatim from the source — never invent or approximate.

Citation requirements:
- After each claim or step, add a footnote marker like [^1] pointing to the source it came from.
- End with a "References" section mapping each marker to its source, using the EXACT id shown in the
  source tag, e.g.:
  [^1]: handbook, Page 27 (id: handbook_p27_c3)
- Reuse the same marker number when citing the same source again.
- EVERY factual claim or step MUST carry at least one footnote.

If the context does not contain the answer, say so explicitly. Do not guess."""


# Parses: [^1]: Doc, Page 5 (id: Doc_p5_c2)
_REF_LINE = re.compile(
    r"\[\^(\d+)\]:\s*(.+?)(?:\(id:\s*([^)]+)\))?\s*$",
    re.MULTILINE,
)


class AnswererAgent(Agent):
    name = "answerer"

    def __init__(self, toolbox, verbose: bool = False, max_tokens: int = 2000):
        super().__init__(toolbox, verbose=verbose)
        self.max_tokens = max_tokens

    def _run(self, query: str, context: AssembledContext) -> AnswererOutput:
        if not context.ordered_chunks:
            self._detail = {"no_context": True}
            return AnswererOutput(
                answer_markdown="I could not find information about this in the indexed documents.",
                citations=[],
            )

        user = (
            f"Context from documents:\n{context.context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer the question thoroughly with footnote citations."
        )
        answer = self.llm(_SYSTEM, user, max_tokens=self.max_tokens, temperature=0.0)

        citations = self._parse_citations(answer, context.ordered_chunks)
        self._detail = {
            "answer_chars": len(answer),
            "num_citations": len(citations),
        }
        return AnswererOutput(answer_markdown=answer, citations=citations)

    def _parse_citations(self, answer: str, chunks: List[RetrievedChunk]) -> List[Citation]:
        """Extract footnote references and resolve them to source chunks.

        We match the explicit `(id: ...)` when the model provides it; otherwise
        we best-effort match on doc name + page parsed from the reference line.
        """
        by_id: Dict[str, RetrievedChunk] = {c.chunk_id: c for c in chunks}
        citations: List[Citation] = []
        seen_markers = set()

        for m in _REF_LINE.finditer(answer):
            marker = int(m.group(1))
            if marker in seen_markers:
                continue
            label = (m.group(2) or "").strip().rstrip("(").strip()
            chunk_id = (m.group(3) or "").strip()

            chunk = by_id.get(chunk_id)
            if chunk is None:
                chunk = self._match_by_label(label, chunks)

            if chunk is not None:
                citations.append(Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    page=chunk.page,
                    section=chunk.section,
                ))
                seen_markers.add(marker)
        return citations

    @staticmethod
    def _match_by_label(label: str, chunks: List[RetrievedChunk]):
        """Fallback: match a 'Doc, Page N' label to a chunk."""
        page_m = re.search(r"page\s+(\w+)", label, re.IGNORECASE)
        page = page_m.group(1) if page_m else None
        for c in chunks:
            if c.doc_name and c.doc_name in label:
                if page is None or str(c.page) == str(page):
                    return c
        return None
