#!/usr/bin/env python3
"""
Assembler Agent
===============

Turns a flat bag of retrieved chunks into a coherent, ordered context package
for the Answerer. This is deliberately deterministic (no LLM call) — it's pure
bookkeeping, so it adds no token cost and no latency variance to the pipeline.

Responsibilities:
    * Group chunks by document and order them by page, then by chunk index, so
      procedural steps appear in their natural sequence.
    * Detect intra-document cross-references ("see Section 3.2") and pull in the
      referenced section's chunks if we don't already have them.
    * Cap the context to a token-ish budget (by characters) so we don't blow the
      Answerer's context window.
    * Emit a prompt-ready `context_text` where every block is tagged with its
      source, plus the ordered chunk list for auditing/citation.
"""

import re
from typing import Dict, List

from .base import Agent
from schemas import RetrieverOutput, AssembledContext, RetrievedChunk


# Matches "Section 3.2", "section 4", "see 2.1.3" style references.
_XREF = re.compile(r"\b(?:see\s+)?section\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)


def _chunk_index(chunk_id: str) -> int:
    """Extract the trailing _cN index from a chunk_id for stable ordering."""
    m = re.search(r"_c(\d+)$", chunk_id or "")
    return int(m.group(1)) if m else 0


def _page_key(page) -> tuple:
    """Sort key that tolerates int or str page values."""
    try:
        return (0, int(page))
    except (TypeError, ValueError):
        return (1, str(page))


class AssemblerAgent(Agent):
    name = "assembler"

    def __init__(self, toolbox, verbose: bool = False,
                 max_context_chars: int = 12000, resolve_xrefs: bool = True):
        super().__init__(toolbox, verbose=verbose)
        self.max_context_chars = max_context_chars
        self.resolve_xrefs = resolve_xrefs

    def _run(self, retrieved: RetrieverOutput) -> AssembledContext:
        chunks = list(retrieved.chunks)
        added_xrefs = 0

        if self.resolve_xrefs and chunks:
            chunks, added_xrefs = self._resolve_cross_references(chunks)

        # Order: by document, then page, then chunk index within page.
        chunks.sort(key=lambda c: (c.doc_name, _page_key(c.page), _chunk_index(c.chunk_id)))

        # Build the prompt-ready context, respecting the char budget.
        blocks: List[str] = []
        used = 0
        ordered: List[RetrievedChunk] = []
        for c in chunks:
            block = (f"[Source: {c.doc_name}, Page {c.page}, "
                     f"Section: {c.section or 'N/A'}, id: {c.chunk_id}]\n{c.text}")
            if used + len(block) > self.max_context_chars and ordered:
                break
            blocks.append(block)
            ordered.append(c)
            used += len(block)

        context_text = "\n\n".join(blocks)
        self._detail = {
            "input_chunks": len(retrieved.chunks),
            "xref_chunks_added": added_xrefs,
            "chunks_in_context": len(ordered),
            "context_chars": used,
        }
        return AssembledContext(
            context_text=context_text,
            ordered_chunks=ordered,
            notes=f"Assembled {len(ordered)} chunks ({added_xrefs} via cross-reference).",
        )

    def _resolve_cross_references(self, chunks: List[RetrievedChunk]):
        """Pull in chunks for sections referenced by the retrieved text."""
        have_ids = {c.chunk_id for c in chunks}
        # Map a section number prefix -> full section header within each doc.
        added = 0
        extra: List[RetrievedChunk] = []

        for c in chunks:
            for m in _XREF.finditer(c.text or ""):
                sec_num = m.group(1)
                # Find sections in the same document whose header starts with sec_num.
                for section in self.tools.get_document_sections(c.doc_name):
                    if section.strip().startswith(sec_num):
                        for sc in self.tools.chunks_for_section(c.doc_name, section):
                            if sc.chunk_id not in have_ids:
                                have_ids.add(sc.chunk_id)
                                sc.source_query = f"xref:{sec_num}"
                                extra.append(sc)
                                added += 1
        return chunks + extra, added
