#!/usr/bin/env python3
"""
Chunk Enrichment ("Smart Indexing")
===================================

A build-time, one-time enrichment stage that moves work off the query-time hot
path. For each chunk it asks the LLM for:

    * a short factual summary, and
    * a handful of synthetic questions the chunk answers (doc2query-style),

then builds a contextual ``embedding_text`` ("Doc > Section" + summary +
questions + the original text). Embedding that enriched text lifts first-pass
retrieval recall, which lets the query-time retriever stop after one pass
(fewer LLM round-trips) — the whole point of Smart Indexing.

The original ``text`` is never modified; it is still what gets displayed and
fed to the answer synthesizer.

The enricher is provider-agnostic: it takes an ``llm(system, user) -> str``
callable, so it is trivially testable with a fake. ``make_openai_llm`` is a
convenience factory for the OpenAI-compatible gateway the app already uses.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

_SYSTEM = """You enrich a document excerpt for a search index.
Given the excerpt, respond with ONLY valid JSON:
{"summary": "1-2 sentence factual summary", "questions": ["a question this excerpt answers", "..."]}
Provide 3-5 questions. Use ONLY information present in the excerpt; never invent details."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from a model response."""
    if not raw:
        return None
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def build_embedding_text(
    doc_name: str,
    section: str,
    summary: str,
    questions: List[str],
    text: str,
) -> str:
    """Compose the contextual text that gets embedded (display text is separate)."""
    parts: List[str] = []
    header = (doc_name or "").strip()
    section = (section or "").strip()
    if section:
        header = f"{header} > {section}" if header else section
    if header:
        parts.append(header)
    if summary:
        parts.append(summary)
    if questions:
        parts.append("Questions answered: " + " ".join(questions))
    parts.append(text or "")
    return "\n".join(p for p in parts if p)


def make_openai_llm(
    api_key: str,
    api_base: str,
    model: str,
    timeout: float = 180.0,
    max_retries: int = 1,
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> Callable[[str, str], str]:
    """Return an ``llm(system, user) -> str`` backed by the OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=api_base or None,
        timeout=timeout,
        max_retries=max_retries,
    )

    def _call(system_prompt: str, user_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _call


class ChunkEnricher:
    """Enriches chunks.json in place with summaries, questions, and embedding_text."""

    def __init__(
        self,
        processed_dir: str,
        llm: Callable[[str, str], str],
        max_workers: int = None,
        max_questions: int = 5,
    ):
        self.processed_dir = Path(processed_dir)
        self.llm = llm
        env_workers = os.environ.get("DOCSEARCH_ENRICH_WORKERS", "")
        self.max_workers = max_workers or (int(env_workers) if env_workers.isdigit() else 6)
        self.max_questions = max_questions

    def load_chunks(self) -> List[Dict]:
        chunks_file = self.processed_dir / "chunks.json"
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        with open(chunks_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_chunks(self, chunks: List[Dict]):
        chunks_file = self.processed_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

    def enrich_chunk(self, chunk: Dict) -> Dict:
        """Enrich a single chunk in place. Never raises — degrades to no enrichment."""
        text = chunk.get("text", "")
        summary, questions = "", []
        try:
            raw = self.llm(_SYSTEM, f"document excerpt:\n{text}\n\nEnrichment JSON:")
            parsed = _parse_json(raw) or {}
            summary = (parsed.get("summary") or "").strip()
            raw_qs = parsed.get("questions") or []
            if isinstance(raw_qs, list):
                questions = [str(q).strip() for q in raw_qs if str(q).strip()][: self.max_questions]
        except Exception as e:  # noqa: BLE001 - enrichment is best-effort
            print(f"[enrich] chunk {chunk.get('chunk_id')} failed (non-fatal): {e}")

        chunk["summary"] = summary
        chunk["questions"] = questions
        chunk["embedding_text"] = build_embedding_text(
            chunk.get("doc_name", ""),
            chunk.get("section", ""),
            summary,
            questions,
            text,
        )
        return chunk

    def run(self, progress_callback=None) -> List[Dict]:
        """Enrich every chunk (concurrently), persist, and return the chunks."""
        chunks = self.load_chunks()
        total = len(chunks)
        done = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.enrich_chunk, c) for c in chunks]
            for _ in as_completed(futures):
                done += 1
                if progress_callback:
                    progress_callback(done, total)

        self.save_chunks(chunks)
        print(f"Enrichment complete: {total} chunks enriched")
        return chunks
