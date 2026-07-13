#!/usr/bin/env python3
"""
Agent Tools
===========

The bridge between the agentic pipeline and the *shared backend*. Every agent
reaches the index and the LLM through this module, so the agentic pipeline and
the traditional pipeline use the identical retrieval stack and LLM provider —
the only thing that differs is orchestration. That is what keeps the
agentic-vs-traditional comparison fair.

Two responsibilities:
    1. Retrieval tools  -> thin wrappers over DocSearcher.
    2. LLM completion   -> ONE provider-agnostic call that also records token
                           usage, so we can measure the cost of being agentic.
"""

from typing import List, Dict, Any, Optional, Tuple

from schemas import LLMUsage, RetrievedChunk


class ToolBox:
    """Provides retrieval + LLM tools to agents, backed by a single DocSearcher.

    The ToolBox deliberately disables HyDE for the agentic path's retrieval
    calls (use_hyde=False), because the agentic Retriever does its own query
    reformulation — so query-time HyDE would be a redundant LLM round-trip.
    (The traditional pipeline keeps HyDE via DocSearcher.search.)
    """

    def __init__(self, searcher):
        self.searcher = searcher

    # ── Retrieval tools ──────────────────────────────────────────────────────

    def search_index(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
    ) -> List[RetrievedChunk]:
        """Run hybrid (FAISS + BM25 + rerank) search and normalize results.

        HyDE is disabled here: the Retriever reformulates queries itself, so a
        per-search HyDE LLM call would be redundant latency.
        """
        raw = self.searcher.search(query, top_k=top_k, rerank=rerank, use_hyde=False)
        return [RetrievedChunk.from_search_result(r, source_query=query) for r in raw]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[RetrievedChunk]:
        """Fetch a single chunk by id straight from the index metadata."""
        for c in self.searcher.metadata:
            if c.get("chunk_id") == chunk_id:
                return RetrievedChunk.from_search_result(c)
        return None

    def get_document_list(self) -> List[str]:
        return self.searcher.get_document_list()

    def get_document_sections(self, doc_name: str) -> List[str]:
        """Distinct section headers for a document — used to resolve
        cross-references like 'see Section 3.2'."""
        seen = []
        for c in self.searcher.metadata:
            if c.get("doc_name") == doc_name:
                section = c.get("section", "")
                if section and section not in seen:
                    seen.append(section)
        return seen

    def chunks_for_section(self, doc_name: str, section: str) -> List[RetrievedChunk]:
        """All chunks in a given document/section, in index order."""
        out = []
        for c in self.searcher.metadata:
            if c.get("doc_name") == doc_name and c.get("section") == section:
                out.append(RetrievedChunk.from_search_result(c))
        return out

    # ── LLM tool (provider-agnostic + instrumented) ──────────────────────────

    def llm_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        usage: Optional[LLMUsage] = None,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        """Provider-agnostic chat completion that records token usage.

        Routes to whatever provider the shared DocSearcher is configured with
        (openai / openai / anthropic / ollama). Increments `usage` so the
        orchestrator can report LLM calls + tokens per stage.

        Returns the model's text output, or a string prefixed with
        '[LLM Error]' on failure so the pipeline can degrade gracefully.
        """
        s = self.searcher
        provider = s.llm_provider

        if not provider:
            return "[LLM Error] No LLM provider configured."

        try:
            if provider in ("openai",) and s._openai_client is not None:
                text, pt, ct = self._complete_openai(
                    system_prompt, user_prompt, max_tokens, temperature
                )
            elif provider == "anthropic" and s._anthropic_client is not None:
                text, pt, ct = self._complete_anthropic(
                    system_prompt, user_prompt, max_tokens, temperature
                )
            elif provider == "ollama" and s.use_ollama:
                text, pt, ct = self._complete_ollama(
                    system_prompt, user_prompt, max_tokens, temperature
                )
            else:
                return f"[LLM Error] Provider '{provider}' not available."
        except Exception as e:  # noqa: BLE001 - we want graceful degradation
            print(f"[ToolBox.llm_complete] {provider} call failed: {e}")
            return f"[LLM Error] {e}"

        if usage is not None:
            usage.record(prompt_tokens=pt, completion_tokens=ct)
        return text

    # ── Provider-specific helpers (return text + token counts) ───────────────

    def _complete_openai(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> Tuple[str, int, int]:
        s = self.searcher
        resp = s._openai_client.chat.completions.create(
            model=s._llm_model or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        pt = ct = 0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
        return text, pt, ct

    def _complete_anthropic(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> Tuple[str, int, int]:
        s = self.searcher
        resp = s._anthropic_client.messages.create(
            model=s._llm_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        pt = ct = 0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            pt = getattr(usage, "input_tokens", 0) or 0
            ct = getattr(usage, "output_tokens", 0) or 0
        return text, pt, ct

    def _complete_ollama(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> Tuple[str, int, int]:
        import ollama
        s = self.searcher
        resp = ollama.chat(
            model=s.ollama_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        text = resp["message"]["content"]
        # Ollama reports eval counts; fall back to 0 when absent.
        pt = int(resp.get("prompt_eval_count", 0) or 0)
        ct = int(resp.get("eval_count", 0) or 0)
        return text, pt, ct
