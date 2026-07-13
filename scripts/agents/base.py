#!/usr/bin/env python3
"""
Agent Base Class
================

Common machinery shared by every agent:

    * a handle to the ToolBox (retrieval + instrumented LLM)
    * a `run()` wrapper that times the agent and accumulates a StageTrace
    * robust JSON parsing for structured LLM outputs

Agents implement `_run(...)` and return their typed output; the public `run`
handles timing and trace bookkeeping so the orchestrator gets consistent
instrumentation for the comparison.
"""

import json
import re
import time
from typing import Any, Dict, Optional

from schemas import StageTrace, LLMUsage


class Agent:
    """Base class for pipeline agents."""

    name: str = "agent"

    def __init__(self, toolbox, verbose: bool = False):
        self.tools = toolbox
        self.verbose = verbose

    # ── Public entrypoint (instrumented) ─────────────────────────────────────

    def run(self, *args, **kwargs):
        """Execute the agent, returning (output, StageTrace).

        Token usage is captured by passing a fresh LLMUsage into the agent's
        LLM calls; the agent must thread `self._usage` through any
        `tools.llm_complete(...)` calls it makes.
        """
        self._usage = LLMUsage()
        self._detail: Dict[str, Any] = {}
        start = time.time()
        output = self._run(*args, **kwargs)
        latency_ms = (time.time() - start) * 1000.0

        trace = StageTrace(
            name=self.name,
            latency_ms=latency_ms,
            usage=self._usage,
            detail=self._detail,
        )
        if self.verbose:
            print(f"  [{self.name}] {latency_ms:.0f}ms, "
                  f"{self._usage.calls} LLM call(s), "
                  f"{self._usage.total_tokens} tokens")
        return output, trace

    def _run(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError

    # ── LLM helper ───────────────────────────────────────────────────────────

    def llm(self, system_prompt: str, user_prompt: str,
            max_tokens: int = 1500, temperature: float = 0.0) -> str:
        """Instrumented LLM call. Usage is folded into this stage's trace."""
        return self.tools.llm_complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            usage=self._usage,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ── Structured-output parsing ────────────────────────────────────────────

    @staticmethod
    def parse_json(text: str) -> Optional[Any]:
        """Best-effort JSON extraction from an LLM response.

        LLMs often wrap JSON in prose or ```json fences. We try, in order:
        a direct parse, a fenced-block parse, then the first balanced {...}
        or [...] span. Returns None if nothing parses.
        """
        if not text:
            return None

        # 1) Direct
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2) Fenced ```json ... ``` (or plain ``` ... ```)
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3) First balanced object/array span
        span = Agent._first_balanced_span(text)
        if span is not None:
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _first_balanced_span(text: str) -> Optional[str]:
        """Return the first balanced {...} or [...] substring, or None."""
        starts = {"{": "}", "[": "]"}
        for i, ch in enumerate(text):
            if ch in starts:
                close = starts[ch]
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == ch:
                        depth += 1
                    elif text[j] == close:
                        depth -= 1
                        if depth == 0:
                            return text[i:j + 1]
                break
        return None
