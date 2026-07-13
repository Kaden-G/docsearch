#!/usr/bin/env python3
"""
Planner Agent
=============

Decides the retrieval strategy. A simple factual question becomes a single
sub-query; a multi-step procedure is decomposed into ordered, intent-tagged
sub-queries (prerequisites, procedure, verification, safety, rollback) so the
Retriever can hunt for each facet instead of hoping one search catches them all.

Degrades gracefully: if no LLM is available or the output won't parse, it falls
back to treating the original question as a single sub-query — so the pipeline
still runs end-to-end.
"""

from typing import List

from .base import Agent
from schemas import PlannerOutput, SubQuery


_SYSTEM = """You are the planning component of a document-search agent.
Users ask plain-English questions answered ONLY from the indexed documents.

Your job: decide whether the question needs decomposition, and if so, break it into focused sub-queries.

Guidelines:
- A simple factual lookup ("what port does X use?") needs ONE sub-query.
- A procedural/how-to question usually needs several: prerequisites, the main steps, verification, and safety/rollback if relevant.
- Each sub-query must be a standalone search string (no pronouns referring to the original question).
- Tag each sub-query with an intent: prerequisites | procedure | verification | safety | rollback | general.

Respond with ONLY valid JSON in this exact shape:
{
  "is_complex": true,
  "reasoning": "one short sentence",
  "sub_queries": [
    {"text": "...", "intent": "prerequisites"},
    {"text": "...", "intent": "procedure"}
  ]
}"""


class PlannerAgent(Agent):
    name = "planner"

    def __init__(self, toolbox, verbose: bool = False, max_sub_queries: int = 5):
        super().__init__(toolbox, verbose=verbose)
        self.max_sub_queries = max_sub_queries

    def _run(self, query: str) -> PlannerOutput:
        user = f"Question: {query}\n\nProduce the planning JSON."
        raw = self.llm(_SYSTEM, user, max_tokens=500, temperature=0.0)

        parsed = self.parse_json(raw)
        if not parsed or "sub_queries" not in parsed:
            # Fallback: single-pass plan on the raw question.
            self._detail = {"fallback": True, "num_sub_queries": 1}
            return PlannerOutput(
                sub_queries=[SubQuery(text=query, intent="general")],
                is_complex=False,
                reasoning="Planner fallback: treating question as a single query.",
            )

        sub_queries: List[SubQuery] = []
        for sq in parsed.get("sub_queries", [])[: self.max_sub_queries]:
            text = (sq.get("text") or "").strip()
            if text:
                sub_queries.append(SubQuery(text=text, intent=(sq.get("intent") or "general").strip()))

        if not sub_queries:
            sub_queries = [SubQuery(text=query, intent="general")]

        self._detail = {
            "fallback": False,
            "is_complex": bool(parsed.get("is_complex", len(sub_queries) > 1)),
            "num_sub_queries": len(sub_queries),
            "intents": [s.intent for s in sub_queries],
        }
        return PlannerOutput(
            sub_queries=sub_queries,
            is_complex=bool(parsed.get("is_complex", len(sub_queries) > 1)),
            reasoning=(parsed.get("reasoning") or "").strip(),
        )
