"""Agentic RAG agents for DocSearch v2.

Each agent is a single-responsibility unit that reads a typed input and
produces a typed output (see scripts/schemas.py). The orchestrator in
scripts/pipeline.py chains them together and collects per-stage
instrumentation for the agentic-vs-traditional comparison.
"""

from .base import Agent
from .planner import PlannerAgent
from .retriever import RetrieverAgent
from .assembler import AssemblerAgent
from .answerer import AnswererAgent
from .verifier import VerifierAgent

__all__ = [
    "Agent",
    "PlannerAgent",
    "RetrieverAgent",
    "AssemblerAgent",
    "AnswererAgent",
    "VerifierAgent",
]
