"""Tests for the agentic pipeline's deterministic logic.

These cover everything that does NOT require a live LLM or a built index:
JSON parsing, schema normalization, citation parsing, and the answer-level
metrics that power the agentic-vs-traditional comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from schemas import (
    RetrievedChunk, Citation, PipelineResult, LLMUsage, StageTrace,
)
from agents.base import Agent
from agents.answerer import AnswererAgent
from agents.verifier import VerifierAgent
from agents.assembler import _chunk_index, _page_key
import answer_metrics as am
from pipeline import _parse_traditional_citations


def _chunk(cid, doc="DocA", page=1, section="1.0 INTRO", text="hello world", score=0.5):
    return RetrievedChunk(chunk_id=cid, doc_name=doc, page=page, section=section,
                          text=text, score=score, similarity=score)


class TestParseJson:
    def test_direct(self):
        assert Agent.parse_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        text = "Sure!\n```json\n{\"a\": 2}\n```\nDone."
        assert Agent.parse_json(text) == {"a": 2}

    def test_balanced_span_in_prose(self):
        text = 'The plan is {"sub_queries": [{"text": "x"}]} okay?'
        out = Agent.parse_json(text)
        assert out["sub_queries"][0]["text"] == "x"

    def test_array(self):
        assert Agent.parse_json("[1, 2, 3]") == [1, 2, 3]

    def test_failure_returns_none(self):
        assert Agent.parse_json("no json here") is None
        assert Agent.parse_json("") is None


class TestRetrievedChunk:
    def test_from_search_result(self):
        r = {"chunk_id": "D_p1_c0", "doc_name": "D", "page": 1,
             "section": "S", "text": "t", "score": 0.9, "similarity": 0.8}
        c = RetrievedChunk.from_search_result(r, source_query="q")
        assert c.chunk_id == "D_p1_c0"
        assert c.score == 0.9
        assert c.source_query == "q"

    def test_citation_label(self):
        c = _chunk("D_p7_c2", doc="SMARTBOOK", page=7)
        assert c.citation_label == "SMARTBOOK, Page 7"

    def test_handles_missing_fields(self):
        c = RetrievedChunk.from_search_result({"chunk_id": "x"})
        assert c.doc_name == ""
        assert c.score == 0.0


class TestPipelineResult:
    def test_retrieved_chunk_ids(self):
        res = PipelineResult(query="q", answer="a", pipeline="agentic",
                             chunks=[_chunk("a"), _chunk("b")])
        assert res.retrieved_chunk_ids == ["a", "b"]

    def test_to_dict_shape(self):
        usage = LLMUsage(calls=3, prompt_tokens=100, completion_tokens=50)
        res = PipelineResult(query="q", answer="a", pipeline="agentic",
                             confidence="high", usage=usage,
                             chunks=[_chunk("a")],
                             citations=[Citation(1, "a", "DocA", 1, "S")])
        d = res.to_dict()
        assert d["total_llm_calls"] == 3
        assert d["total_tokens"] == 150
        assert d["retrieved_chunks"] == ["a"]
        assert d["citations"][0]["chunk_id"] == "a"


class TestLLMUsage:
    def test_record_and_add(self):
        u = LLMUsage()
        u.record(prompt_tokens=10, completion_tokens=5)
        u.record(prompt_tokens=20, completion_tokens=5)
        assert u.calls == 2
        assert u.total_tokens == 40
        other = LLMUsage(calls=1, prompt_tokens=1, completion_tokens=1)
        u.add(other)
        assert u.calls == 3
        assert u.total_tokens == 42


class TestAssemblerHelpers:
    def test_chunk_index(self):
        assert _chunk_index("Doc_p3_c12") == 12
        assert _chunk_index("no_index") == 0

    def test_page_key_orders_ints_before_strings(self):
        assert _page_key(2) < _page_key(10)
        assert _page_key(5) < _page_key("appendix")


class TestAnswererCitationParsing:
    def test_parse_with_explicit_id(self):
        agent = AnswererAgent(None)
        chunks = [_chunk("DocA_p5_c2", doc="DocA", page=5)]
        answer = ("Do the thing.[^1]\n\nReferences\n"
                  "[^1]: DocA, Page 5 (id: DocA_p5_c2)")
        cites = agent._parse_citations(answer, chunks)
        assert len(cites) == 1
        assert cites[0].chunk_id == "DocA_p5_c2"
        assert cites[0].marker == 1

    def test_parse_fallback_by_label(self):
        agent = AnswererAgent(None)
        chunks = [_chunk("DocA_p5_c2", doc="DocA", page=5)]
        answer = "Step.[^1]\n\nReferences\n[^1]: DocA, Page 5"
        cites = agent._parse_citations(answer, chunks)
        assert len(cites) == 1
        assert cites[0].chunk_id == "DocA_p5_c2"

    def test_dedupes_markers(self):
        agent = AnswererAgent(None)
        chunks = [_chunk("DocA_p5_c2", doc="DocA", page=5)]
        answer = "A.[^1] B.[^1]\n\nReferences\n[^1]: DocA, Page 5 (id: DocA_p5_c2)"
        cites = agent._parse_citations(answer, chunks)
        assert len(cites) == 1


class TestTraditionalCitationParsing:
    def test_parses_v1_style_refs(self):
        chunks = [_chunk("handbook_p27_c3", doc="handbook", page=27)]
        answer = ("Steps here.[^1]\n\nReferences\n"
                  "[^1]: handbook, Page 27")
        cites = _parse_traditional_citations(answer, chunks)
        assert len(cites) == 1
        assert cites[0].doc_name == "handbook"


class TestAnswerMetrics:
    def test_fact_coverage(self):
        res = PipelineResult(query="q", answer="You must stop the service and edit the config.",
                             pipeline="x")
        score = am.fact_coverage(res.answer, ["stop the service", "edit the config", "reboot"])
        assert abs(score - (2 / 3)) < 1e-9

    def test_citation_validity_detects_fabrication(self):
        res = PipelineResult(
            query="q", answer="a", pipeline="x",
            chunks=[_chunk("real_1")],
            citations=[Citation(1, "real_1", "DocA", 1), Citation(2, "ghost_9", "DocA", 2)],
        )
        # one of two citations points to a non-retrieved chunk
        assert am.citation_validity(res) == 0.5

    def test_grounding_flags_invented_value(self):
        res = PipelineResult(
            query="q",
            answer="Set the timeout to 30 seconds and the port to 8080.",
            pipeline="x",
            chunks=[_chunk("c1", text="Set the timeout to 30 seconds.")],
        )
        # '30' is grounded, '8080' is not -> 0.5
        assert am.grounding(res) == 0.5

    def test_grounding_nan_when_no_values(self):
        res = PipelineResult(query="q", answer="Follow the procedure carefully.",
                             pipeline="x", chunks=[_chunk("c1")])
        assert am.grounding(res) != am.grounding(res)  # NaN

    def test_abstained_detection(self):
        assert am.abstained("This is not in the indexed documents.") is True
        assert am.abstained("Here are the steps: ...") is False

    def test_must_cite_recall(self):
        res = PipelineResult(
            query="q", answer="a", pipeline="x",
            citations=[Citation(1, "c1", "RSA_SecurID_Setup", 1)],
        )
        assert am.must_cite_recall(res, ["RSA_SecurID_Setup"]) == 1.0
        assert am.must_cite_recall(res, ["Other_Doc"]) == 0.0

    def test_score_answer_out_of_scope_resilience(self):
        res = PipelineResult(query="q", answer="That is not in the documents.",
                             pipeline="x", chunks=[_chunk("c1")])
        metrics = am.score_answer(res, {"out_of_scope": True})
        assert metrics["resilience"] == 1.0


class TestVerifierDeterministic:
    def test_check_values_finds_mismatch(self):
        agent = VerifierAgent(None)
        chunks = [_chunk("c1", text="The timeout is 30 seconds.")]
        answer = "Set timeout to 30 seconds, then open port 8080."
        mismatches = agent._check_values(answer, chunks)
        assert "8080" in mismatches
        assert "30" not in mismatches

    def test_combine_confidence_low_on_value_mismatch(self):
        from schemas import ClaimCheck
        checks = [ClaimCheck("c", True), ClaimCheck("c2", True)]
        # even with all claims supported, a value mismatch forces low
        assert VerifierAgent._combine_confidence("high", checks, ["8080"]) == "low"

    def test_combine_confidence_high(self):
        from schemas import ClaimCheck
        checks = [ClaimCheck("c", True), ClaimCheck("c2", True)]
        assert VerifierAgent._combine_confidence("high", checks, []) == "high"

    def test_combine_confidence_medium(self):
        from schemas import ClaimCheck
        checks = [ClaimCheck("a", True), ClaimCheck("b", True),
                  ClaimCheck("c", True), ClaimCheck("d", False)]
        # 0.75 supported, no value mismatch -> medium
        assert VerifierAgent._combine_confidence("medium", checks, []) == "medium"


class _FakeSearcher:
    llm_provider = "openai"


class _FakeToolBox:
    """Records search/LLM calls so we can assert the retriever stays lean."""

    def __init__(self, hits):
        self._hits = hits
        self.searcher = _FakeSearcher()
        self.search_calls = 0
        self.llm_calls = 0

    def search_index(self, query, top_k=5, rerank=True):
        self.search_calls += 1
        return list(self._hits)

    def llm_complete(self, system_prompt, user_prompt, usage=None,
                     max_tokens=1500, temperature=0.0):
        self.llm_calls += 1
        if usage is not None:
            usage.record(prompt_tokens=1, completion_tokens=1)
        return '{"reformulated_query": "reformulated"}'


class TestRetrieverDeterministic:
    def _plan(self, *texts):
        from schemas import PlannerOutput, SubQuery
        return PlannerOutput(sub_queries=[SubQuery(t) for t in texts], is_complex=len(texts) > 1)

    def test_default_is_single_pass_no_llm(self):
        from agents.retriever import RetrieverAgent
        tb = _FakeToolBox([_chunk("c1", score=0.8)])  # similarity 0.8 >= gate
        out, trace = RetrieverAgent(tb).run(self._plan("q1", "q2"))
        assert tb.search_calls == 2   # exactly one search per sub-query
        assert tb.llm_calls == 0      # no self-assess / reformulation calls
        assert trace.usage.calls == 0
        assert out.passes == 2

    def test_self_assess_skips_llm_when_gate_cleared(self):
        from agents.retriever import RetrieverAgent
        tb = _FakeToolBox([_chunk("c1", score=0.9)])  # strong coverage
        agent = RetrieverAgent(tb, self_assess=True, score_gate=0.45)
        agent.run(self._plan("q1"))
        assert tb.llm_calls == 0      # deterministic gate short-circuits the LLM
        assert tb.search_calls == 1

    def test_self_assess_reformulates_when_weak(self):
        from agents.retriever import RetrieverAgent
        tb = _FakeToolBox([_chunk("c1", score=0.1)])  # weak coverage every pass
        agent = RetrieverAgent(tb, self_assess=True, max_passes=3, score_gate=0.45)
        agent.run(self._plan("q1"))
        assert tb.llm_calls >= 1      # spends reformulation calls when weak
        assert tb.search_calls >= 2
