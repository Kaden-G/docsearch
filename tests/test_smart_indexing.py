"""Tests for Smart Indexing: scripts/enrich.py + embed.py embedding_text selection."""

import json
import sys
from pathlib import Path

import numpy as np

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import embed as embed_mod
from enrich import ChunkEnricher, build_embedding_text, _parse_json


def _write_chunks(tmp_path, chunks):
    (tmp_path / 'chunks.json').write_text(json.dumps(chunks), encoding='utf-8')


class TestBuildEmbeddingText:
    def test_full_composition(self):
        out = build_embedding_text('DocA', 'Section 1', 'A summary.', ['Q1?', 'Q2?'], 'Body text.')
        assert 'DocA > Section 1' in out
        assert 'A summary.' in out
        assert 'Q1?' in out and 'Q2?' in out
        assert out.strip().endswith('Body text.')

    def test_missing_section_and_enrichment(self):
        out = build_embedding_text('DocA', '', '', [], 'Body only.')
        assert out.startswith('DocA')
        assert out.strip().endswith('Body only.')
        assert 'Questions answered' not in out

    def test_no_header_keeps_summary_and_text(self):
        out = build_embedding_text('', '', 'sum', [], 'text')
        assert 'sum' in out and 'text' in out


class TestParseJson:
    def test_plain_object(self):
        assert _parse_json('{"a": 1}') == {'a': 1}

    def test_embedded_in_prose(self):
        raw = 'Sure!\n{"summary": "x", "questions": ["q"]}\nDone'
        parsed = _parse_json(raw)
        assert parsed['summary'] == 'x'
        assert parsed['questions'] == ['q']

    def test_garbage_returns_none(self):
        assert _parse_json('no json here') is None
        assert _parse_json('') is None


def _fake_llm(system, user):
    return '{"summary": "Concise summary.", "questions": ["How to X?", "What is Y?"]}'


class TestChunkEnricher:
    def test_enriches_and_persists(self, tmp_path):
        chunks = [
            {'chunk_id': 'c1', 'doc_name': 'DocA', 'section': 'S1', 'page': 1, 'text': 'Alpha body.'},
            {'chunk_id': 'c2', 'doc_name': 'DocA', 'section': 'S2', 'page': 2, 'text': 'Beta body.'},
        ]
        _write_chunks(tmp_path, chunks)
        progress = []
        enr = ChunkEnricher(str(tmp_path), _fake_llm, max_workers=2)
        result = enr.run(progress_callback=lambda d, t: progress.append((d, t)))

        assert len(result) == 2
        for c in result:
            assert c['summary'] == 'Concise summary.'
            assert c['questions'] == ['How to X?', 'What is Y?']
            assert c['text'] in c['embedding_text']      # original text preserved
            assert 'How to X?' in c['embedding_text']     # questions folded in

        assert progress and progress[-1] == (2, 2)

        on_disk = json.loads((tmp_path / 'chunks.json').read_text(encoding='utf-8'))
        assert on_disk[0]['summary'] == 'Concise summary.'
        assert 'embedding_text' in on_disk[0]

    def test_max_questions_cap(self, tmp_path):
        def many_q(system, user):
            return json.dumps({'summary': 's', 'questions': [f'Q{i}?' for i in range(10)]})
        _write_chunks(tmp_path, [{'chunk_id': 'c1', 'doc_name': 'D', 'section': '', 'page': 1, 'text': 't'}])
        result = ChunkEnricher(str(tmp_path), many_q, max_workers=1, max_questions=5).run()
        assert len(result[0]['questions']) == 5

    def test_graceful_degradation_on_llm_error(self, tmp_path):
        def boom(system, user):
            raise RuntimeError('gateway down')
        _write_chunks(tmp_path, [{'chunk_id': 'c1', 'doc_name': 'D', 'section': 'S', 'page': 1, 'text': 'Body.'}])
        result = ChunkEnricher(str(tmp_path), boom, max_workers=1).run()
        assert result[0]['summary'] == ''
        assert result[0]['questions'] == []
        assert 'Body.' in result[0]['embedding_text']     # still embeddable

    def test_bad_json_degrades(self, tmp_path):
        _write_chunks(tmp_path, [{'chunk_id': 'c1', 'doc_name': 'D', 'section': '', 'page': 1, 'text': 'Body.'}])
        result = ChunkEnricher(str(tmp_path), lambda s, u: 'not json', max_workers=1).run()
        assert result[0]['summary'] == ''
        assert result[0]['questions'] == []


class _FakeEmbedder:
    name = 'fake'
    provider = 'fake'

    def __init__(self):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen.extend(texts)
        return np.zeros((len(texts), 4), dtype='float32')


class TestEmbeddingTextSelection:
    def test_prefers_embedding_text_with_fallback(self, tmp_path, monkeypatch):
        fake = _FakeEmbedder()
        monkeypatch.setattr(embed_mod, 'get_embedder', lambda **kw: fake)
        indexer = embed_mod.EmbeddingIndexer(str(tmp_path), str(tmp_path))
        chunks = [
            {'chunk_id': 'c1', 'doc_name': 'D', 'page': 1, 'section': 'S', 'text': 'RAW1', 'embedding_text': 'ENRICHED1'},
            {'chunk_id': 'c2', 'doc_name': 'D', 'page': 2, 'section': 'S', 'text': 'RAW2'},
        ]
        indexer.create_embeddings(chunks)
        assert 'ENRICHED1' in fake.seen   # enriched text used when present
        assert 'RAW1' not in fake.seen     # raw text skipped when enriched present
        assert 'RAW2' in fake.seen         # falls back to raw text when absent
