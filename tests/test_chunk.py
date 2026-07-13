"""Tests for scripts/chunk.py — section detection, char offsets, chunking logic."""

import sys
from pathlib import Path

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from chunk import DocumentChunker


def _make_chunker(tmp_path):
    """Create a chunker pointed at a temp directory."""
    return DocumentChunker(processed_dir=str(tmp_path), chunk_size=50, overlap=10)


class TestDetectSectionHeaders:
    def test_numbered_header(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = "Some intro text\n1.0 PURPOSE AND SCOPE\nBody here"
        headers = chunker.detect_section_headers(text)
        assert len(headers) == 1
        assert headers[0][1] == "1.0 PURPOSE AND SCOPE"

    def test_all_caps_header(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = "intro\nSAFETY REQUIREMENTS\nbody"
        headers = chunker.detect_section_headers(text)
        assert len(headers) == 1
        assert "SAFETY REQUIREMENTS" in headers[0][1]

    def test_step_header(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = "STEP 1: Turn on the system\nDetails here"
        headers = chunker.detect_section_headers(text)
        assert len(headers) == 1

    def test_char_offset_is_correct(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        line1 = "intro text here"
        line2 = "1.0 HEADER LINE"
        text = f"{line1}\n{line2}\nmore text"
        headers = chunker.detect_section_headers(text)
        assert len(headers) == 1
        # Offset should be len(line1) + 1 (for the newline)
        assert headers[0][0] == len(line1) + 1

    def test_no_headers(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = "just some plain text\nnothing special here"
        headers = chunker.detect_section_headers(text)
        assert headers == []

    def test_multiple_headers(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = "1.0 FIRST SECTION\nbody\n2.0 SECOND SECTION\nbody"
        headers = chunker.detect_section_headers(text)
        assert len(headers) == 2
        assert headers[0][0] < headers[1][0]


class TestChunkText:
    def test_basic_chunking(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = " ".join(f"word{i}" for i in range(100))
        metadata = {'doc_name': 'test_doc', 'page': 1}
        chunks = chunker.chunk_text(text, metadata)
        assert len(chunks) > 0
        assert all(c['doc_name'] == 'test_doc' for c in chunks)

    def test_char_offsets_are_valid(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = "The quick brown fox jumps over the lazy dog and many more words follow in this sentence to ensure chunking"
        metadata = {'doc_name': 'test_doc', 'page': 1}
        chunks = chunker.chunk_text(text, metadata)
        for chunk in chunks:
            assert chunk['char_start'] >= 0
            assert chunk['char_end'] > chunk['char_start']
            assert chunk['char_end'] <= len(text)

    def test_section_assignment_uses_char_offsets(self, tmp_path):
        chunker = DocumentChunker(processed_dir=str(tmp_path), chunk_size=100, overlap=20)
        text = "1.0 INTRODUCTION\n" + " ".join(f"word{i}" for i in range(50)) + "\n2.0 DETAILS\n" + " ".join(f"more{i}" for i in range(50))
        metadata = {'doc_name': 'test_doc', 'page': 1}
        chunks = chunker.chunk_text(text, metadata)
        # First chunk should reference INTRODUCTION section
        assert "INTRODUCTION" in chunks[0]['section'] or "Unknown" in chunks[0]['section']

    def test_chunk_id_format(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = " ".join(f"word{i}" for i in range(100))
        metadata = {'doc_name': 'my_doc', 'page': 3}
        chunks = chunker.chunk_text(text, metadata)
        for i, chunk in enumerate(chunks):
            assert chunk['chunk_id'] == f"my_doc_p3_c{i}"

    def test_single_word_text(self, tmp_path):
        chunker = _make_chunker(tmp_path)
        text = "hello"
        metadata = {'doc_name': 'test', 'page': 1}
        chunks = chunker.chunk_text(text, metadata)
        assert len(chunks) == 1
        assert chunks[0]['text'] == "hello"

    def test_long_sentence_does_not_hang(self, tmp_path):
        """Regression: a sentence longer than (chunk_size - overlap) must terminate.

        Before the forward-progress guard this looped forever, re-flushing the
        same overlap and growing the chunk list without bound.
        """
        chunker = _make_chunker(tmp_path)  # chunk_size=50, overlap=10
        text = "Short start. " + ("x" * 80) + ". Short end."
        metadata = {'doc_name': 'test_doc', 'page': 1}
        chunks = chunker.chunk_text(text, metadata)
        assert 0 < len(chunks) < 20
        assert any('x' * 80 in c['text'] for c in chunks)

    def test_many_long_sentences_terminate(self, tmp_path):
        """Regression: repeated oversized sentences still terminate with a bounded count."""
        chunker = _make_chunker(tmp_path)
        text = ". ".join("y" * 70 for _ in range(10)) + "."
        metadata = {'doc_name': 'd', 'page': 1}
        chunks = chunker.chunk_text(text, metadata)
        assert 0 < len(chunks) < 100
