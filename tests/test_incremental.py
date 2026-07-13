"""Tests for incremental indexing — detect_changes, manifest, extract_files."""

import json
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from extract import DocumentExtractor


def _get_indexer_class():
    """Import EmbeddingIndexer with heavy deps mocked to avoid segfaults in test."""
    with mock.patch.dict('sys.modules', {
        'sentence_transformers': mock.MagicMock(),
        'faiss': mock.MagicMock(),
    }):
        # Force reimport with mocked deps
        if 'embed' in sys.modules:
            del sys.modules['embed']
        from embed import EmbeddingIndexer
        return EmbeddingIndexer


def _make_indexer(tmp_path):
    """Create an EmbeddingIndexer with mocked model loading."""
    cls = _get_indexer_class()
    index_dir = tmp_path / "index"
    index_dir.mkdir(exist_ok=True)
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir(exist_ok=True)
    # Bypass __init__ model loading by constructing manually
    obj = object.__new__(cls)
    obj.processed_dir = proc_dir
    obj.index_dir = index_dir
    obj.model_name = 'mock-model'
    return obj


class TestGetRawFiles:
    def test_finds_supported_formats(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.pdf").write_bytes(b"fake")
        (raw / "b.docx").write_bytes(b"fake")
        (raw / "c.txt").write_text("hello")
        (raw / "d.csv").write_text("skip me")
        (raw / "~$temp.docx").write_bytes(b"temp")

        extractor = DocumentExtractor(str(raw), str(tmp_path / "proc"))
        files = extractor.get_raw_files()
        names = [f.name for f in files]

        assert "a.pdf" in names
        assert "b.docx" in names
        assert "c.txt" in names
        assert "d.csv" not in names
        assert "~$temp.docx" not in names

    def test_empty_dir(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        extractor = DocumentExtractor(str(raw), str(tmp_path / "proc"))
        assert extractor.get_raw_files() == []


class TestExtractFiles:
    def test_extracts_only_specified(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.txt").write_text("Document A content", encoding='utf-8')
        (raw / "b.txt").write_text("Document B content", encoding='utf-8')

        proc = tmp_path / "proc"
        proc.mkdir()

        extractor = DocumentExtractor(str(raw), str(proc))
        results = extractor.extract_files([raw / "a.txt"])

        assert "a" in results
        assert "b" not in results
        assert (proc / "a.json").exists()
        assert not (proc / "b.json").exists()


class TestSourceManifest:
    def test_save_and_load(self, tmp_path):
        indexer = _make_indexer(tmp_path)
        manifest = {"test.txt": {"mtime": 12345.0, "size": 100}}
        indexer.save_source_manifest(manifest)

        loaded = indexer.get_source_manifest()
        assert loaded == manifest

    def test_empty_manifest(self, tmp_path):
        indexer = _make_indexer(tmp_path)
        assert indexer.get_source_manifest() == {}


class TestDetectChanges:
    def _setup(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        indexer = _make_indexer(tmp_path)
        return raw, indexer

    def test_all_new_without_manifest(self, tmp_path):
        raw, indexer = self._setup(tmp_path)
        (raw / "a.txt").write_text("content")
        (raw / "b.txt").write_text("content")

        changes = indexer.detect_changes(str(raw))
        assert len(changes['new']) == 2
        assert len(changes['modified']) == 0
        assert len(changes['deleted']) == 0

    def test_detects_new_file(self, tmp_path):
        raw, indexer = self._setup(tmp_path)

        # Create initial file and save manifest
        f1 = raw / "existing.txt"
        f1.write_text("old content")
        stat = f1.stat()
        indexer.save_source_manifest({
            "existing.txt": {"mtime": stat.st_mtime, "size": stat.st_size}
        })

        # Add a new file
        (raw / "brand_new.txt").write_text("new content")

        changes = indexer.detect_changes(str(raw))
        new_names = [f.name for f in changes['new']]
        assert "brand_new.txt" in new_names
        assert "existing.txt" not in new_names
        assert len(changes['modified']) == 0

    def test_detects_modified_file(self, tmp_path):
        raw, indexer = self._setup(tmp_path)

        f1 = raw / "doc.txt"
        f1.write_text("original")
        stat = f1.stat()
        indexer.save_source_manifest({
            "doc.txt": {"mtime": stat.st_mtime, "size": stat.st_size}
        })

        # Modify the file (ensure mtime changes)
        time.sleep(0.1)
        f1.write_text("modified content that is longer")

        changes = indexer.detect_changes(str(raw))
        assert len(changes['new']) == 0
        mod_names = [f.name for f in changes['modified']]
        assert "doc.txt" in mod_names

    def test_detects_deleted_file(self, tmp_path):
        raw, indexer = self._setup(tmp_path)

        indexer.save_source_manifest({
            "gone.txt": {"mtime": 12345.0, "size": 50}
        })

        changes = indexer.detect_changes(str(raw))
        assert "gone.txt" in changes['deleted']
        assert len(changes['new']) == 0

    def test_no_changes(self, tmp_path):
        raw, indexer = self._setup(tmp_path)

        f1 = raw / "stable.txt"
        f1.write_text("unchanged")
        stat = f1.stat()
        indexer.save_source_manifest({
            "stable.txt": {"mtime": stat.st_mtime, "size": stat.st_size}
        })

        changes = indexer.detect_changes(str(raw))
        assert len(changes['new']) == 0
        assert len(changes['modified']) == 0
        assert len(changes['deleted']) == 0
