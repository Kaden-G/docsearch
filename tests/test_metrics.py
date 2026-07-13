"""Tests for scripts/metrics.py — SQLite-based metrics tracking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from metrics import MetricsTracker


def _make_tracker(tmp_path):
    db_path = tmp_path / "test_metrics.db"
    return MetricsTracker(db_path=str(db_path))


def _fake_results(n=3, similarity=0.8):
    return [
        {
            'similarity': similarity,
            'rerank_score': similarity - 0.1,
            'chunk_id': f'chunk_{i}',
            'doc_name': 'test_doc',
            'page': 1,
            'text': f'result text {i}'
        }
        for i in range(n)
    ]


class TestLogQuery:
    def test_returns_query_id(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        qid = tracker.log_query("test query", _fake_results(), latency=0.05)
        assert isinstance(qid, int)
        assert qid > 0
        tracker.close()

    def test_sequential_ids(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        qid1 = tracker.log_query("query 1", _fake_results(), latency=0.05)
        qid2 = tracker.log_query("query 2", _fake_results(), latency=0.03)
        assert qid2 == qid1 + 1
        tracker.close()

    def test_empty_results(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        qid = tracker.log_query("no results", [], latency=0.01)
        assert qid > 0
        tracker.close()


class TestLogFeedback:
    def test_feedback_stores(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        qid = tracker.log_query("test", _fake_results(), latency=0.05)
        tracker.log_feedback(qid, rating=5, relevant_results=['chunk_0'])
        stats = tracker.get_feedback_stats()
        assert stats['total_feedback'] == 1
        assert stats['avg_rating'] == 5.0
        tracker.close()


class TestGetQueryStats:
    def test_stats_after_queries(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        tracker.log_query("query 1", _fake_results(), latency=0.05)
        tracker.log_query("query 2", _fake_results(), latency=0.10)
        stats = tracker.get_query_stats(days=7)
        assert stats['total_queries'] == 2
        assert stats['unique_queries'] == 2
        assert stats['avg_latency_ms'] > 0
        tracker.close()

    def test_empty_stats(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        stats = tracker.get_query_stats(days=7)
        assert stats['total_queries'] == 0
        tracker.close()


class TestGetTopQueries:
    def test_top_queries_ordering(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        for _ in range(5):
            tracker.log_query("popular query", _fake_results(), latency=0.05)
        for _ in range(2):
            tracker.log_query("less popular", _fake_results(), latency=0.05)

        top = tracker.get_top_queries(limit=10)
        assert top[0]['query'] == 'popular query'
        assert top[0]['count'] == 5
        assert top[1]['count'] == 2
        tracker.close()


class TestPersistentConnection:
    def test_wal_mode_enabled(self, tmp_path):
        tracker = _make_tracker(tmp_path)
        cursor = tracker._conn.execute('PRAGMA journal_mode')
        mode = cursor.fetchone()[0]
        assert mode == 'wal'
        tracker.close()

    def test_close_and_reopen(self, tmp_path):
        db_path = tmp_path / "reopen_test.db"
        tracker = MetricsTracker(db_path=str(db_path))
        tracker.log_query("persist test", _fake_results(), latency=0.05)
        tracker.close()

        # Reopen and verify data persisted
        tracker2 = MetricsTracker(db_path=str(db_path))
        stats = tracker2.get_query_stats(days=7)
        assert stats['total_queries'] == 1
        tracker2.close()
