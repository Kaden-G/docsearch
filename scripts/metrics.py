#!/usr/bin/env python3
"""
DocSearch Metrics Tracking
Logs queries, results, and performance metrics over time.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3


class MetricsTracker:
    """Track and log search metrics."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).parent.parent
            metrics_dir = base_dir / 'data' / 'metrics'
            metrics_dir.mkdir(exist_ok=True)
            db_path = metrics_dir / 'docsearch_metrics.db'

        self.db_path = str(db_path)
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False  # Safe: SQLite serializes writes internally
        )
        self._conn.execute('PRAGMA journal_mode=WAL')  # Concurrent reads
        self._init_db()

    def close(self):
        """Close the persistent database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_db(self):
        """Initialize SQLite database tables."""
        cursor = self._conn.cursor()

        # Query metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                top_k INTEGER,
                num_results INTEGER,
                latency_ms REAL,
                cache_hit BOOLEAN,
                use_rerank BOOLEAN,
                generate_answer BOOLEAN,
                avg_similarity REAL,
                avg_rerank_score REAL,
                top_similarity REAL,
                top_rerank_score REAL
            )
        ''')

        # User feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                timestamp TEXT NOT NULL,
                rating INTEGER,
                relevant_results TEXT,
                comments TEXT,
                FOREIGN KEY (query_id) REFERENCES queries(id)
            )
        ''')

        # System performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL
            )
        ''')

        self._conn.commit()

    def log_query(
        self,
        query: str,
        results: List[Dict],
        latency: float,
        top_k: int = 5,
        cache_hit: bool = False,
        use_rerank: bool = True,
        generate_answer: bool = False
    ) -> int:
        """
        Log a search query and its results.

        Args:
            query: Search query
            results: Search results
            latency: Query latency in seconds
            top_k: Number of results requested
            cache_hit: Whether result was from cache
            use_rerank: Whether re-ranking was used
            generate_answer: Whether answer was generated

        Returns:
            Query ID
        """
        cursor = self._conn.cursor()

        # Calculate metrics
        avg_similarity = sum(r['similarity'] for r in results) / len(results) if results else 0
        avg_rerank_score = None
        if results and 'rerank_score' in results[0]:
            avg_rerank_score = sum(r['rerank_score'] for r in results) / len(results)

        top_similarity = results[0]['similarity'] if results else 0
        top_rerank_score = results[0].get('rerank_score') if results else None

        cursor.execute('''
            INSERT INTO queries (
                timestamp, query, top_k, num_results, latency_ms, cache_hit,
                use_rerank, generate_answer, avg_similarity, avg_rerank_score,
                top_similarity, top_rerank_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            query,
            top_k,
            len(results),
            latency * 1000,
            cache_hit,
            use_rerank,
            generate_answer,
            avg_similarity,
            avg_rerank_score,
            top_similarity,
            top_rerank_score
        ))

        query_id = cursor.lastrowid
        self._conn.commit()

        return query_id

    def log_feedback(
        self,
        query_id: int,
        rating: int,
        relevant_results: List[str],
        comments: str = ""
    ):
        """
        Log user feedback for a query.

        Args:
            query_id: Query ID from log_query
            rating: 1-5 star rating
            relevant_results: List of chunk_ids that were relevant
            comments: Optional text feedback
        """
        cursor = self._conn.cursor()

        cursor.execute('''
            INSERT INTO feedback (query_id, timestamp, rating, relevant_results, comments)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            query_id,
            datetime.now().isoformat(),
            rating,
            json.dumps(relevant_results),
            comments
        ))

        self._conn.commit()

    def get_query_stats(self, days: int = 7) -> Dict:
        """Get query statistics for the last N days."""
        cursor = self._conn.cursor()

        # Calculate date threshold
        threshold = datetime.now().isoformat()[:10]  # YYYY-MM-DD

        cursor.execute('''
            SELECT
                COUNT(*) as total_queries,
                AVG(latency_ms) as avg_latency,
                AVG(num_results) as avg_num_results,
                AVG(avg_similarity) as avg_similarity,
                AVG(avg_rerank_score) as avg_rerank_score,
                SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits,
                COUNT(DISTINCT query) as unique_queries
            FROM queries
            WHERE DATE(timestamp) >= DATE(?, '-' || ? || ' days')
        ''', (threshold, days))

        row = cursor.fetchone()

        if row:
            total = row[0] or 0
            return {
                'total_queries': total,
                'avg_latency_ms': row[1] or 0,
                'avg_num_results': row[2] or 0,
                'avg_similarity': row[3] or 0,
                'avg_rerank_score': row[4] or 0,
                'cache_hits': row[5] or 0,
                'cache_hit_rate': (row[5] / total) if total > 0 else 0,
                'unique_queries': row[6] or 0,
                'days': days
            }
        return {}

    def get_top_queries(self, limit: int = 10) -> List[Dict]:
        """Get most frequent queries."""
        cursor = self._conn.cursor()

        cursor.execute('''
            SELECT
                query,
                COUNT(*) as count,
                AVG(latency_ms) as avg_latency,
                AVG(avg_similarity) as avg_similarity
            FROM queries
            GROUP BY query
            ORDER BY count DESC
            LIMIT ?
        ''', (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'query': row[0],
                'count': row[1],
                'avg_latency_ms': row[2],
                'avg_similarity': row[3]
            })

        return results

    def get_feedback_stats(self) -> Dict:
        """Get user feedback statistics."""
        cursor = self._conn.cursor()

        cursor.execute('''
            SELECT
                COUNT(*) as total_feedback,
                AVG(rating) as avg_rating,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive_feedback,
                SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative_feedback
            FROM feedback
        ''')

        row = cursor.fetchone()

        if row:
            total = row[0] or 0
            return {
                'total_feedback': total,
                'avg_rating': row[1] or 0,
                'positive_feedback': row[2] or 0,
                'negative_feedback': row[3] or 0,
                'satisfaction_rate': (row[2] / total) if total > 0 else 0
            }
        return {}

    def export_metrics(self, output_file: str):
        """Export all metrics to JSON."""
        metrics = {
            'query_stats_7d': self.get_query_stats(7),
            'query_stats_30d': self.get_query_stats(30),
            'top_queries': self.get_top_queries(20),
            'feedback_stats': self.get_feedback_stats(),
            'export_timestamp': datetime.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics exported to: {output_file}")


def main():
    """Demo metrics tracking."""
    tracker = MetricsTracker()

    print("="*80)
    print("DocSearch Metrics Tracker")
    print("="*80)

    # Get stats
    print("\n[1] Query Statistics (Last 7 Days)")
    print("-" * 80)
    stats = tracker.get_query_stats(7)
    for metric, value in stats.items():
        if 'rate' in metric or 'satisfaction' in metric:
            print(f"  {metric}: {value:.1%}")
        elif 'latency' in metric or 'similarity' in metric or 'score' in metric:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    print("\n[2] Top Queries")
    print("-" * 80)
    top_queries = tracker.get_top_queries(10)
    for i, q in enumerate(top_queries, 1):
        print(f"  [{i}] \"{q['query'][:50]}...\" (count: {q['count']})")

    print("\n[3] Feedback Statistics")
    print("-" * 80)
    feedback = tracker.get_feedback_stats()
    if feedback.get('total_feedback', 0) > 0:
        for metric, value in feedback.items():
            if 'rate' in metric:
                print(f"  {metric}: {value:.1%}")
            elif 'rating' in metric:
                print(f"  {metric}: {value:.2f} / 5.0")
            else:
                print(f"  {metric}: {value}")
    else:
        print("  No feedback collected yet")

    # Export
    print("\n[4] Export Metrics")
    print("-" * 80)
    output_file = Path(__file__).parent.parent / 'data' / 'metrics' / 'metrics_report.json'
    tracker.export_metrics(str(output_file))

    print("\n" + "="*80)
    print("Metrics tracking complete!")
    print("="*80)


if __name__ == '__main__':
    main()
