#!/usr/bin/env python3
"""
DocSearch Evaluation Suite
=======================

Measures the two things that matter for a RAG retriever:

    1. QUALITY  -- when I ask a question, do the *right* chunks come back,
                   and do they come back near the *top*?
    2. SPEED    -- how long does a search take, p50/p95/p99?

Think of the retriever as a librarian. Quality asks "did the librarian hand me
the right books?"; speed asks "how long did they take to fetch them?". This
suite scores both.

Note on dependencies: this harness is pleasantly self-contained -- it leans on a
local FAISS index via DocSearcher, not on any hosted model API. So there's no
"foundation-model landlord" collecting rent here; you can re-run it offline in
the air-gapped environment as many times as you like for free.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple   # NOTE: Tuple is imported but never used; safe to drop.
from datetime import datetime
import numpy as np
from search import DocSearcher  # Your retriever. Must expose .search(), .metadata, .query_cache


class DocSearchEvaluator:
    """
    Wraps a single DocSearcher and runs evaluation/benchmark routines against it.

    One evaluator == one index. Construct it pointing at the directory where your
    FAISS index + chunk metadata live.
    """

    def __init__(self, index_dir: str):
        # Where the index lives on disk (FAISS vectors + chunk metadata).
        self.index_dir = Path(index_dir)

        # The thing under test. We instantiate the real searcher so the numbers
        # we report reflect the real production path (embedding + FAISS + rerank).
        self.searcher = DocSearcher(str(index_dir))

        # Where to drop evaluation result JSON files. We resolve relative to THIS
        # file (../data/evaluation), not the current working directory, so results
        # land in the same place no matter where you launch the script from.
        self.results_dir = Path(__file__).parent.parent / 'data' / 'evaluation'

        # BEST-PRACTICE FLAG: exist_ok=True only stops it complaining if the leaf
        # folder already exists -- it will still raise FileNotFoundError if the
        # parent 'data/' directory doesn't exist yet. Safer:
        #     self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

    def evaluate_query(
        self,
        query: str,
        relevant_chunks: List[str],
        top_k: int = 5
    ) -> Dict:
        """
        Score ONE query against a hand-labeled ground-truth set.

        Args:
            query:           The search string a user would type.
            relevant_chunks: The chunk_ids YOU have judged as correct answers
                             for this query (your "answer key").
            top_k:           How many results to consider when scoring.

        Returns:
            A metrics dict for this single query (precision, recall, MRR, NDCG, latency).
        """
        # --- Run the search and time it --------------------------------------
        # We wrap ONLY the .search() call so latency reflects retrieval, not the
        # bookkeeping around it.
        start = time.time()
        results = self.searcher.search(query, top_k=top_k, rerank=True)
        latency = time.time() - start

        # Pull just the chunk IDs out of the result objects, in rank order.
        # results[0] is the top hit, results[1] the next, etc. Order matters
        # for the rank-aware metrics below (MRR, NDCG).
        retrieved = [r['chunk_id'] for r in results]

        # Start assembling this query's scorecard.
        metrics = {
            'query': query,
            'latency_ms': latency * 1000,          # seconds -> milliseconds
            'num_results': len(results),
            'relevant_chunks': relevant_chunks,    # the answer key (for auditing)
            'retrieved_chunks': retrieved,         # what we actually returned
        }

        # The set of items we got right: returned AND in the answer key.
        # (set & set == intersection)
        relevant_retrieved = set(retrieved[:top_k]) & set(relevant_chunks)

        # --- Precision@K -----------------------------------------------------
        # "Of the K things I handed back, what fraction were correct?"
        # High precision = low junk in the results.
        # NOTE: divides by top_k even if fewer than K results came back, which
        # mildly penalizes thin result sets. To score against what was actually
        # returned, divide by min(top_k, len(retrieved)) instead.
        metrics['precision@k'] = len(relevant_retrieved) / top_k if top_k > 0 else 0

        # --- Recall@K --------------------------------------------------------
        # "Of all the correct answers that exist, what fraction did I find in
        # the top K?" High recall = you're not missing the good stuff.
        # Precision and recall trade off: returning more usually raises recall
        # but lowers precision.
        metrics['recall@k'] = len(relevant_retrieved) / len(relevant_chunks) if relevant_chunks else 0

        # --- F1@K ------------------------------------------------------------
        # The harmonic mean of precision and recall -- a single number that
        # punishes lopsidedness. (Being great at one and terrible at the other
        # gives a low F1.) Guarded against divide-by-zero when both are 0.
        if metrics['precision@k'] + metrics['recall@k'] > 0:
            metrics['f1@k'] = 2 * (metrics['precision@k'] * metrics['recall@k']) / \
                              (metrics['precision@k'] + metrics['recall@k'])
        else:
            metrics['f1@k'] = 0

        # --- Mean Reciprocal Rank (MRR) --------------------------------------
        # "How high up was the FIRST correct answer?"
        #   - first result correct  -> 1/1 = 1.0
        #   - second result correct -> 1/2 = 0.5
        #   - third                 -> 1/3 = 0.33 ...
        # Great metric when there's essentially one right answer and you care
        # about it being at the top. The for/else below is a Python idiom: the
        # `else` runs ONLY if the loop finishes without hitting `break` (i.e.
        # no correct answer was found at all), so MRR defaults to 0.
        for i, chunk_id in enumerate(retrieved):
            if chunk_id in relevant_chunks:
                metrics['mrr'] = 1.0 / (i + 1)   # +1 because enumerate starts at 0
                break
        else:
            metrics['mrr'] = 0.0

        # --- NDCG@K ----------------------------------------------------------
        # Normalized Discounted Cumulative Gain. The most "complete" ranking
        # metric here: rewards correct answers AND rewards them more for being
        # near the top, normalized so 1.0 == the ideal ordering. See helper.
        metrics['ndcg@k'] = self._calculate_ndcg(retrieved, relevant_chunks, top_k)

        # --- Score diagnostics ----------------------------------------------
        # Average raw similarity (and rerank score, if present) across results.
        # These aren't quality metrics per se -- they're a sanity check on how
        # confident the retriever felt. Useful for spotting drift over time.
        if results:
            metrics['avg_similarity'] = np.mean([r['similarity'] for r in results])
            if 'rerank_score' in results[0]:
                metrics['avg_rerank_score'] = np.mean([r['rerank_score'] for r in results])

        return metrics

    def _calculate_ndcg(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain over the top-k results.

        Intuition: a correct answer is worth more the higher it ranks. The
        "discount" is 1/log2(rank+1), so position 1 is worth ~1.0, position 2
        ~0.63, position 3 ~0.5, and so on -- diminishing rewards as you go down.

        NDCG = DCG (what you got) / IDCG (the best you *could* have gotten).
        Dividing by the ideal makes scores comparable across queries with
        different numbers of correct answers. 1.0 == perfect ordering.

        NOTE: this uses BINARY relevance -- a chunk is either correct (gain 1)
        or not (gain 0). If you ever add graded relevance (e.g. "perfect" vs
        "okay" matches), you'd use (2**gain - 1) in the numerator instead.
        """
        # DCG: sum the discounted gains of the correct items we actually returned.
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved[:k]):
            if chunk_id in relevant:
                dcg += 1.0 / np.log2(i + 2)  # +2 so the top rank uses log2(2)=1, avoiding log2(1)=0

        # IDCG: the DCG of a perfect ranking, i.e. every correct answer packed
        # into the top slots. Capped at k (and at the number of relevant items).
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))

        # Guard the divide: if there were no relevant items, NDCG is 0 by convention.
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_test_set(self, test_queries: List[Dict]) -> Dict:
        """
        Run evaluate_query over a whole labeled test set and average the results.

        Args:
            test_queries: list of {"query": str, "relevant": [chunk_ids]} dicts.

        Returns:
            Aggregated (averaged) metrics across all queries, also written to disk.
        """
        all_metrics = []

        # Score each query one at a time, with a lightweight progress readout.
        print(f"\nEvaluating {len(test_queries)} test queries...")
        for i, test in enumerate(test_queries, 1):
            print(f"  [{i}/{len(test_queries)}] {test['query'][:50]}...")  # truncate long queries in the log
            metrics = self.evaluate_query(
                test['query'],
                test['relevant'],
                top_k=5
            )
            all_metrics.append(metrics)

        # --- Aggregate -------------------------------------------------------
        # Mean across every query gives the headline numbers. We also report p95
        # latency -- the slow-tail experience, which averages tend to hide.
        # NOTE: indexing all_metrics[0] below assumes test_queries was non-empty.
        # An empty list will raise IndexError; guard upstream if that's possible.
        aggregated = {
            'num_queries': len(test_queries),
            'avg_precision@5': np.mean([m['precision@k'] for m in all_metrics]),
            'avg_recall@5': np.mean([m['recall@k'] for m in all_metrics]),
            'avg_f1@5': np.mean([m['f1@k'] for m in all_metrics]),
            'avg_mrr': np.mean([m['mrr'] for m in all_metrics]),
            'avg_ndcg@5': np.mean([m['ndcg@k'] for m in all_metrics]),
            'avg_latency_ms': np.mean([m['latency_ms'] for m in all_metrics]),
            'p95_latency_ms': np.percentile([m['latency_ms'] for m in all_metrics], 95),
        }

        # These two keys only exist if the searcher produced them, so we check
        # the first query's metrics before trying to average them.
        if 'avg_similarity' in all_metrics[0]:
            aggregated['avg_similarity'] = np.mean([m['avg_similarity'] for m in all_metrics])
        if 'avg_rerank_score' in all_metrics[0]:
            aggregated['avg_rerank_score'] = np.mean([m['avg_rerank_score'] for m in all_metrics])

        # --- Persist results -------------------------------------------------
        # Timestamped filename so successive runs don't clobber each other --
        # this is what lets you track quality regressions over time.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'eval_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'aggregated': aggregated,
                'individual_queries': all_metrics,  # keep per-query detail for drill-down
                'timestamp': timestamp
            }, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        return aggregated

    def benchmark_performance(self, num_queries: int = 100) -> Dict:
        """
        SPEED-only benchmark (no quality scoring). Hammers the searcher with many
        queries and reports the latency distribution + a rough cache-hit rate.

        IMPORTANT: this measures how *fast* the system is, not how *good* its
        answers are. It builds queries by feeding chunk text back in as the
        query, so each query trivially matches its own source chunk -- that's
        fine for timing, but it tells you nothing about ranking quality. Keep
        evaluate_test_set() as the source of truth for quality.
        """
        # Sample real indexed chunks to use as query material. replace=False so
        # we don't time the same chunk twice; size is capped at how many we have.
        # NOTE: np.random.choice over a list of dicts coerces it to an object
        # array, which works but is a touch fragile. random.sample(self.searcher.metadata, n)
        # is a cleaner stdlib alternative for sampling Python objects.
        sample_chunks = np.random.choice(
            self.searcher.metadata,
            size=min(num_queries, len(self.searcher.metadata)),
            replace=False
        )

        latencies = []
        cache_hits = 0

        print(f"\nBenchmarking with {len(sample_chunks)} queries...")
        for i, chunk in enumerate(sample_chunks, 1):
            # Use the first 100 chars of the chunk as a stand-in query.
            query = chunk['text'][:100]

            # First call: this is the latency we actually record (cold-ish path).
            start = time.time()
            results = self.searcher.search(query, top_k=5)
            latency = time.time() - start
            latencies.append(latency * 1000)

            # Second call with the SAME query: should hit the cache and be much
            # faster. We infer a cache hit if the repeat is >10x faster.
            # BEST-PRACTICE FLAG: timing-based cache detection is noisy -- a busy
            # CPU or GC pause can flip the result. If DocSearcher can expose a
            # real hit/miss counter (e.g. searcher.query_cache stats), trust that
            # instead of this heuristic.
            start = time.time()
            _ = self.searcher.search(query, top_k=5)
            cached_latency = time.time() - start
            if cached_latency < latency / 10:  # 10x faster == assume cache hit
                cache_hits += 1

            if i % 20 == 0:  # progress ping every 20 queries
                print(f"  Progress: {i}/{len(sample_chunks)}")

        # Report the full latency distribution. p95/p99 are the ones to watch:
        # they describe the worst experiences your users actually feel, which a
        # single average will quietly bury.
        return {
            'num_queries': len(latencies),
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'cache_hit_rate': cache_hits / len(latencies),
        }

    def compare_configurations(self) -> Dict:
        """
        A/B the retriever WITH vs WITHOUT reranking on a single fixed query.

        Reranking usually improves the top result's relevance but costs latency;
        this is the head-to-head that shows you the tradeoff in numbers.
        """
        test_query = "What are the safety procedures?"

        # The two configs we're comparing -- passed through as **kwargs to search().
        configs = {
            'baseline': {'rerank': False},
            'with_reranking': {'rerank': True},
        }

        results = {}

        print("\nComparing configurations...")
        for name, config in configs.items():
            print(f"  Testing: {name}")

            # Run each config 10 times and average, to smooth out timing jitter.
            # We clear the query cache EACH iteration so every run measures a true
            # cold search -- otherwise the cache would make runs 2-10 fake-fast.
            # NOTE: this reaches into searcher.query_cache directly, so it's
            # coupled to DocSearcher's internals. If that attribute is ever
            # renamed, this breaks; a public searcher.clear_cache() would be safer.
            latencies = []
            for _ in range(10):
                self.searcher.query_cache.clear()
                start = time.time()
                search_results = self.searcher.search(test_query, top_k=5, **config)
                latency = time.time() - start
                latencies.append(latency * 1000)

            # Record average latency plus the top hit's scores for this config.
            # (search_results here is from the LAST of the 10 runs -- fine, since
            # the result content is identical across runs for a fixed query.)
            results[name] = {
                'avg_latency_ms': np.mean(latencies),
                'top_similarity': search_results[0]['similarity'] if search_results else 0,
                'top_rerank_score': search_results[0].get('rerank_score', 0) if search_results else 0,
            }

        return results


def create_sample_test_set() -> List[Dict]:
    """
    A skeleton test set to copy from. The 'relevant' lists are intentionally
    empty -- you MUST fill them with real chunk_ids from your index before the
    quality metrics mean anything. Empty answer keys => recall/precision of 0.
    """
    return [
        {
            'query': 'What safety equipment is required?',
            'relevant': []  # TODO: fill in the chunk_ids that correctly answer this
        },
        {
            'query': 'How to initialize the system?',
            'relevant': []  # TODO
        },
        {
            'query': 'Emergency shutdown procedure',
            'relevant': []  # TODO
        },
    ]


def main():
    """
    CLI entry point. Runs three things in order:
      [1] a pure speed benchmark,
      [2] a baseline-vs-reranking comparison,
      [3] a quality eval IF a labeled test_queries.json exists.
    """
    # Resolve paths relative to this file so the script runs from anywhere.
    base_dir = Path(__file__).parent.parent
    index_dir = base_dir / 'data' / 'index'

    evaluator = DocSearchEvaluator(str(index_dir))

    print("="*80)
    print("DocSearch Evaluation Suite")
    print("="*80)

    # --- [1] Performance benchmark (speed only) ------------------------------
    print("\n[1] Performance Benchmark")
    print("-" * 80)
    perf_results = evaluator.benchmark_performance(num_queries=50)
    print("\nPerformance Results:")
    # Format each metric for readability: ms for latency, % for rates, raw otherwise.
    for metric, value in perf_results.items():
        if 'latency' in metric:
            print(f"  {metric}: {value:.2f} ms")
        elif 'rate' in metric:
            print(f"  {metric}: {value:.1%}")
        else:
            print(f"  {metric}: {value}")

    # --- [2] Config comparison (baseline vs rerank) --------------------------
    print("\n[2] Configuration Comparison")
    print("-" * 80)
    config_results = evaluator.compare_configurations()
    print("\nConfiguration Results:")
    for config, metrics in config_results.items():
        print(f"\n  {config}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    # --- [3] Quality eval (only if you've labeled a test set) ----------------
    print("\n[3] Test Set Evaluation")
    print("-" * 80)
    print("\nTo run test set evaluation:")
    print("  1. Create labeled test queries in data/evaluation/test_queries.json")
    print("  2. Format: [{'query': '...', 'relevant': ['chunk_id1', ...]}]")
    print("  3. Run: evaluator.evaluate_test_set(test_queries)")

    # Only runs the real quality eval if the labeled file exists; otherwise the
    # benchmarks above are all you get this run.
    test_file = base_dir / 'data' / 'evaluation' / 'test_queries.json'
    if test_file.exists():
        with open(test_file) as f:
            test_queries = json.load(f)
        eval_results = evaluator.evaluate_test_set(test_queries)
        print("\nEvaluation Results:")
        for metric, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    else:
        print(f"\nNo test file found at: {test_file}")
        print("Using default evaluation metrics from benchmarks above.")

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


# Standard guard: only run main() when executed directly (python evaluate.py),
# not when this module is imported elsewhere.
if __name__ == '__main__':
    main()
