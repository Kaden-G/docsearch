#!/usr/bin/env python3
"""
Agentic vs. Traditional Comparison Harness
===========================================

Runs a labeled test set through BOTH pipelines on the SAME index with the SAME
LLM, then reports a side-by-side scorecard across the four axes the project
cares about:

    relevant   -> fact_coverage, citation_validity, must_cite_recall, retrieval P/R
    timely     -> latency p50/p95
    resilient  -> grounding (hallucination signal), correct abstention
    scalable   -> LLM calls/query, tokens/query

Because both pipelines share DocSearcher, the only independent variable is
orchestration — so any delta is attributable to "agentic vs. traditional",
not to a different retrieval stack or model.

Usage
-----
    python scripts/compare.py --testset data/evaluation/testset.json
    python scripts/compare.py --testset ... --pipelines agentic traditional
    python scripts/compare.py --testset ... --no-verify    # ablate the verifier

Test-set format (one object per query):
    {
      "query": "How do I switch databases on TTC?",
      "relevant": ["chunk_id_a", "chunk_id_b"],     # optional, for retrieval metrics
      "expected_facts": ["stop the service", "edit the config"],  # optional
      "must_cite": ["HOW_TO_SWITCH_DATABASES_ON_TTC"],            # optional
      "out_of_scope": false                                       # optional
    }
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Any, Optional

# Make sibling modules importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).parent))

from search import DocSearcher
from pipeline import TraditionalPipeline, AgenticPipeline
from answer_metrics import score_answer
import evaluate as eval_mod


def _nanmean(values: List[float]) -> float:
    """Mean that ignores NaN (unlabeled) entries; returns NaN if all are NaN."""
    clean = [v for v in values if v == v]  # NaN != NaN
    return mean(clean) if clean else float("nan")


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class ComparisonHarness:
    def __init__(self, index_dir: str, verbose: bool = False):
        print("Loading shared backend (DocSearcher)...")
        self.searcher = DocSearcher(index_dir)
        self.verbose = verbose
        # Reuse the retrieval-metric helpers from evaluate.py without duplicating them.
        self._retrieval = eval_mod.DocSearchEvaluator.__new__(eval_mod.DocSearchEvaluator)
        self._retrieval.searcher = self.searcher

    def _make_pipeline(self, kind: str, verify: bool, top_k: int, max_passes: int,
                       self_assess: bool):
        if kind == "agentic":
            return AgenticPipeline(
                self.searcher, top_k=top_k, max_passes=max_passes,
                self_assess=self_assess, verify=verify, verbose=self.verbose,
            )
        return TraditionalPipeline(self.searcher, top_k=top_k, verbose=self.verbose)

    def run(self, test_queries: List[Dict[str, Any]], pipelines: List[str],
            verify: bool, top_k: int, max_passes: int, self_assess: bool) -> Dict[str, Any]:
        report: Dict[str, Any] = {"pipelines": {}, "per_query": []}

        built = {k: self._make_pipeline(k, verify, top_k, max_passes, self_assess)
                 for k in pipelines}

        per_pipeline_rows: Dict[str, List[Dict]] = {k: [] for k in pipelines}

        for i, test in enumerate(test_queries, 1):
            query = test["query"]
            print(f"\n[{i}/{len(test_queries)}] {query[:70]}")
            row: Dict[str, Any] = {"query": query}

            for kind, pipe in built.items():
                result = pipe.run(query)

                ans_metrics = score_answer(result, test)

                # Retrieval metrics if ground-truth chunks are labeled.
                relevant = test.get("relevant", [])
                if relevant:
                    retrieved = result.retrieved_chunk_ids
                    rel_set = set(relevant)
                    found = set(retrieved[:top_k]) & rel_set
                    ans_metrics["precision@k"] = len(found) / top_k if top_k else 0.0
                    ans_metrics["recall@k"] = len(found) / len(rel_set) if rel_set else 0.0
                    ans_metrics["ndcg@k"] = self._retrieval._calculate_ndcg(retrieved, relevant, top_k)

                per_pipeline_rows[kind].append(ans_metrics)
                row[kind] = ans_metrics
                print(f"    {kind:12s} "
                      f"facts={_fmt(ans_metrics.get('fact_coverage'))} "
                      f"ground={_fmt(ans_metrics.get('grounding'))} "
                      f"cites_ok={_fmt(ans_metrics.get('citation_validity'))} "
                      f"calls={ans_metrics['llm_calls']} "
                      f"{ans_metrics['latency_ms']:.0f}ms "
                      f"conf={ans_metrics['confidence']}")

            report["per_query"].append(row)

        # Aggregate per pipeline.
        for kind, rows in per_pipeline_rows.items():
            report["pipelines"][kind] = self._aggregate(rows)

        return report

    @staticmethod
    def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        latencies = [r["latency_ms"] for r in rows]
        agg = {
            "num_queries": len(rows),
            "fact_coverage": _nanmean([r.get("fact_coverage", float("nan")) for r in rows]),
            "must_cite_recall": _nanmean([r.get("must_cite_recall", float("nan")) for r in rows]),
            "citation_validity": _nanmean([r.get("citation_validity", float("nan")) for r in rows]),
            "grounding": _nanmean([r.get("grounding", float("nan")) for r in rows]),
            "abstain_rate": _nanmean([1.0 if r.get("abstained") else 0.0 for r in rows]),
            "avg_llm_calls": _nanmean([r["llm_calls"] for r in rows]),
            "avg_tokens": _nanmean([r["total_tokens"] for r in rows]),
            "avg_latency_ms": _nanmean(latencies),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
        }
        # Optional metrics if present.
        for key in ("precision@k", "recall@k", "ndcg@k", "resilience"):
            vals = [r[key] for r in rows if key in r]
            if vals:
                agg[key] = _nanmean(vals)
        return agg


def _fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "  -  "
    return f"{v:.2f}"


def print_scorecard(report: Dict[str, Any]) -> None:
    pipelines = list(report["pipelines"].keys())
    if not pipelines:
        print("No results.")
        return

    metrics_order = [
        ("fact_coverage", "Fact coverage", "higher"),
        ("citation_validity", "Citation validity", "higher"),
        ("must_cite_recall", "Must-cite recall", "higher"),
        ("grounding", "Grounding (anti-hallu.)", "higher"),
        ("precision@k", "Precision@k", "higher"),
        ("recall@k", "Recall@k", "higher"),
        ("ndcg@k", "NDCG@k", "higher"),
        ("resilience", "Resilience (abstain OK)", "higher"),
        ("abstain_rate", "Abstain rate", "info"),
        ("avg_llm_calls", "LLM calls / query", "lower"),
        ("avg_tokens", "Tokens / query", "lower"),
        ("avg_latency_ms", "Latency avg (ms)", "lower"),
        ("p95_latency_ms", "Latency p95 (ms)", "lower"),
    ]

    width = 26
    header = "Metric".ljust(width) + "".join(p.center(14) for p in pipelines) + "  Better"
    print("\n" + "=" * len(header))
    print("AGENTIC vs TRADITIONAL — SCORECARD")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for key, label, better in metrics_order:
        if not any(key in report["pipelines"][p] for p in pipelines):
            continue
        line = label.ljust(width)
        for p in pipelines:
            val = report["pipelines"][p].get(key)
            line += _fmt(val).center(14) if val is not None else "  -  ".center(14)
        line += f"  {better}"
        print(line)
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser(description="Compare agentic vs traditional RAG.")
    ap.add_argument("--index", default=str(Path(__file__).parent.parent / "data" / "index"))
    ap.add_argument("--testset", required=True, help="Path to labeled test-set JSON.")
    ap.add_argument("--pipelines", nargs="+", default=["traditional", "agentic"],
                    choices=["traditional", "agentic"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--no-self-assess", action="store_true",
                    help="Disable the retriever's self-assessment loop (ablation).")
    ap.add_argument("--no-verify", action="store_true",
                    help="Disable the verifier (ablation).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.testset, "r", encoding="utf-8") as f:
        test_queries = json.load(f)
    if isinstance(test_queries, dict) and "queries" in test_queries:
        test_queries = test_queries["queries"]

    harness = ComparisonHarness(args.index, verbose=args.verbose)
    report = harness.run(
        test_queries,
        pipelines=args.pipelines,
        verify=not args.no_verify,
        top_k=args.top_k,
        max_passes=args.max_passes,
        self_assess=not args.no_self_assess,
    )

    print_scorecard(report)

    out_dir = Path(__file__).parent.parent / "data" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"comparison_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {out_file}")


if __name__ == "__main__":
    main()
