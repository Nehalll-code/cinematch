#!/usr/bin/env python3
"""Train every model on one split and emit the comparison table.

    python experiments/run_eval.py --data data --split temporal --k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cinematch.data import load_movielens
from cinematch.evaluate import evaluate, results_table
from cinematch.recommenders import (
    ContentRecommender,
    HybridRecommender,
    IALSRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
)
from cinematch.split import filter_test_to_known_users, global_temporal_split, leave_one_out_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", choices=["temporal", "loo"], default="temporal")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    ds = load_movielens(args.data)
    print(f"{len(ds.interactions)} positive interactions | {ds.n_users} users | {ds.n_items} items")

    if args.split == "temporal":
        train, test = global_temporal_split(ds.interactions, args.test_frac)
    else:
        train, test = leave_one_out_split(ds.interactions)
    test = filter_test_to_known_users(train, test, args.min_train)
    print(f"split={args.split}: train={len(train)} test={len(test)} "
          f"evaluable_users={test['user_idx'].nunique()}")

    content = ContentRecommender()
    knn = ItemKNNRecommender(k=200, shrink=10.0)
    ials = IALSRecommender(factors=64, regularization=0.05, alpha=20.0, iterations=15)

    models = [
        PopularityRecommender(),
        content,
        knn,
        ials,
        HybridRecommender({"content": (content, 0.3), "ials": (ials, 0.7)}, cold_threshold=3),
    ]

    rows = []
    for model in models:
        t0 = time.perf_counter()
        model.fit(ds, train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        m = evaluate(model, ds, train, test, k=args.k)
        eval_s = time.perf_counter() - t0
        row = m.as_row()
        row["fit_seconds"] = round(fit_s, 2)
        row["eval_seconds"] = round(eval_s, 2)
        rows.append(row)
        print(f"  {m.model:<11} recall@{args.k}={m.recall:.4f}  ndcg@{args.k}={m.ndcg:.4f}  "
              f"cov={m.coverage:.3f}  fit={fit_s:.1f}s")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"results_{args.split}_k{args.k}.csv", index=False)
    (out / f"results_{args.split}_k{args.k}.json").write_text(json.dumps(rows, indent=2))

    print("\n" + df[["model", "recall", "precision", "ndcg", "map", "coverage", "diversity",
                     "novelty", "fit_seconds"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
