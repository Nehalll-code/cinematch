#!/usr/bin/env python3
"""Train every model on one split and emit the comparison table.

    python experiments/run_eval.py --data data/ml-1m --split temporal --k 10 [--tune-hybrid]
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
from cinematch.tuning import DEFAULT_GRID, tune_content_weight


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", choices=["temporal", "loo"], default="temporal")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--tune-hybrid", action="store_true",
                    help="tune the content/iALS fusion weight on a validation split carved from train")
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
    def make_ials() -> IALSRecommender:
        return IALSRecommender(factors=64, regularization=0.05, alpha=20.0, iterations=15)

    ials = make_ials()

    untuned = HybridRecommender({"content": (content, 0.3), "ials": (ials, 0.7)}, cold_threshold=3)
    untuned.name = "hybrid_0.3"
    models = [PopularityRecommender(), content, knn, ials, untuned]

    tuning_result = None
    if args.tune_hybrid:
        # Tuned on a validation split carved from `train`; `test` is not passed in.
        tuning_result = tune_content_weight(
            ds, train, args.split, grid=DEFAULT_GRID, k=args.k, min_train=args.min_train,
            make_content=ContentRecommender, make_ials=make_ials,
        )
        w = tuning_result.best_weight
        print(f"tuned on validation ({tuning_result.n_validation_users} users): "
              f"content weight {w} (val ndcg {tuning_result.best_ndcg:.4f})")
        for c in tuning_result.curve:
            print(f"    w={c['content_weight']:<5} val_ndcg={c['ndcg']:.4f} val_recall={c['recall']:.4f}")
        tuned = HybridRecommender({"content": (content, w), "ials": (ials, 1.0 - w)}, cold_threshold=3)
        tuned.name = f"hybrid_tuned_{w:g}"
        models.append(tuned)

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
    payload = {"rows": rows}
    if tuning_result is not None:
        payload["hybrid_tuning"] = {
            "best_content_weight": tuning_result.best_weight,
            "n_validation_users": tuning_result.n_validation_users,
            "validation_curve": tuning_result.curve,
        }
    (out / f"results_{args.split}_k{args.k}.json").write_text(json.dumps(payload, indent=2))

    print("\n" + df[["model", "recall", "precision", "ndcg", "map", "coverage", "diversity",
                     "novelty", "fit_seconds"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
