#!/usr/bin/env python3
"""Why do the temporal and leave-one-out splits rank models differently?

Fits popularity and the TUNED item-kNN / iALS (hyperparameters read from the JSON that
`run_eval.py --tune-models` wrote; they were chosen on validation, not here) and breaks the test
metric down by user history length and by number of relevant items. It also measures how
concentrated each split's test interactions are on popular films. Nothing here is used to choose
anything; it exists to explain results, and each hypothesis it can refute is printed with the data.

    python experiments/analyze_splits.py --source db --tuned artifacts/tuned_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from cinematch.evaluate import evaluate
from cinematch.loading import SOURCES, load_dataset
from cinematch.recommenders import IALSRecommender, ItemKNNRecommender, PopularityRecommender
from cinematch.split import filter_test_to_known_users, global_temporal_split, leave_one_out_split


def per_user_ndcg(model, ds, train, test, k=10, batch=512) -> pd.Series:
    """NDCG@k per user, same definition as evaluate.evaluate (asserted against it below)."""
    truth = {int(u): g["item_idx"].to_numpy() for u, g in test.groupby("user_idx")}
    users = np.array(sorted(truth), dtype=np.int64)
    tm = ds.matrix(train).tocsr()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    cum = np.concatenate([[0.0], np.cumsum(disc)])
    out = []
    for s in range(0, len(users), batch):
        b = users[s : s + batch]
        sc = model.score_users(b).astype(np.float32, copy=True)
        for r, u in enumerate(b):
            sc[r, tm[u].indices] = -np.inf
        top = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(sc, top, axis=1), axis=1), axis=1)
        for r, u in enumerate(b):
            hits = np.isin(top[r], truth[int(u)]).astype(float)
            ideal = cum[min(len(truth[int(u)]), k)]
            out.append((hits * disc).sum() / max(ideal, 1e-12))
    return pd.Series(out, index=users)


def bucketed(frame: pd.DataFrame, by: str, edges: list[int], labels: list[str]) -> pd.DataFrame:
    b = pd.cut(frame[by], bins=edges, labels=labels, right=False)
    g = frame.groupby(b, observed=True)
    res = g[["popularity", "item_knn_tuned", "ials_tuned"]].mean()
    res.insert(0, "users", g.size())
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="csv")
    ap.add_argument("--data", default="data/ml-1m")
    ap.add_argument("--tuned", required=True, help="dir holding results_<split>_k10.json from --tune-models")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=None, help="optional path to write the numbers as JSON")
    args = ap.parse_args()

    ds = load_dataset(args.source, args.data)
    report = {}
    for split in ("temporal", "loo"):
        tuning = json.loads((Path(args.tuned) / f"results_{split}_k10.json").read_text())["tuning"]
        train, test = global_temporal_split(ds.interactions) if split == "temporal" else leave_one_out_split(ds.interactions)
        test = filter_test_to_known_users(train, test, 5)

        models = {
            "popularity": PopularityRecommender(),
            "item_knn_tuned": ItemKNNRecommender(**tuning["item_knn"]["best_params"]),
            "ials_tuned": IALSRecommender(**tuning["ials"]["best_params"]),
        }
        cols = {}
        for name, model in models.items():
            model.fit(ds, train)
            cols[name] = per_user_ndcg(model, ds, train, test, args.k)
            check = evaluate(model, ds, train, test, k=args.k).ndcg
            assert abs(cols[name].mean() - check) < 1e-6, f"per-user NDCG disagrees with evaluate() for {name}"
        df = pd.DataFrame(cols)
        df["train_len"] = train.groupby("user_idx").size().reindex(df.index).to_numpy()
        df["n_relevant"] = test.groupby("user_idx").size().reindex(df.index).to_numpy()

        pop = train.groupby("item_idx").size()
        top100 = set(pop.sort_values(ascending=False).index[:100])
        in_train = set(pop.index)
        share_top100 = float(test["item_idx"].isin(top100).mean())
        share_seen_item = float(test["item_idx"].isin(in_train).mean())

        print(f"\n=== {split}: {len(df)} users, mean NDCG " + "  ".join(f"{c}={df[c].mean():.4f}" for c in cols))
        print(f"test interactions on the 100 most popular TRAIN films: {share_top100:.1%}; "
              f"on films that appear in train at all: {share_seen_item:.1%}")
        print(f"by training-history length (positives in train):")
        h = bucketed(df, "train_len", [0, 20, 50, 100, 200, 10**9], ["<20", "20-49", "50-99", "100-199", "200+"])
        print(h.to_string(float_format=lambda v: f"{v:.4f}"))
        r = bucketed(df, "n_relevant", [0, 2, 5, 20, 50, 10**9], ["1", "2-4", "5-19", "20-49", "50+"])
        print("by number of relevant (held-out) items:")
        print(r.to_string(float_format=lambda v: f"{v:.4f}"))
        report[split] = {
            "users": len(df), "mean_ndcg": {c: float(df[c].mean()) for c in cols},
            "test_share_on_top100_train_popular": share_top100, "test_share_on_items_seen_in_train": share_seen_item,
            "median_train_len": float(df["train_len"].median()), "median_relevant": float(df["n_relevant"].median()),
            "by_train_len": h.reset_index().rename(columns={"train_len": "bucket"}).to_dict("records"),
            "by_n_relevant": r.reset_index().rename(columns={"n_relevant": "bucket"}).to_dict("records"),
        }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
