#!/usr/bin/env python3
"""Are the gaps between the tuned models bigger than test-user sampling noise?

Fits popularity and the tuned item-kNN / iALS once (hyperparameters read from the JSON that
`run_eval.py --tune-models` wrote; chosen on validation), computes per-user NDCG@k on the test
split, then bootstraps the TEST USERS: each resample draws users with replacement and the same
draw is applied to every model, so the CI on a difference is paired. Nothing here chooses anything.

    python experiments/bootstrap_ci.py --tuned artifacts/tuned_v4 --split temporal --out artifacts/tuned_v4/bootstrap_temporal.json
    (also writes per_user_ndcg_<split>.npz next to --out)

It quantifies test-user sampling noise only. It does not include variance from the model fit
(iALS init seed) or from the choice of validation split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from analyze_splits import per_user_ndcg
from cinematch.evaluate import evaluate
from cinematch.loading import SOURCES, load_dataset
from cinematch.recommenders import IALSRecommender, ItemKNNRecommender, PopularityRecommender
from cinematch.split import filter_test_to_known_users, global_temporal_split, leave_one_out_split

PAIRS = [("ials_tuned", "item_knn_tuned"), ("ials_tuned", "popularity"), ("item_knn_tuned", "popularity")]


def paired_bootstrap(df: pd.DataFrame, pairs, n_boot: int, seed: int) -> dict:
    """Percentile 95% CIs on mean NDCG per model and on mean paired differences."""
    rng = np.random.default_rng(seed)
    vals = df.to_numpy()
    n = len(df)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = {c: vals[draws, j].mean(axis=1) for j, c in enumerate(df.columns)}
    out = {"n_users": n, "n_boot": n_boot, "seed": seed, "models": {}, "differences": {}}
    for c in df.columns:
        lo, hi = np.percentile(means[c], [2.5, 97.5])
        out["models"][c] = {"mean": float(df[c].mean()), "ci95": [float(lo), float(hi)]}
    for a, b in pairs:
        d = means[a] - means[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["differences"][f"{a} - {b}"] = {
            "observed": float(df[a].mean() - df[b].mean()), "ci95": [float(lo), float(hi)],
            "distinguishable": bool(lo > 0 or hi < 0),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="csv")
    ap.add_argument("--data", default="data/ml-1m")
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--split", choices=["temporal", "loo"], required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = load_dataset(args.source, args.data)
    tuning = json.loads((Path(args.tuned) / f"results_{args.split}_k10.json").read_text())["tuning"]
    train, test = (global_temporal_split(ds.interactions) if args.split == "temporal"
                   else leave_one_out_split(ds.interactions))
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
    if args.out:  # the refit is the expensive part: keep the per-user vectors so it is paid once
        frame = pd.DataFrame(cols)
        np.savez_compressed(Path(args.out).with_name(f"per_user_ndcg_{args.split}.npz"),
                            user_idx=frame.index.to_numpy(), **{c: frame[c].to_numpy() for c in frame})
    res = paired_bootstrap(pd.DataFrame(cols), PAIRS, args.n_boot, args.seed)
    res["split"] = args.split
    res["params"] = {"ials_tuned": tuning["ials"]["best_params"], "item_knn_tuned": tuning["item_knn"]["best_params"]}

    print(f"{args.split}: {res['n_users']} test users, {args.n_boot} paired bootstrap resamples")
    for c, v in res["models"].items():
        print(f"  {c:<15} NDCG@{args.k} {v['mean']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    for d, v in res["differences"].items():
        verdict = "distinguishable" if v["distinguishable"] else "NOT distinguishable (CI spans 0)"
        print(f"  {d:<30} {v['observed']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
