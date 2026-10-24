#!/usr/bin/env python3
"""Close a grid-edge optimum with a probe: fit iALS at extra `factors` values on the VALIDATION split.

Reuses the other tuned hyperparameters from the earlier results JSON and the same validation split
`run_eval.py --tune-models` used (tune_model carves it from train), so the new trials are directly
comparable to the stored ones. Test data is never passed in.

    python experiments/probe_factors.py --tuned artifacts/tuned_v4 --split temporal --factors 256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cinematch.loading import SOURCES, load_dataset
from cinematch.recommenders import IALSRecommender
from cinematch.split import global_temporal_split, leave_one_out_split
from cinematch.tuning import tune_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="csv")
    ap.add_argument("--data", default="data/ml-1m")
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--split", choices=["temporal", "loo"], required=True)
    ap.add_argument("--factors", type=int, nargs="+", default=[256])
    ap.add_argument("--alphas", type=float, nargs="*", default=None, help="also probe these alphas at the tuned reg/factors")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    ds = load_dataset(args.source, args.data)
    old = json.loads((Path(args.tuned) / f"results_{args.split}_k10.json").read_text())["tuning"]["ials"]
    best = old["best_params"]
    train, _ = global_temporal_split(ds.interactions) if args.split == "temporal" else leave_one_out_split(ds.interactions)

    def make(**p):
        return IALSRecommender(**{**dict(iterations=15), **p})

    grid = {"factors": args.factors, "regularization": [best["regularization"]], "alpha": [best["alpha"]]}
    if args.alphas:
        grid = {"factors": sorted({best["factors"], *args.factors}), "regularization": [best["regularization"]],
                "alpha": args.alphas}
    res = tune_model(ds, train, args.split, make_model=make, grid=grid, k=args.k, prior_trials=old["trials"])
    print(f"{args.split}: previous best {best} val_ndcg={old['best_val_ndcg']:.4f}")
    new = [t for t in res.trials if t not in old["trials"]]
    for t in new:
        print(f"  probe {t['params']} val_ndcg={t['ndcg']:.4f} fit={t['fit_seconds']}s")
    print(f"best after probe: {res.best_params} val_ndcg={res.best_ndcg:.4f} edge_of_grid={res.edge_params or 'no'}")


if __name__ == "__main__":
    main()
