#!/usr/bin/env python3
"""Fit the production model and persist artifacts for the API to load.

The artifact carries the Dataset, whose item_ids/user_ids are the index space the model was
trained on. The API compares them with the database at startup and refuses to run on a mismatch
(cinematch.db.verify)."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cinematch.loading import SOURCES, load_dataset
from cinematch.recommenders import ContentRecommender, HybridRecommender, IALSRecommender


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="csv")
    ap.add_argument("--data", default="data/ml-1m", help="csv source only")
    ap.add_argument("--out", default="artifacts/model.pkl")
    ap.add_argument("--factors", type=int, default=64)
    # 0.0 is the weight chosen on a validation split carved from train (run_eval.py
    # --tune-hybrid) on both ml-1m splits; see README. The content model stays in the hybrid
    # because /similar reads its item features, but it contributes nothing to the blend.
    ap.add_argument("--content-weight", type=float, default=0.0)
    args = ap.parse_args()

    ds = load_dataset(args.source, args.data)
    content = ContentRecommender()
    ials = IALSRecommender(factors=args.factors, iterations=15)
    w = args.content_weight
    model = HybridRecommender({"content": (content, w), "ials": (ials, 1.0 - w)}, cold_threshold=3)
    # trained on the full history: the temporal holdout exists to estimate
    # quality, not to be left permanently unused at serving time
    model.fit(ds, ds.interactions)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump({"model": model, "dataset": ds}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
