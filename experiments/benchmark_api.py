#!/usr/bin/env python3
"""Measure cold vs cached recommendation latency in-process.

In-process, not over HTTP: this isolates the recommender and the cache
from network and ASGI overhead. A number produced here is a lower bound
on what a deployed service does, and should be described that way.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model.pkl")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    import pickle

    with open(args.model, "rb") as fh:
        blob = pickle.load(fh)
    model, ds = blob["model"], blob["dataset"]

    rng = np.random.default_rng(0)
    users = rng.integers(0, ds.n_users, args.requests)

    cold = []
    for u in users:
        t0 = time.perf_counter()
        model.recommend(int(u), k=args.k)
        cold.append((time.perf_counter() - t0) * 1000)

    from functools import lru_cache

    @lru_cache(maxsize=4096)
    def cached(u: int, k: int):
        return tuple(model.recommend(u, k=k))

    for u in users:
        cached(int(u), args.k)  # warm
    warm = []
    for u in users:
        t0 = time.perf_counter()
        cached(int(u), args.k)
        warm.append((time.perf_counter() - t0) * 1000)

    def summarize(label, xs):
        xs = sorted(xs)
        print(f"{label:<8} n={len(xs):<5} mean={statistics.mean(xs):8.3f}ms  "
              f"p50={xs[len(xs)//2]:8.3f}ms  p95={xs[int(len(xs)*0.95)]:8.3f}ms  "
              f"p99={xs[int(len(xs)*0.99)]:8.3f}ms")

    print(f"catalogue: {ds.n_items} items | {ds.n_users} users | k={args.k}")
    summarize("cold", cold)
    summarize("cached", warm)
    print(f"speedup (p95): {sorted(cold)[int(len(cold)*0.95)] / max(sorted(warm)[int(len(warm)*0.95)], 1e-9):.0f}x")


if __name__ == "__main__":
    main()
