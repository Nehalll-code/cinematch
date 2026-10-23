#!/usr/bin/env python3
"""Generate a synthetic dataset in MovieLens schema.

Purpose: let the full pipeline and test suite run without network access.
Numbers produced on this data are NOT comparable to published MovieLens
results and must not be reported as such -- it exists to prove the code
runs and the metrics behave, nothing else.

Generative model: each item has a genre vector and a latent quality;
each user has a latent genre preference and an activity window. A user
rates an item with probability driven by item popularity (power law) and
user-item affinity, which gives all three model families real signal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=800)
    ap.add_argument("--items", type=int, default=3000)
    ap.add_argument("--interactions", type=int, default=90_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data_synthetic")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_u, n_i, n_g = args.users, args.items, len(GENRES)

    item_genres = np.zeros((n_i, n_g))
    for i in range(n_i):
        picks = rng.choice(n_g, size=rng.integers(1, 4), replace=False)
        item_genres[i, picks] = 1.0
    item_genres /= np.linalg.norm(item_genres, axis=1, keepdims=True)

    user_pref = rng.dirichlet(np.ones(n_g) * 0.4, size=n_u)
    user_pref /= np.linalg.norm(user_pref, axis=1, keepdims=True)

    quality = rng.normal(0.0, 1.0, n_i)
    pop_weight = rng.pareto(1.3, n_i) + 1.0
    pop_weight /= pop_weight.sum()

    affinity = user_pref @ item_genres.T  # (users, items)

    rows = []
    base_ts = 1_100_000_000
    per_user = rng.multinomial(args.interactions, np.ones(n_u) / n_u)
    for u in range(n_u):
        n = int(per_user[u])
        if n == 0:
            continue
        logits = 3.0 * affinity[u] + 0.6 * quality + 2.5 * np.log(pop_weight * n_i + 1e-9)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        items = rng.choice(n_i, size=min(n, n_i), replace=False, p=p)
        latent = 3.0 + 2.5 * affinity[u, items] + 0.5 * quality[items] + rng.normal(0, 0.6, len(items))
        ratings = np.clip(np.round(latent * 2) / 2, 0.5, 5.0)
        start = base_ts + int(rng.integers(0, 60_000_000))
        ts = np.sort(start + rng.integers(0, 40_000_000, len(items)))
        rows.append(pd.DataFrame({"userId": u + 1, "movieId": items + 1, "rating": ratings, "timestamp": ts}))

    ratings_df = pd.concat(rows, ignore_index=True).sort_values("timestamp", ignore_index=True)
    titles = [f"Synthetic Film {i+1} ({1950 + (i % 70)})" for i in range(n_i)]
    genre_strs = ["|".join(np.array(GENRES)[item_genres[i] > 0]) for i in range(n_i)]
    movies_df = pd.DataFrame({"movieId": np.arange(1, n_i + 1), "title": titles, "genres": genre_strs})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(out / "ratings.csv", index=False)
    movies_df.to_csv(out / "movies.csv", index=False)
    pos = (ratings_df["rating"] >= 4.0).sum()
    print(f"{len(ratings_df)} interactions, {pos} positive (>=4.0), {n_u} users, {n_i} items -> {out}")


if __name__ == "__main__":
    main()
