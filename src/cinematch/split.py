"""Leakage-safe train/test splitting.

Two strategies, and the difference is the thing to be able to defend:

GLOBAL TEMPORAL (default). Sort every interaction in the dataset by
timestamp and cut at a quantile. Training data is strictly in the past
relative to every test interaction. This is the only split here that
matches deployment: at serving time you have yesterday's data and are
ranking for tomorrow.

LEAVE-ONE-OUT (per user). Hold out each user's single most recent
interaction. It is the convention in much of the recsys literature and
gives denser per-user evaluation, but it leaks: user A's held-out item
may predate user B's training items, so the model sees the future
relative to some test points. Provided for comparability with published
numbers, not as the headline metric.

A random split is not offered. On interaction data it leaks hard and
inflates every metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def global_temporal_split(
    interactions: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut the timeline at the (1 - test_frac) quantile of timestamps."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    d = interactions.sort_values("timestamp", kind="mergesort", ignore_index=True)
    cut = int(len(d) * (1.0 - test_frac))
    cut_ts = d["timestamp"].iloc[cut]
    # Split on the timestamp value, not the row index, so interactions
    # sharing a timestamp never straddle the boundary.
    train = d[d["timestamp"] < cut_ts].reset_index(drop=True)
    test = d[d["timestamp"] >= cut_ts].reset_index(drop=True)
    return train, test


def leave_one_out_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent interaction."""
    d = interactions.sort_values(["user_idx", "timestamp"], kind="mergesort", ignore_index=True)
    is_last = d.groupby("user_idx", sort=False).cumcount(ascending=False) == 0
    # Users with a single interaction contribute to train only; holding out
    # their one item would leave nothing to profile them with.
    counts = d.groupby("user_idx", sort=False)["item_idx"].transform("size")
    holdout = is_last & (counts > 1)
    return d[~holdout].reset_index(drop=True), d[holdout].reset_index(drop=True)


def filter_test_to_known_users(
    train: pd.DataFrame, test: pd.DataFrame, min_train_interactions: int = 1
) -> pd.DataFrame:
    """Drop test rows for users with too little training history.

    Cold-start users are a separate problem with a separate answer
    (the popularity fallback). Mixing them into the main table makes
    every personalised model look worse than it is, and hides the
    fallback's contribution.
    """
    counts = train.groupby("user_idx")["item_idx"].size()
    eligible = set(counts[counts >= min_train_interactions].index)
    return test[test["user_idx"].isin(eligible)].reset_index(drop=True)


def ground_truth(test: pd.DataFrame) -> dict[int, np.ndarray]:
    """user_idx -> array of held-out item indices."""
    return {int(u): g["item_idx"].to_numpy() for u, g in test.groupby("user_idx", sort=True)}
