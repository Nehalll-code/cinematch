"""Weighted score-level hybrid with a cold-start popularity fallback.

Components are rank-normalised before fusion so that incomparable score
scales do not let one model dominate by magnitude alone (see
base.rank_normalize).

The fallback is the part that matters in production and is usually
missing from coursework versions: a user with fewer than
`cold_threshold` training interactions gets popularity, because a
profile built from one or two items is noise and the personalised models
will confidently rank garbage for them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from .base import Recommender, rank_normalize
from .popularity import PopularityRecommender


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(self, components: dict[str, tuple[Recommender, float]], cold_threshold: int = 3) -> None:
        super().__init__()
        self.components = components
        self.cold_threshold = cold_threshold
        self.fallback = PopularityRecommender()

    def fit(self, dataset: Dataset, train: pd.DataFrame) -> "HybridRecommender":
        self.dataset = dataset
        self.train_matrix = dataset.matrix(train)
        for rec, _ in self.components.values():
            if getattr(rec, "dataset", None) is None:
                rec.fit(dataset, train)
        self.fallback.fit(dataset, train)
        self.user_counts_ = np.asarray(self.train_matrix.sum(axis=1)).ravel()
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        total_w = sum(w for _, w in self.components.values())
        out = np.zeros((len(user_indices), self.dataset.n_items), dtype=np.float32)
        for rec, weight in self.components.values():
            out += (weight / total_w) * rank_normalize(rec.score_users(user_indices))
        cold = self.user_counts_[user_indices] < self.cold_threshold
        if cold.any():
            out[cold] = rank_normalize(self.fallback.score_users(user_indices[cold]))
        return out

    def score_from_items(self, item_indices: np.ndarray, cache_key=None) -> np.ndarray:
        """`score_users` for a user known only by the items they liked (e.g. registered after training).

        Same rule as the batch path: below `cold_threshold` liked items the user gets popularity;
        otherwise a weighted blend of rank-normalised component scores. Zero-weight components are
        skipped (they would contribute exactly 0), which also avoids their cost.
        """
        item_indices = np.asarray(item_indices, dtype=np.int64)
        if len(item_indices) < self.cold_threshold:
            return rank_normalize(self.fallback.scores_[None, :])[0]
        total_w = sum(w for _, w in self.components.values())
        out = np.zeros(self.dataset.n_items, dtype=np.float32)
        for rec, weight in self.components.values():
            if weight == 0:
                continue
            out += (weight / total_w) * rank_normalize(rec.score_from_items(item_indices, cache_key)[None, :])[0]
        return out
