"""Popularity baseline.

The number every personalised model has to beat. Non-personalised,
trivially cheap, and on sparse data it is a genuinely hard baseline for
top-N ranking -- which is exactly why reporting it is not optional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Dataset
from .base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def fit(self, dataset: Dataset, train: pd.DataFrame) -> "PopularityRecommender":
        self.dataset = dataset
        self.train_matrix = dataset.matrix(train)
        counts = np.asarray(self.train_matrix.sum(axis=0)).ravel()
        self.scores_ = counts.astype(np.float32)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return np.tile(self.scores_, (len(user_indices), 1))

    def score_from_items(self, item_indices: np.ndarray, cache_key=None) -> np.ndarray:
        return self.scores_.copy()
