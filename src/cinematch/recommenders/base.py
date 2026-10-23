"""Common recommender interface.

Every model implements `fit` then `score_users`, returning a dense
(n_batch_users, n_items) score matrix. Masking of already-seen items is
the caller's job (see evaluate.py) so that models stay composable: the
hybrid needs raw, unmasked scores from its components.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data import Dataset


class Recommender(ABC):
    name: str = "recommender"

    def __init__(self) -> None:
        self.dataset: Dataset | None = None
        self.train_matrix: sp.csr_matrix | None = None

    @abstractmethod
    def fit(self, dataset: Dataset, train: pd.DataFrame) -> "Recommender":
        ...

    @abstractmethod
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        ...

    def score_from_items(self, item_indices: np.ndarray, cache_key=None) -> np.ndarray:
        """Score every item for a user described ONLY by the items they liked.

        This is how a user who was not in training gets personalised results without retraining:
        the fitted item-side parameters stay fixed and the user side is derived from the items.
        `cache_key` is an optional hashable (user, version-stamp) that models with an expensive
        derivation may use to reuse it. Returns raw scores of shape (n_items,); masking already-seen
        items is the caller's job, as everywhere else.
        """
        raise NotImplementedError(f"{self.name} cannot score a user from items alone")

    def recommend_with_scores(
        self, user_idx: int, k: int = 10, exclude_seen: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k item indices and their scores from ONE scoring pass.

        Callers that need to show scores (the API) must use this rather than calling
        `recommend` and then `score_users` again: scoring is the expensive part, and doing it twice
        defeats any cache placed in front.
        """
        scores = self.score_users(np.array([user_idx]))[0]
        if exclude_seen and self.train_matrix is not None:
            seen = self.train_matrix[user_idx].indices
            scores[seen] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top, scores[top]

    def recommend(self, user_idx: int, k: int = 10, exclude_seen: bool = True) -> np.ndarray:
        return self.recommend_with_scores(user_idx, k, exclude_seen)[0]


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Map each row to [0, 1] by rank, descending.

    Score-level fusion across heterogeneous models is only meaningful if
    the scales are comparable. Cosine similarity lives in [-1, 1], iALS
    dot products are unbounded, popularity counts are integers. Min-max
    is dominated by outliers; rank normalisation is scale-free and
    distribution-free, at the cost of discarding score margins.
    """
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(scores.shape[1])[None, :]
    n = scores.shape[1]
    return 1.0 - ranks / max(n - 1, 1)
