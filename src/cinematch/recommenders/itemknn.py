"""Item-item collaborative filtering with cosine similarity and top-k pruning.

Included because on small, sparse datasets a well-tuned item-kNN
regularly beats matrix factorisation, and omitting it is how people end
up claiming MF "wins" when it was never tested against the strong
neighbourhood baseline. Shrinkage damps similarities estimated from few
co-occurrences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from ..data import Dataset
from .base import Recommender


class ItemKNNRecommender(Recommender):
    name = "item_knn"

    def __init__(self, k: int = 100, shrink: float = 10.0) -> None:
        super().__init__()
        self.k = k
        self.shrink = shrink

    def fit(self, dataset: Dataset, train: pd.DataFrame) -> "ItemKNNRecommender":
        self.dataset = dataset
        self.train_matrix = dataset.matrix(train)
        X = self.train_matrix.tocsc()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        co = (X.T @ X).toarray().astype(np.float32)  # co-occurrence counts
        denom = np.outer(norms, norms) + self.shrink + 1e-9
        sim = co / denom
        np.fill_diagonal(sim, 0.0)
        # prune to top-k neighbours per item: removes long-tail noise and
        # keeps the similarity matrix sparse enough to be worth storing
        if self.k < sim.shape[1]:
            keep = np.argpartition(-sim, self.k, axis=1)[:, : self.k]
            pruned = np.zeros_like(sim)
            rows = np.arange(sim.shape[0])[:, None]
            pruned[rows, keep] = sim[rows, keep]
            sim = pruned
        self.similarity_ = sp.csr_matrix(sim)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        profile = self.train_matrix[user_indices]
        return np.asarray((profile @ self.similarity_).todense(), dtype=np.float32)

    def score_from_items(self, item_indices: np.ndarray, cache_key=None) -> np.ndarray:
        profile = sp.csr_matrix(
            (np.ones(len(item_indices), dtype=np.float32), (np.zeros(len(item_indices), dtype=int), item_indices)),
            shape=(1, self.similarity_.shape[0]),
        )
        return np.asarray((profile @ self.similarity_).todense(), dtype=np.float32).ravel()
