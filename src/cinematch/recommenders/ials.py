"""Implicit-feedback matrix factorisation (iALS, Hu/Koren/Volinsky 2008).

Explicit-rating MF (predict the star rating, minimise RMSE) is the wrong
objective here: we binarised to positives, and unobserved entries are
missing-not-at-random, not negatives. iALS handles this by treating every
unobserved cell as a zero-preference observation with low confidence, and
observed cells as one-preference with confidence 1 + alpha * r.

The alternating least-squares step uses the standard Gramian trick:
precompute YtY once per half-iteration so each user solve is
O(f^2 * |I_u| + f^3) rather than O(f^2 * n_items).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data import Dataset
from .base import Recommender


class IALSRecommender(Recommender):
    name = "ials"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 20.0,
        iterations: int = 15,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, dataset: Dataset, train: pd.DataFrame) -> "IALSRecommender":
        self.dataset = dataset
        self.train_matrix = dataset.matrix(train)
        Cui = (self.train_matrix * self.alpha).tocsr()
        Ciu = Cui.T.tocsr()

        rng = np.random.default_rng(self.random_state)
        f = self.factors
        self.user_factors_ = rng.normal(0, 0.01, (dataset.n_users, f)).astype(np.float64)
        self.item_factors_ = rng.normal(0, 0.01, (dataset.n_items, f)).astype(np.float64)

        for _ in range(self.iterations):
            self._solve(self.user_factors_, self.item_factors_, Cui)
            self._solve(self.item_factors_, self.user_factors_, Ciu)
        self.user_factors_ = self.user_factors_.astype(np.float32)
        self.item_factors_ = self.item_factors_.astype(np.float32)
        return self

    def _solve(self, X: np.ndarray, Y: np.ndarray, C: sp.csr_matrix) -> None:
        f = X.shape[1]
        YtY = Y.T @ Y + self.regularization * np.eye(f)
        indptr, indices, data = C.indptr, C.indices, C.data
        for row in range(X.shape[0]):
            lo, hi = indptr[row], indptr[row + 1]
            if lo == hi:
                X[row] = 0.0
                continue
            idx = indices[lo:hi]
            conf = data[lo:hi]  # this is alpha * r; full confidence is 1 + that
            Yi = Y[idx]
            # A = YtY + Yi^T diag(conf) Yi ; b = Yi^T (conf + 1)
            A = YtY + (Yi.T * conf) @ Yi
            b = Yi.T @ (conf + 1.0)
            X[row] = np.linalg.solve(A, b)

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.user_factors_[user_indices] @ self.item_factors_.T).astype(np.float32)
