"""Implicit-feedback matrix factorisation (iALS, Hu/Koren/Volinsky 2008).

Explicit-rating MF (predict the star rating, minimise RMSE) is the wrong
objective here: we binarised to positives, and unobserved entries are
missing-not-at-random, not negatives. iALS handles this by treating every
unobserved cell as a zero-preference observation with low confidence, and
observed cells as one-preference with confidence 1 + alpha * r.

The alternating least-squares step uses the standard Gramian trick:
precompute YtY once per half-iteration so each user solve is
O(f^2 * |I_u| + f^3) rather than O(f^2 * n_items).

FOLD-IN. The user half-step of ALS solves one user's latent vector against FIXED item factors. That
is exactly what is needed for a user who was not in training: `fold_in` runs that one solve
(milliseconds) and the user is personalised without retraining. What it gives up: the item
factors never learn from the new user, so item-side drift only gets fixed by a full retrain.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data import Dataset
from .base import Recommender


_FOLD_IN_LOCK = threading.Lock()  # module-level: locks cannot be pickled with the model


class IALSRecommender(Recommender):
    name = "ials"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 20.0,
        iterations: int = 15,
        random_state: int = 42,
        fold_in_cache_size: int = 10_000,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.fold_in_cache_size = fold_in_cache_size
        self._fold_in_cache: OrderedDict = OrderedDict()

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

    def _solve_row(self, YtY: np.ndarray, Y: np.ndarray, idx: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """One row's regularised least-squares solve. `conf` is alpha*r; full confidence is 1 + conf.

        A = YtY + Yi^T diag(conf) Yi ;  b = Yi^T (conf + 1). Shared by training (_solve) and fold-in
        so there is exactly one implementation of the math.
        """
        Yi = Y[idx]
        A = YtY + (Yi.T * conf) @ Yi
        b = Yi.T @ (conf + 1.0)
        return np.linalg.solve(A, b)

    def _solve(self, X: np.ndarray, Y: np.ndarray, C: sp.csr_matrix) -> None:
        f = X.shape[1]
        YtY = Y.T @ Y + self.regularization * np.eye(f)
        indptr, indices, data = C.indptr, C.indices, C.data
        for row in range(X.shape[0]):
            lo, hi = indptr[row], indptr[row + 1]
            if lo == hi:
                X[row] = 0.0
                continue
            X[row] = self._solve_row(YtY, Y, indices[lo:hi], data[lo:hi])

    # ---- fold-in ---------------------------------------------------------------------------

    def _fixed_item_side(self) -> tuple[np.ndarray, np.ndarray]:
        """(regularised Gramian, float64 item factors), computed once. Lazy so artifacts trained
        before fold-in existed (which lack them) still work."""
        if getattr(self, "_item_side", None) is None:
            Y = self.item_factors_.astype(np.float64)
            self._item_side = (Y.T @ Y + self.regularization * np.eye(Y.shape[1]), Y)
        return self._item_side

    def _cache(self) -> OrderedDict:
        # artifacts pickled before fold-in existed have neither the cache nor its size setting
        if not hasattr(self, "_fold_in_cache"):
            self._fold_in_cache = OrderedDict()
        return self._fold_in_cache

    def fold_in(self, item_indices: np.ndarray, cache_key: tuple | None = None) -> np.ndarray:
        """Latent vector for a user described by the items they liked, against the fixed item factors.

        `cache_key` is (subject, stamp), e.g. (user_id, version-of-their-interactions). The vector
        is reused while the stamp is unchanged and recomputed when it moves; one entry per subject,
        bounded LRU. A stamp must change whenever the user's liked-item SET can change (see the API:
        the interaction count alone is not enough, because re-rating keeps the count).
        """
        if cache_key is not None:
            subject, stamp = cache_key
            with _FOLD_IN_LOCK:
                cache = self._cache()
                hit = cache.get(subject)
                if hit is not None and hit[0] == stamp:
                    cache.move_to_end(subject)
                    return hit[1]

        idx = np.unique(np.asarray(item_indices, dtype=np.int64))
        if len(idx) == 0:
            vec = np.zeros(self.factors, dtype=np.float32)
        else:
            YtY, Y = self._fixed_item_side()
            conf = np.full(len(idx), self.alpha)  # binary positives: alpha * 1, as in training
            vec = self._solve_row(YtY, Y, idx, conf).astype(np.float32)

        if cache_key is not None:
            with _FOLD_IN_LOCK:
                cache = self._cache()
                cache[subject] = (stamp, vec)
                cache.move_to_end(subject)
                while len(cache) > getattr(self, "fold_in_cache_size", 10_000):
                    cache.popitem(last=False)
        return vec

    def score_from_items(self, item_indices: np.ndarray, cache_key: tuple | None = None) -> np.ndarray:
        return (self.item_factors_ @ self.fold_in(item_indices, cache_key)).astype(np.float32)

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.user_factors_[user_indices] @ self.item_factors_.T).astype(np.float32)
