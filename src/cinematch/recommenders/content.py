"""Content-based model: TF-IDF over item metadata, cosine against a user profile.

The user profile is the mean of the TF-IDF vectors of items they liked
in training, L2-normalised. Scoring is then a single sparse matmul.

Cold items are reachable (they have metadata even with no interactions),
which is the one thing collaborative filtering structurally cannot do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..data import Dataset
from .base import Recommender


# words (letters, hyphens, apostrophes) or any run of digits
TOKEN_PATTERN = r"[A-Za-z][A-Za-z\-']+|\d+"


class ContentRecommender(Recommender):
    name = "content"

    def __init__(self, max_features: int | None = None) -> None:
        super().__init__()
        self.max_features = max_features

    @staticmethod
    def _documents(items: pd.DataFrame) -> list[str]:
        # Genres are pipe-delimited in MovieLens. The release year is in the title ("Heat (1995)")
        # and era plausibly clusters taste, so the whole title is kept and TOKEN_PATTERN admits
        # digits. (It used to be letters-only, which silently dropped the year while this comment
        # claimed otherwise.) Each exact year is its own token, so years are matched exactly, not
        # bucketed into decades.
        genres = items["genres"].fillna("").str.replace("|", " ", regex=False)
        titles = items["title"].fillna("")
        return (genres + " " + genres + " " + titles).tolist()  # genres weighted x2

    def fit(self, dataset: Dataset, train: pd.DataFrame) -> "ContentRecommender":
        self.dataset = dataset
        self.train_matrix = dataset.matrix(train)
        vec = TfidfVectorizer(max_features=self.max_features, token_pattern=TOKEN_PATTERN)
        self.item_features_ = normalize(vec.fit_transform(self._documents(dataset.items)))
        self.vectorizer_ = vec
        # profile = normalised mean of liked item vectors
        counts = np.asarray(self.train_matrix.sum(axis=1)).ravel()
        inv = sp.diags(1.0 / np.maximum(counts, 1.0))
        self.user_profiles_ = normalize(inv @ self.train_matrix @ self.item_features_)
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        prof = self.user_profiles_[user_indices]
        return np.asarray((prof @ self.item_features_.T).todense(), dtype=np.float32)
