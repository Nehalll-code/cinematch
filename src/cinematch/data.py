"""Dataset loading and interaction-matrix construction.

MovieLens ships explicit 0.5-5.0 ratings. For top-N ranking we binarize:
a rating >= POSITIVE_THRESHOLD is a positive signal, everything else is
discarded (not treated as a negative). Rationale: we are evaluating
ranking utility, not rating reconstruction, and RMSE on explicit ratings
is a poor proxy for top-N quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

POSITIVE_THRESHOLD = 4.0


@dataclass
class Dataset:
    """Interactions plus item metadata, with contiguous internal indices."""

    interactions: pd.DataFrame  # user_idx, item_idx, rating, timestamp
    items: pd.DataFrame  # item_idx, movie_id, title, genres
    user_ids: np.ndarray  # user_idx -> original userId
    item_ids: np.ndarray  # item_idx -> original movieId

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def matrix(self, df: pd.DataFrame | None = None) -> sp.csr_matrix:
        """Binary user-item CSR matrix over the given interaction frame."""
        d = self.interactions if df is None else df
        return sp.csr_matrix(
            (np.ones(len(d), dtype=np.float32), (d["user_idx"].to_numpy(), d["item_idx"].to_numpy())),
            shape=(self.n_users, self.n_items),
        )


def load_movielens(data_dir: str | Path, threshold: float = POSITIVE_THRESHOLD) -> Dataset:
    """Load a MovieLens export (ml-latest-small / ml-25m layout: ratings.csv, movies.csv)."""
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    return _assemble(ratings, movies, threshold)


def _assemble(ratings: pd.DataFrame, movies: pd.DataFrame, threshold: float) -> Dataset:
    ratings = ratings[ratings["rating"] >= threshold].copy()
    if ratings.empty:
        raise ValueError(f"no interactions at or above threshold {threshold}")

    # Keep only movies we have metadata for; content model needs genres.
    ratings = ratings[ratings["movieId"].isin(set(movies["movieId"]))]

    user_ids = np.sort(ratings["userId"].unique())
    item_ids = np.sort(ratings["movieId"].unique())
    umap = {u: i for i, u in enumerate(user_ids)}
    imap = {m: i for i, m in enumerate(item_ids)}

    interactions = pd.DataFrame(
        {
            "user_idx": ratings["userId"].map(umap).to_numpy(dtype=np.int32),
            "item_idx": ratings["movieId"].map(imap).to_numpy(dtype=np.int32),
            "rating": ratings["rating"].to_numpy(dtype=np.float32),
            "timestamp": ratings["timestamp"].to_numpy(dtype=np.int64),
        }
    ).sort_values("timestamp", kind="mergesort", ignore_index=True)

    items = movies[movies["movieId"].isin(set(item_ids))].copy()
    items["item_idx"] = items["movieId"].map(imap)
    items = items.sort_values("item_idx", ignore_index=True)[["item_idx", "movieId", "title", "genres"]]

    return Dataset(interactions=interactions, items=items, user_ids=user_ids, item_ids=item_ids)
