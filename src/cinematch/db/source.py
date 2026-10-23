"""Build a cinematch.data.Dataset from the database.

This is the DB twin of cinematch.data.load_movielens and must stay indistinguishable from it
(tests/test_db_source.py asserts identical frames, dtypes, order and timestamps). Both read the
SAME persistent index columns, so there is one index space, not two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data import POSITIVE_THRESHOLD, Dataset
from .models import Interaction, Movie, User

_EPOCH = pd.Timestamp(0, tz="UTC")


class IndexIntegrityError(RuntimeError):
    """The persisted index columns are not a contiguous 0..n-1 range."""


def _require_contiguous(idx: list[int], what: str) -> None:
    if idx != list(range(len(idx))):
        raise IndexIntegrityError(
            f"{what} is not contiguous 0..{len(idx) - 1}: rows were deleted or indices edited. "
            "Indices are append-only; a hole would shift every row after it in a trained model."
        )


def persisted_item_ids(session: Session) -> np.ndarray:
    """movie_id for item_idx 0..n-1, in index order."""
    rows = session.execute(select(Movie.item_idx, Movie.movie_id).order_by(Movie.item_idx)).all()
    _require_contiguous([r.item_idx for r in rows], "movies.item_idx")
    return np.array([r.movie_id for r in rows], dtype=np.int64)


def persisted_user_ids(session: Session) -> np.ndarray:
    """MovieLens id (or -users.id for registered accounts) for user_idx 0..n-1, in index order.

    Registered accounts have no MovieLens id; -users.id keeps the array unique and integer-typed,
    and cannot collide with a MovieLens id because those are positive.
    """
    rows = session.execute(select(User.user_idx, User.ml_user_id, User.id).order_by(User.user_idx)).all()
    _require_contiguous([r.user_idx for r in rows], "users.user_idx")
    return np.array([r.ml_user_id if r.ml_user_id is not None else -r.id for r in rows], dtype=np.int64)


def load_dataset_from_db(session: Session, threshold: float = POSITIVE_THRESHOLD) -> Dataset:
    movie_rows = session.execute(
        select(Movie.item_idx, Movie.movie_id, Movie.title, Movie.genres).order_by(Movie.item_idx)
    ).all()
    item_ids = persisted_item_ids(session)
    user_ids = persisted_user_ids(session)

    items = pd.DataFrame(
        {
            "item_idx": np.array([r.item_idx for r in movie_rows], dtype=np.int64),
            "movieId": item_ids,
            "title": [r.title for r in movie_rows],
            "genres": [r.genres for r in movie_rows],
        }
    )

    # ORDER BY interactions.id = insertion order = original file order, which the stable sort
    # below (and leave-one-out among tied timestamps) depends on.
    rows = session.execute(
        select(User.user_idx, Movie.item_idx, Interaction.rating, Interaction.created_at)
        .join(User, User.id == Interaction.user_id)
        .join(Movie, Movie.movie_id == Interaction.movie_id)
        .where(Interaction.type == "rating", Interaction.rating >= threshold)
        .order_by(Interaction.id)
    ).all()
    if not rows:
        raise ValueError(f"no interactions at or above threshold {threshold}")

    created = pd.to_datetime(pd.Series([r.created_at for r in rows]), utc=True)
    interactions = pd.DataFrame(
        {
            "user_idx": np.array([r.user_idx for r in rows], dtype=np.int32),
            "item_idx": np.array([r.item_idx for r in rows], dtype=np.int32),
            "rating": np.array([r.rating for r in rows], dtype=np.float32),
            "timestamp": ((created - _EPOCH) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64),
        }
    ).sort_values("timestamp", kind="mergesort", ignore_index=True)

    return Dataset(interactions=interactions, items=items, user_ids=user_ids, item_ids=item_ids)
