"""Refuse to serve a model whose index space no longer matches the database.

A trained model addresses items and users by POSITION: column j of its item-factor matrix belongs
to whichever movie held item_idx j when it was trained. If the database now assigns j to a
different movie, scoring still succeeds and returns plausible-looking, wrong recommendations.
Nothing raises. This check is what turns that silent failure into a hard one at startup.

What is and is not a mismatch:
  * database has MORE movies/users than the model knows -> fine. They were appended after
    training (indices are append-only); the model simply cannot recommend them yet.
  * database disagrees at ANY index the model knows, or has FEWER rows -> mismatch, fatal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from .source import persisted_item_ids, persisted_user_ids


class IndexMismatchError(RuntimeError):
    """The database's index space disagrees with the one the model was trained on."""


@dataclass
class IndexReport:
    items_added_since_training: int
    users_added_since_training: int


def _compare(trained: np.ndarray, db: np.ndarray, what: str, idx_name: str) -> int:
    trained = np.asarray(trained, dtype=np.int64)
    if len(db) < len(trained):
        raise IndexMismatchError(
            f"database has fewer {what} ({len(db)}) than the model was trained on ({len(trained)}); "
            "rows were deleted, so every later index would point at the wrong row"
        )
    bad = np.flatnonzero(db[: len(trained)] != trained)
    if len(bad):
        shown = "; ".join(f"{idx_name} {i}: model={int(trained[i])} db={int(db[i])}" for i in bad[:5])
        raise IndexMismatchError(
            f"{len(bad)} of {len(trained)} {what} map to a different id than at training time "
            f"(first: {shown}). Retrain the model against this database, or restore the original mapping."
        )
    return len(db) - len(trained)


def verify_artifact_index(session: Session, item_ids: np.ndarray, user_ids: np.ndarray) -> IndexReport:
    items_added = _compare(item_ids, persisted_item_ids(session), "movies", "item_idx")
    users_added = _compare(user_ids, persisted_user_ids(session), "users", "user_idx")
    return IndexReport(items_added_since_training=items_added, users_added_since_training=users_added)
