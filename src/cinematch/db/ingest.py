"""Import a MovieLens export (ratings.csv + movies.csv) into the database.

Idempotent and append-only: re-running inserts nothing new, and movies/users that already exist
keep the index they were first given. New ones are numbered in ascending-id order from the
counters, so a FRESH import reproduces exactly the mapping cinematch.data.load_movielens builds
(sorted by id), while later imports append instead of renumbering.

Interactions keep the original MovieLens timestamps in `created_at`: the temporal split cuts on
them. Rows are inserted in file order so `interactions.id` preserves it, which matters for
leave-one-out among equal timestamps (see cinematch.db.source).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .indexing import ITEM_IDX, USER_IDX, allocate
from .models import Interaction, Movie, User

CHUNK = 50_000


@dataclass
class IngestStats:
    movies: int
    users: int
    interactions: int


def _insert(session: Session, model):
    # ON CONFLICT DO NOTHING exists in both dialects but under different modules.
    if session.get_bind().dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(model)


def ingest_movielens(session: Session, data_dir: str | Path) -> IngestStats:
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    if movies["movieId"].duplicated().any():
        raise ValueError("movies.csv contains duplicate movieId values")

    # same rule as the CSV loader: ratings of movies without metadata are dropped
    ratings = ratings[ratings["movieId"].isin(set(movies["movieId"]))]

    n_movies = _ingest_movies(session, movies)
    n_users, user_pk = _ingest_users(session, sorted(ratings["userId"].unique().tolist()))
    n_inter = _ingest_interactions(session, ratings, user_pk)
    session.flush()
    return IngestStats(movies=n_movies, users=n_users, interactions=n_inter)


def _ingest_movies(session: Session, movies: pd.DataFrame) -> int:
    existing = set(session.scalars(select(Movie.movie_id)))
    new = movies[~movies["movieId"].isin(existing)].sort_values("movieId")
    if new.empty:
        return 0
    first = allocate(session, ITEM_IDX, len(new))
    rows = [
        {"movie_id": int(r.movieId), "item_idx": first + i, "title": str(r.title),
         "genres": "" if pd.isna(r.genres) else str(r.genres)}
        for i, r in enumerate(new.itertuples(index=False))
    ]
    session.execute(_insert(session, Movie), rows)
    return len(rows)


def _ingest_users(session: Session, ml_ids: list[int]) -> tuple[int, dict[int, int]]:
    existing = {ml: pk for ml, pk in session.execute(select(User.ml_user_id, User.id)) if ml is not None}
    new = [u for u in ml_ids if u not in existing]
    if new:
        first = allocate(session, USER_IDX, len(new))
        session.execute(
            _insert(session, User),
            [{"user_idx": first + i, "ml_user_id": int(u)} for i, u in enumerate(new)],
        )
    pk = {ml: pk for ml, pk in session.execute(select(User.ml_user_id, User.id)) if ml is not None}
    return len(new), pk


def _ingest_interactions(session: Session, ratings: pd.DataFrame, user_pk: dict[int, int]) -> int:
    before = session.scalar(select(func.count()).select_from(Interaction))
    stmt = _insert(session, Interaction).on_conflict_do_nothing(
        index_elements=["user_id", "movie_id", "type"]
    )
    for start in range(0, len(ratings), CHUNK):
        part = ratings.iloc[start : start + CHUNK]
        rows = [
            {
                "user_id": user_pk[int(u)], "movie_id": int(m), "type": "rating", "rating": float(r),
                "created_at": datetime.fromtimestamp(int(t), tz=timezone.utc),
            }
            for u, m, r, t in zip(part["userId"], part["movieId"], part["rating"], part["timestamp"])
        ]
        session.execute(stmt, rows)
    return session.scalar(select(func.count()).select_from(Interaction)) - before
