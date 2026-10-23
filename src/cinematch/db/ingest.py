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
from sqlalchemy import bindparam, func, select, update
from sqlalchemy.orm import Session

from .dialect import insert_for
from .indexing import ITEM_IDX, USER_IDX, allocate
from .models import Interaction, Movie, User

CHUNK = 50_000


@dataclass
class IngestStats:
    movies: int
    users: int
    interactions: int


def ingest_movielens(session: Session, data_dir: str | Path) -> IngestStats:
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    if movies["movieId"].duplicated().any():
        raise ValueError("movies.csv contains duplicate movieId values")

    # same rule as the CSV loader: ratings of movies without metadata are dropped
    ratings = ratings[ratings["movieId"].isin(set(movies["movieId"]))]

    n_movies = _ingest_movies(session, movies)
    _ingest_links(session, data_dir, set(movies["movieId"]))
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
    session.execute(insert_for(session, Movie), rows)
    return len(rows)


def _ingest_links(session: Session, data_dir: Path, known_movies: set[int]) -> None:
    """Fill movies.tmdb_id from an optional links.csv (MovieLens: movieId, imdbId, tmdbId).

    Optional because ml-1m does not ship one (see scripts/download_links.py). Safe to re-run, and it
    also back-fills a catalogue that was ingested before the column existed.
    """
    path = data_dir / "links.csv"
    if not path.exists():
        return
    links = pd.read_csv(path)
    links = links[links["movieId"].isin(known_movies) & links["tmdbId"].notna()]
    if links.empty:
        return
    # Core table, not the ORM entity: with the entity, a list of parameter dicts is interpreted as an
    # ORM "bulk update by primary key" and rejects the WHERE bindparam.
    movies = Movie.__table__
    session.execute(
        update(movies).where(movies.c.movie_id == bindparam("b_movie_id")).values(tmdb_id=bindparam("b_tmdb_id")),
        [{"b_movie_id": int(m), "b_tmdb_id": int(t)} for m, t in zip(links["movieId"], links["tmdbId"])],
    )


def _ingest_users(session: Session, ml_ids: list[int]) -> tuple[int, dict[int, int]]:
    existing = {ml: pk for ml, pk in session.execute(select(User.ml_user_id, User.id)) if ml is not None}
    new = [u for u in ml_ids if u not in existing]
    if new:
        first = allocate(session, USER_IDX, len(new))
        session.execute(
            insert_for(session, User),
            [{"user_idx": first + i, "ml_user_id": int(u)} for i, u in enumerate(new)],
        )
    pk = {ml: pk for ml, pk in session.execute(select(User.ml_user_id, User.id)) if ml is not None}
    return len(new), pk


def _ingest_interactions(session: Session, ratings: pd.DataFrame, user_pk: dict[int, int]) -> int:
    before = session.scalar(select(func.count()).select_from(Interaction))
    if session.get_bind().dialect.name == "postgresql":
        _copy_interactions(session, ratings, user_pk)
    else:
        _executemany_interactions(session, ratings, user_pk)
    return session.scalar(select(func.count()).select_from(Interaction)) - before


def _rows(ratings: pd.DataFrame, user_pk: dict[int, int]):
    for u, m, r, t in zip(ratings["userId"], ratings["movieId"], ratings["rating"], ratings["timestamp"]):
        yield user_pk[int(u)], int(m), "rating", float(r), datetime.fromtimestamp(int(t), tz=timezone.utc)


def _executemany_interactions(session: Session, ratings: pd.DataFrame, user_pk: dict[int, int]) -> None:
    """Portable path (SQLite): batched INSERT .. ON CONFLICT DO NOTHING."""
    stmt = insert_for(session, Interaction).on_conflict_do_nothing(index_elements=["user_id", "movie_id", "type"])
    for start in range(0, len(ratings), CHUNK):
        part = ratings.iloc[start : start + CHUNK]
        session.execute(stmt, [
            {"user_id": u, "movie_id": m, "type": ty, "rating": r, "created_at": ts}
            for u, m, ty, r, ts in _rows(part, user_pk)
        ])


def _copy_interactions(session: Session, ratings: pd.DataFrame, user_pk: dict[int, int]) -> None:
    """PostgreSQL bulk path: COPY into a staging table, then one INSERT .. SELECT.

    Why not just COPY into `interactions`: COPY streams rows through a single protocol command
    with no per-statement parse/plan/round-trip cost (that is the whole speed-up over a
    parameterised INSERT per chunk), but it has no ON CONFLICT, so a re-run would fail on the
    first duplicate and the import would stop being idempotent. Staging keeps both: the stream is
    COPY, the dedup is the final INSERT. `ORDER BY ord` makes the serial ids follow file order,
    which leave-one-out among tied timestamps depends on (tests/test_db_source.py).
    The staging table is TEMP and ON COMMIT DROP, and everything shares the caller's transaction.
    """
    conn = session.connection().connection.driver_connection  # the psycopg connection, same transaction
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE _stage_interactions (ord bigint, user_id bigint, movie_id integer, "
            "type text, rating double precision, created_at timestamptz) ON COMMIT DROP"
        )
        with cur.copy("COPY _stage_interactions (ord, user_id, movie_id, type, rating, created_at) FROM STDIN") as cp:
            for i, row in enumerate(_rows(ratings, user_pk)):
                cp.write_row((i, *row))
        cur.execute(
            "INSERT INTO interactions (user_id, movie_id, type, rating, created_at) "
            "SELECT user_id, movie_id, type, rating, created_at FROM _stage_interactions ORDER BY ord "
            "ON CONFLICT (user_id, movie_id, type) DO NOTHING"
        )
