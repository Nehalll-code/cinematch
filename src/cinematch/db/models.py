"""Relational schema. SQLAlchemy 2.0 typed declarative.

Three decisions worth being able to defend:

ONE PERSISTENT INDEX SPACE. `users.user_idx` and `movies.item_idx` are the row/column positions
a trained model uses. They are assigned once, from `index_counters`, and never change or get
reused. If they were derived from the data on each load (as the CSV loader once did), a single new
rating would renumber everything after it and a pickled model would silently point at the wrong
films. The artifact stores the movie_id vector it was trained on and the API refuses to start if
the database disagrees (see cinematch.db.verify).

ONE USERS TABLE. The offline eval needs the 6,040 MovieLens users (no email, no password); the
product needs registered accounts. A single table with nullable credentials keeps `interactions`
on a single foreign key. Check constraints stop a row from being both unreachable and
unimportable.

CHECK CONSTRAINTS, NOT PG ENUMS. `interactions.type` is a VARCHAR with a CHECK. Adding a value to
a native Postgres ENUM needs ALTER TYPE inside a migration and cannot be rolled back cleanly in
one transaction; changing a CHECK is an ordinary drop/create.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Stable constraint names: Alembic needs them to drop/alter constraints by name later.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# SQLite only autoincrements a column declared exactly INTEGER PRIMARY KEY; Postgres wants BIGINT.
BigIntPK = BigInteger().with_variant(Integer, "sqlite")

INTERACTION_TYPES = ("rating", "watch")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class IndexCounter(Base):
    """Next unused value of an append-only index space ('user_idx', 'item_idx').

    A counter, not MAX(col)+1: MAX+1 reuses an index if the highest row is ever deleted, and two
    concurrent inserts can both read the same MAX. The UPDATE that bumps this row takes a row
    lock, which serialises allocators, and it rolls back with the surrounding transaction so a
    failed insert leaves no gap.
    """

    __tablename__ = "index_counters"

    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    next_value: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("email IS NOT NULL OR ml_user_id IS NOT NULL", name="has_identity"),
        # Emails are lowercased by the app; the constraint makes the unique index case-insensitive
        # without a functional index or the citext extension.
        CheckConstraint("email IS NULL OR email = lower(email)", name="email_lowercase"),
        CheckConstraint("password_hash IS NULL OR email IS NOT NULL", name="password_needs_email"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_idx: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    ml_user_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    email: Mapped[str | None] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Movie(Base):
    __tablename__ = "movies"

    # MovieLens movieId is the primary key: it is already the id the API exposes (/similar/{id}),
    # and a second surrogate id would be a second id space to keep straight.
    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    item_idx: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    # Pipe-delimited exactly as MovieLens ships it: the content model tokenises this string, and
    # a normalised genre table would be reassembled into the same string on every training run.
    genres: Mapped[str] = mapped_column(Text, nullable=False, server_default="")


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        CheckConstraint("type IN ('rating', 'watch')", name="type_known"),
        CheckConstraint(
            "(type = 'rating' AND rating IS NOT NULL AND rating >= 0.5 AND rating <= 5.0)"
            " OR (type <> 'rating' AND rating IS NULL)",
            name="rating_matches_type",
        ),
        # One current row per (user, movie, type): re-rating updates in place. This loses rating
        # history; it is what keeps the offline eval's "one interaction per user-item" invariant.
        # The unique index's leading column is user_id, so it also serves "all of a user's
        # interactions" and no separate user_id index is needed.
        UniqueConstraint("user_id", "movie_id", "type", name="uq_interactions_user_movie_type"),
        Index("ix_interactions_movie_id", "movie_id"),
    )

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.movie_id", ondelete="RESTRICT"), nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    rating: Mapped[float | None] = mapped_column(Float)
    # Real event time. Ingest keeps MovieLens' original timestamp here: the temporal split cuts
    # on it, so stamping imports with now() would destroy the evaluation.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class WatchlistItem(Base):
    __tablename__ = "watchlist"

    # Composite PK: a movie is on a user's watchlist at most once, and the PK's leading user_id
    # makes "my watchlist" an index range scan.
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.movie_id", ondelete="RESTRICT"), primary_key=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
