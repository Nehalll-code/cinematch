"""Engine and session construction.

The database URL comes from DATABASE_URL and has no default on purpose: a service that silently
falls back to some local file or localhost database will happily "work" against the wrong data.
"""
from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. For local development: "
            "postgresql+psycopg://cinematch:cinematch_dev@127.0.0.1:5432/cinematch (see .env.example)"
        )
    return url


def make_engine(url: str | None = None) -> Engine:
    engine = create_engine(url or database_url(), pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        # SQLite ignores foreign keys unless asked, per connection. Without this the SQLite-backed
        # tests would pass constraints that Postgres enforces.
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _record):  # noqa: ANN001
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
