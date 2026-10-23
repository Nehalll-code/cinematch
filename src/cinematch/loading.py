"""One entry point for "give me the Dataset", whichever backing store it lives in.

CSV is the default and needs nothing installed beyond the core requirements: the offline
evaluation must keep working with no database running. `db` reads the same data (and the same
persistent index columns) from the database named by DATABASE_URL.
"""
from __future__ import annotations

from pathlib import Path

from .data import Dataset, load_movielens

SOURCES = ("csv", "db")


def load_dataset(source: str, data_dir: str | Path | None = None) -> Dataset:
    if source == "csv":
        if data_dir is None:
            raise ValueError("--data is required with --source csv")
        return load_movielens(data_dir)
    if source == "db":
        # imported lazily so the CSV path never touches SQLAlchemy or a database driver
        from sqlalchemy.orm import Session

        from .db.session import make_engine
        from .db.source import load_dataset_from_db

        engine = make_engine()
        try:
            with Session(engine) as session:
                return load_dataset_from_db(session)
        finally:
            engine.dispose()
    raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
