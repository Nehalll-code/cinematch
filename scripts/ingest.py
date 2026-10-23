#!/usr/bin/env python3
"""Load a MovieLens export into the database named by DATABASE_URL.

    docker compose up -d --wait && python -m alembic upgrade head
    python scripts/ingest.py --data data/ml-1m

Safe to re-run: existing movies/users keep their index and duplicate ratings are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy.orm import Session

from cinematch.db.ingest import ingest_movielens
from cinematch.db.session import make_engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ml-1m")
    args = ap.parse_args()

    engine = make_engine()
    t0 = time.perf_counter()
    with Session(engine) as session:
        stats = ingest_movielens(session, args.data)
        session.commit()  # one transaction: a failed import leaves no half-loaded catalogue
    print(f"ingested {stats.movies} movies, {stats.users} users, {stats.interactions} interactions "
          f"in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
