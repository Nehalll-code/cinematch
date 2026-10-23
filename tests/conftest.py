import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Hermetic: a developer's DATABASE_URL must never leak into tests (the API would try to verify a
# synthetic artifact against their real database). DB tests build their own engines explicitly.
os.environ.pop("DATABASE_URL", None)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic.py"),
         "--users", "200", "--items", "600", "--interactions", "12000", "--out", str(out)],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def dataset(synthetic_dir):
    from cinematch.data import load_movielens
    return load_movielens(synthetic_dir)


# --------------------------------------------------------------------------------------------
# Database fixtures. Every DB test runs against SQLite; the same tests also run against real
# PostgreSQL when CINEMATCH_TEST_DATABASE_URL is set (CI service container, or `make db-up`).
# Without it the Postgres variants are reported as skipped, not silently passed.
# --------------------------------------------------------------------------------------------


def _alembic_config(url: str):
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["url"] = url
    return cfg


@pytest.fixture(scope="session")
def alembic_config_for():
    return _alembic_config


@pytest.fixture(scope="session", params=["sqlite", "postgres"])
def migrated_engine(request, tmp_path_factory):
    """An engine whose schema was built by running the real Alembic migrations."""
    from alembic import command

    from cinematch.db.session import make_engine

    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path_factory.mktemp('db') / 'test.db'}"
    else:
        url = os.environ.get("CINEMATCH_TEST_DATABASE_URL")
        if not url:
            pytest.skip("CINEMATCH_TEST_DATABASE_URL not set; Postgres variant not run")
        if "test" not in url.rsplit("/", 1)[-1]:
            pytest.fail("refusing to run destructive DB tests against a database whose name lacks 'test'")
    engine = make_engine(url)
    cfg = _alembic_config(url)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def db(migrated_engine):
    """A session on a clean database: all rows removed and counters reset after each test."""
    from sqlalchemy import delete, update
    from sqlalchemy.orm import Session

    from cinematch.db.models import IndexCounter, Interaction, Movie, User, WatchlistItem

    with Session(migrated_engine, expire_on_commit=False) as session:
        yield session
        session.rollback()
        for model in (WatchlistItem, Interaction, Movie, User):
            session.execute(delete(model))
        session.execute(update(IndexCounter).values(next_value=0))
        session.commit()
