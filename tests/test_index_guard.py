"""Hard constraint: if the movieId -> item_idx mapping drifts, a pickled model points at the wrong
films and nothing errors. The guard turns that silent failure into a refusal to start."""
import os
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from conftest import ROOT
from cinematch.db.ingest import ingest_movielens
from cinematch.db.models import Movie, User
from cinematch.db.source import load_dataset_from_db
from cinematch.db.verify import IndexMismatchError, verify_artifact_index


@pytest.fixture
def trained(db, synthetic_dir):
    """(item_ids, user_ids) exactly as a model trained now would persist them."""
    ingest_movielens(db, synthetic_dir)
    db.commit()
    ds = load_dataset_from_db(db)
    return ds.item_ids.copy(), ds.user_ids.copy()


def test_matching_database_passes(db, trained):
    report = verify_artifact_index(db, *trained)
    assert report.items_added_since_training == 0 and report.users_added_since_training == 0


def test_rows_appended_after_training_are_allowed_and_reported(db, trained, tmp_path):
    # a brand-new movie whose id sorts BEFORE every existing one, rated by a brand-new user
    pd.DataFrame([(99999, 0, 5.0, 1_200_000_000)], columns=["userId", "movieId", "rating", "timestamp"]).to_csv(
        tmp_path / "ratings.csv", index=False)
    pd.DataFrame([(0, "Newcomer", "Drama")], columns=["movieId", "title", "genres"]).to_csv(
        tmp_path / "movies.csv", index=False)
    ingest_movielens(db, tmp_path)
    db.commit()
    report = verify_artifact_index(db, *trained)
    assert (report.items_added_since_training, report.users_added_since_training) == (1, 1)


def _swap_item_idx(db, a, b):
    ia = db.scalar(select(Movie.item_idx).where(Movie.movie_id == a))
    ib = db.scalar(select(Movie.item_idx).where(Movie.movie_id == b))
    db.execute(update(Movie).where(Movie.movie_id == a).values(item_idx=10**7))  # dodge UNIQUE
    db.execute(update(Movie).where(Movie.movie_id == b).values(item_idx=ia))
    db.execute(update(Movie).where(Movie.movie_id == a).values(item_idx=ib))
    db.commit()


def test_two_swapped_item_indices_are_detected(db, trained):
    item_ids, _ = trained
    _swap_item_idx(db, int(item_ids[3]), int(item_ids[4]))
    with pytest.raises(IndexMismatchError, match="item_idx 3"):
        verify_artifact_index(db, *trained)


def test_user_index_drift_is_detected(db, trained):
    _, user_ids = trained
    u0, u1 = db.scalars(select(User).where(User.user_idx.in_([0, 1])).order_by(User.user_idx)).all()
    db.execute(update(User).where(User.id == u0.id).values(user_idx=10**7))
    db.execute(update(User).where(User.id == u1.id).values(user_idx=0))
    db.execute(update(User).where(User.id == u0.id).values(user_idx=1))
    db.commit()
    with pytest.raises(IndexMismatchError, match="user_idx 0"):
        verify_artifact_index(db, *trained)


def test_database_with_fewer_rows_than_the_model_knows_is_a_mismatch(db, trained):
    item_ids, user_ids = trained
    longer = list(item_ids) + [123456789]
    with pytest.raises(IndexMismatchError, match="fewer"):
        verify_artifact_index(db, longer, user_ids)


# ---- end to end: the service itself refuses to start --------------------------------------------

@pytest.fixture
def served_artifact(db, migrated_engine, synthetic_dir, tmp_path):
    ingest_movielens(db, synthetic_dir)
    db.commit()
    url = migrated_engine.url.render_as_string(hide_password=False)
    art = tmp_path / "model.pkl"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--source", "db", "--out", str(art), "--factors", "8"],
        check=True, capture_output=True, env={**os.environ, "DATABASE_URL": url}, cwd=ROOT,
    )
    return url, art


def _start(url, art, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("CINEMATCH_MODEL", str(art))
    monkeypatch.setenv("CINEMATCH_JWT_SECRET", "x" * 40)
    from cinematch.api import app

    return TestClient(app)


def test_service_starts_when_the_mapping_matches(served_artifact, monkeypatch):
    url, art = served_artifact
    with _start(url, art, monkeypatch) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["index_verified"] is True


def test_service_refuses_to_start_when_the_mapping_drifted(db, served_artifact, monkeypatch):
    url, art = served_artifact
    item_ids = db.scalars(select(Movie.movie_id).order_by(Movie.item_idx)).all()
    _swap_item_idx(db, item_ids[0], item_ids[1])
    with pytest.raises(IndexMismatchError):
        with _start(url, art, monkeypatch):
            pytest.fail("service started against a drifted mapping")
