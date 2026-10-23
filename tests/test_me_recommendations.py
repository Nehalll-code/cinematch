"""Personalised recommendations for registered users via fold-in (no retraining)."""
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from conftest import ROOT
from cinematch.db.ingest import ingest_movielens
from cinematch.db.models import Movie

SECRET = "s" * 40
AUTH_BODY = {"email": "ada@example.com", "password": "correct horse battery"}


@pytest.fixture(scope="module")
def artifact(synthetic_dir, tmp_path_factory):
    # trained from CSV; each test re-ingests the same CSV, which reproduces the same index space,
    # so the startup index check passes without retraining per test
    path = tmp_path_factory.mktemp("art") / "model.pkl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(synthetic_dir),
                    "--out", str(path), "--factors", "8", "--content-weight", "0.0"],
                   check=True, capture_output=True, cwd=ROOT)
    return path


@pytest.fixture
def client(db, migrated_engine, synthetic_dir, artifact, monkeypatch):
    ingest_movielens(db, synthetic_dir)
    db.commit()
    monkeypatch.setenv("DATABASE_URL", migrated_engine.url.render_as_string(hide_password=False))
    monkeypatch.setenv("CINEMATCH_MODEL", str(artifact))
    monkeypatch.setenv("CINEMATCH_JWT_SECRET", SECRET)
    from cinematch.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def headers(client):
    client.post("/auth/register", json=AUTH_BODY)
    token = client.post("/auth/login", json=AUTH_BODY).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _rate(client, headers, movie_id, rating):
    res = client.post("/interactions", headers=headers, json={"movie_id": movie_id, "type": "rating", "rating": rating})
    assert res.status_code in (200, 201), res.text


def _recs(client, headers, k=10):
    res = client.get(f"/me/recommendations?k={k}", headers=headers)
    assert res.status_code == 200, res.text
    return res.json()


def _ids(body):
    return [m["movie_id"] for m in body["items"]]


def test_requires_authentication(client):
    assert client.get("/me/recommendations").status_code == 401


def test_a_brand_new_user_gets_popularity_flagged_as_cold_start(client, headers):
    body = _recs(client, headers, k=8)
    assert body["cold_start"] is True and len(body["items"]) == 8
    assert len(set(_ids(body))) == 8


def test_rating_a_few_films_makes_the_user_personalised_and_hides_what_they_rated(client, headers):
    cold = _recs(client, headers)
    liked = [1, 2, 3, 4, 5]
    for m in liked:
        _rate(client, headers, m, 5.0)
    warm = _recs(client, headers)
    assert warm["cold_start"] is False
    assert not set(liked) & set(_ids(warm))  # already-seen films are never recommended
    assert _ids(warm) != _ids(cold)  # the results actually changed


def test_low_ratings_and_watches_are_not_positives_but_are_still_hidden(client, headers):
    for m in (1, 2, 3, 4, 5):
        _rate(client, headers, m, 5.0)
    base = _recs(client, headers, k=100)
    target = _ids(base)[0]
    _rate(client, headers, target, 1.0)  # a 1-star is not a positive signal...
    client.post("/interactions", headers=headers, json={"movie_id": _ids(base)[1], "type": "watch"})
    after = _ids(_recs(client, headers, k=100))
    assert target not in after and _ids(base)[1] not in after  # ...but it has been seen, so it is hidden


def test_vector_is_reused_until_the_users_interactions_change(client, headers):
    from cinematch.state import STATE

    ials = STATE["model"].components["ials"][0]
    solves = []
    original = ials._solve_row
    ials._solve_row = lambda *a, **k: (solves.append(1), original(*a, **k))[1]
    try:
        for m in (1, 2, 3, 4):
            _rate(client, headers, m, 5.0)
        _recs(client, headers)
        _recs(client, headers)
        assert len(solves) == 1  # second request: cache hit, no solve
        _rate(client, headers, 6, 5.0)  # new interaction: count moves
        _recs(client, headers)
        assert len(solves) == 2
    finally:
        del ials._solve_row


def test_re_rating_changes_the_vector_even_though_the_interaction_count_does_not(client, headers):
    # why the cache stamp is (count, last-updated) and not the count alone: flipping a 5-star to a
    # 1-star leaves the number of interactions unchanged but removes a positive
    from cinematch.state import STATE

    ials = STATE["model"].components["ials"][0]
    solves = []
    original = ials._solve_row
    ials._solve_row = lambda *a, **k: (solves.append(1), original(*a, **k))[1]
    try:
        for m in (1, 2, 3, 4, 5):
            _rate(client, headers, m, 5.0)
        first = _recs(client, headers)
        _rate(client, headers, 5, 1.0)  # same count (5), different positive set
        second = _recs(client, headers)
        assert len(solves) == 2
        assert 5 not in _ids(second)  # still seen, so still hidden
        assert first["items"] != second["items"]
    finally:
        del ials._solve_row


def test_one_users_cache_entry_never_serves_another_user(client, headers):
    for m in (1, 2, 3, 4):
        _rate(client, headers, m, 5.0)
    mine = _recs(client, headers)
    other = {"email": "bob@example.com", "password": "correct horse battery"}
    client.post("/auth/register", json=other)
    token = client.post("/auth/login", json=other).json()["access_token"]
    bobs = _recs(client, {"Authorization": f"Bearer {token}"})
    assert bobs["cold_start"] is True and _ids(bobs) != _ids(mine)


def test_movies_added_after_training_are_ignored_not_fatal(client, headers, db):
    n_trained = db.scalar(select(Movie.item_idx).order_by(Movie.item_idx.desc()).limit(1)) + 1
    db.add(Movie(movie_id=999999, item_idx=n_trained, title="Newcomer (2030)", genres="Drama"))
    db.commit()
    for m in (1, 2, 3, 999999):
        _rate(client, headers, m, 5.0)
    body = _recs(client, headers)
    assert 999999 not in _ids(body) and len(body["items"]) == 10
