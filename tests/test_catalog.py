"""Catalogue endpoints the frontend reads: film detail, search, taste profile, poster proxy."""
import subprocess
import sys
from collections import Counter

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import ROOT
from cinematch.db.ingest import ingest_movielens
from cinematch.db.models import Movie
from cinematch.posters import PosterService

SECRET = "s" * 40


@pytest.fixture(scope="module")
def artifact(synthetic_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp("art") / "model.pkl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(synthetic_dir),
                    "--out", str(path), "--factors", "8"], check=True, capture_output=True, cwd=ROOT)
    return path


@pytest.fixture
def plain(artifact, monkeypatch):
    """The API with a model and NO database: detail, search and the MovieLens-user profile."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("CINEMATCH_MODEL", str(artifact))
    from cinematch.api import app

    with TestClient(app) as c:
        yield c


# ---- film detail ------------------------------------------------------------------------------

def test_movie_detail_has_title_year_genres(plain):
    body = plain.get("/movies/1").json()
    assert body["movie_id"] == 1 and body["title"].startswith("Synthetic Film 1 (")
    assert body["year"] == 1950 and isinstance(body["genres"], list) and body["genres"]
    assert body["poster_available"] is False  # no database / TMDB configured


def test_unknown_movie_is_404(plain):
    assert plain.get("/movies/99999999").status_code == 404


# ---- search -----------------------------------------------------------------------------------

def test_search_is_case_insensitive_and_ranks_prefix_matches_first(plain):
    hits = plain.get("/search?q=synthetic film 1").json()
    assert hits and all("synthetic film 1" in h["title"].lower() for h in hits)
    prefix = [h["title"].lower().startswith("synthetic film 1") for h in hits]
    assert prefix == sorted(prefix, reverse=True)  # prefix matches precede mere substring matches


def test_search_respects_limit_and_validates_input(plain):
    assert len(plain.get("/search?q=film&limit=7").json()) == 7
    assert plain.get("/search?q=a").status_code == 422  # too short to be a useful query
    assert plain.get("/search?q=film&limit=500").status_code == 422
    assert plain.get("/search?q=zzzzzz-no-such-film").json() == []


# ---- taste profile (MovieLens user) -----------------------------------------------------------

def test_profile_is_the_content_models_real_tfidf_profile(plain):
    from cinematch.state import STATE

    body = plain.get("/profile/3").json()
    model, ds = STATE["model"], STATE["dataset"]
    content = model.components["content"][0]
    uidx = int(list(ds.user_ids).index(3))
    row = content.user_profiles_[uidx].toarray().ravel()
    vocab = content.vectorizer_.vocabulary_
    assert body["user_id"] == 3 and body["n_positive"] == int(model.user_counts_[uidx])
    for axis in body["axes"]:
        assert axis["value"] == pytest.approx(float(row[vocab[axis["genre"].lower()]]), abs=1e-6)
    assert 0.0 < body["genre_mass"] <= 1.0 + 1e-6  # share of the profile's squared norm sitting on genre terms


def test_profile_axes_cover_every_genre_in_a_stable_order(plain):
    from cinematch.state import STATE

    ds = STATE["dataset"]
    every = sorted({g for s in ds.items["genres"] for g in s.split("|") if g and g != "(no genres listed)"})
    assert [a["genre"] for a in plain.get("/profile/3").json()["axes"]] == every
    assert [a["genre"] for a in plain.get("/profile/4").json()["axes"]] == every


def test_the_users_top_genre_matches_what_they_actually_liked(plain):
    from cinematch.state import STATE

    model, ds = STATE["model"], STATE["dataset"]
    uidx = int(list(ds.user_ids).index(5))
    liked = model.train_matrix[uidx].indices
    counts = Counter(g for i in liked for g in ds.items["genres"].iloc[i].split("|"))
    axes = plain.get("/profile/5").json()["axes"]
    top3 = {a["genre"] for a in sorted(axes, key=lambda a: -a["value"])[:3]}
    assert counts.most_common(1)[0][0] in top3


def test_profile_for_an_unknown_user_is_404(plain):
    assert plain.get("/profile/99999999").status_code == 404


# ---- registered users' profile and the poster proxy need the database ---------------------------

@pytest.fixture
def served(db, migrated_engine, synthetic_dir, artifact, monkeypatch):
    ingest_movielens(db, synthetic_dir)
    db.commit()
    monkeypatch.setenv("DATABASE_URL", migrated_engine.url.render_as_string(hide_password=False))
    monkeypatch.setenv("CINEMATCH_MODEL", str(artifact))
    monkeypatch.setenv("CINEMATCH_JWT_SECRET", SECRET)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    from cinematch.api import app

    with TestClient(app) as c:
        yield c


def _auth(client):
    body = {"email": "ada@example.com", "password": "correct horse battery"}
    client.post("/auth/register", json=body)
    return {"Authorization": "Bearer " + client.post("/auth/login", json=body).json()["access_token"]}


def test_my_profile_reflects_only_my_positive_ratings(served):
    h = _auth(served)
    empty = served.get("/me/profile", headers=h).json()
    assert empty["n_positive"] == 0 and all(a["value"] == 0 for a in empty["axes"])  # real empty state
    for m in (1, 2, 3, 4):
        served.post("/interactions", headers=h, json={"movie_id": m, "type": "rating", "rating": 5.0})
    served.post("/interactions", headers=h, json={"movie_id": 9, "type": "rating", "rating": 1.0})  # not a positive
    after = served.get("/me/profile", headers=h).json()
    assert after["n_positive"] == 4 and any(a["value"] > 0 for a in after["axes"])


def test_my_profile_requires_auth(served):
    assert served.get("/me/profile").status_code == 401


# ---- poster proxy -------------------------------------------------------------------------------

def _service(tmp_path, handler, key="k" * 32):
    return PosterService(cache_dir=tmp_path / "posters", api_key=key,
                         client=httpx.Client(transport=httpx.MockTransport(handler)))


JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


def _tmdb(calls, poster_path="/abc.jpg", movie_status=200, image_status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url.copy_with(query=None)))
        if request.url.host == "api.themoviedb.org":
            if movie_status != 200:
                return httpx.Response(movie_status)
            return httpx.Response(200, json={"poster_path": poster_path})
        return httpx.Response(image_status, content=JPEG, headers={"content-type": "image/jpeg"})
    return handler


@pytest.fixture
def with_tmdb_ids(db, served):
    db.query(Movie).filter(Movie.movie_id == 1).update({"tmdb_id": 862})
    db.commit()
    return served


def _with_service(client, service):
    from cinematch.state import STATE
    STATE["posters"] = service
    return client


def test_poster_is_fetched_once_then_served_from_disk(with_tmdb_ids, tmp_path):
    calls = []
    client = _with_service(with_tmdb_ids, _service(tmp_path, _tmdb(calls)))
    first, second = client.get("/poster/1?w=342"), client.get("/poster/1?w=342")
    assert first.status_code == second.status_code == 200
    assert first.content == second.content == JPEG and first.headers["content-type"] == "image/jpeg"
    assert "max-age" in first.headers["cache-control"]
    assert len(calls) == 2  # one metadata lookup + one image, total, across both requests


def test_the_api_key_never_leaks_to_the_client(with_tmdb_ids, tmp_path):
    key = "super-secret-tmdb-key-0123456789"
    client = _with_service(with_tmdb_ids, _service(tmp_path, _tmdb([]), key=key))
    res = client.get("/poster/1")
    assert key not in res.text and key not in str(res.headers)


def test_a_v4_bearer_token_goes_in_a_header_not_the_url(with_tmdb_ids, tmp_path):
    seen = []
    token = "eyJhbGciOiJIUzI1NiJ9.payload.signature"

    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization")))
        return _tmdb([])(request)

    _with_service(with_tmdb_ids, _service(tmp_path, handler, key=token)).get("/poster/1")
    assert seen[0][1] == f"Bearer {token}" and token not in seen[0][0]


def test_poster_is_404_without_a_key_or_a_tmdb_id(with_tmdb_ids, tmp_path):
    from cinematch.state import STATE
    STATE["posters"] = None
    assert with_tmdb_ids.get("/poster/1").status_code == 404  # TMDB not configured
    _with_service(with_tmdb_ids, _service(tmp_path, _tmdb([])))
    assert with_tmdb_ids.get("/poster/2").status_code == 404  # movie 2 has no tmdb_id


def test_a_film_without_a_poster_is_negatively_cached(with_tmdb_ids, tmp_path):
    calls = []
    client = _with_service(with_tmdb_ids, _service(tmp_path, _tmdb(calls, poster_path=None)))
    assert client.get("/poster/1").status_code == 404
    assert client.get("/poster/1").status_code == 404
    assert len(calls) == 1  # asked TMDB once, remembered the answer


def test_an_upstream_failure_is_a_502_and_is_not_remembered(with_tmdb_ids, tmp_path):
    calls = []
    client = _with_service(with_tmdb_ids, _service(tmp_path, _tmdb(calls, movie_status=500)))
    assert client.get("/poster/1").status_code == 502
    assert client.get("/poster/1").status_code == 502
    assert len(calls) == 2  # a transient outage must not become a permanent "no poster"


def test_only_known_widths_are_allowed(with_tmdb_ids, tmp_path):
    client = _with_service(with_tmdb_ids, _service(tmp_path, _tmdb([])))
    assert client.get("/poster/1?w=342").status_code == 200
    assert client.get("/poster/1?w=9999").status_code == 422


def test_movie_detail_reports_poster_availability(with_tmdb_ids, tmp_path):
    client = _with_service(with_tmdb_ids, _service(tmp_path, _tmdb([])))
    assert client.get("/movies/1").json()["poster_available"] is True
    assert client.get("/movies/2").json()["poster_available"] is False  # no tmdb_id
