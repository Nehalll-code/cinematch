import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(synthetic_dir, tmp_path_factory):
    art = tmp_path_factory.mktemp("art") / "model.pkl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"),
                    "--data", str(synthetic_dir), "--out", str(art), "--factors", "16"],
                   check=True, capture_output=True)
    os.environ["CINEMATCH_MODEL"] = str(art)
    from cinematch.api import app
    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["n_items"] > 0


def test_recommend_returns_k_distinct_movies(client):
    body = client.get("/recommend/1?k=5").json()
    assert len(body["items"]) == 5
    assert len({m["movie_id"] for m in body["items"]}) == 5
    assert all(isinstance(m["title"], str) and m["title"] for m in body["items"])


def test_recommend_scores_are_monotonically_non_increasing(client):
    scores = [m["score"] for m in client.get("/recommend/1?k=10").json()["items"]]
    assert scores == sorted(scores, reverse=True)


def test_unknown_user_is_404(client):
    assert client.get("/recommend/99999999").status_code == 404


def test_k_is_validated(client):
    assert client.get("/recommend/1?k=0").status_code == 422
    assert client.get("/recommend/1?k=500").status_code == 422


def test_second_identical_request_is_served_from_cache(client):
    client.get("/recommend/2?k=10")
    assert client.get("/recommend/2?k=10").json()["cached"] is True


def test_metrics_expose_latency_percentiles(client):
    for _ in range(5):
        client.get("/recommend/3?k=10")
    body = client.get("/metrics").json()
    assert body["requests"] >= 5
    # cache hits are legitimately sub-microsecond, so p50 may be ~0;
    # the invariant that must hold is percentile ordering
    assert body["p95_ms"] >= body["p50_ms"] >= 0
    assert body["p99_ms"] >= body["p95_ms"]


def test_similar_excludes_the_query_movie(client):
    body = client.get("/similar/1?k=5").json()
    assert len(body) == 5 and all(m["movie_id"] != 1 for m in body)
