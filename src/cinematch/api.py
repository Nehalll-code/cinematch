"""FastAPI serving layer.

Scope note, stated plainly because it matters for how this should be
read: there is no PostgreSQL and no Redis here. Artifacts are loaded
into process memory at startup and recommendations are memoised in an
LRU. That is the honest shape of a single-node service; adding a
database and an external cache changes the operational story, not the
recommendation quality, and is listed as future work rather than
claimed.

Latency is measured in-process and exposed at /metrics so the numbers
reported anywhere else can be reproduced rather than asserted.
"""
from __future__ import annotations

import os
import pickle
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

STATE: dict = {}
LATENCIES: list[float] = []


class Movie(BaseModel):
    movie_id: int
    title: str
    genres: list[str]
    score: float


class Recommendations(BaseModel):
    user_id: int
    cold_start: bool
    latency_ms: float
    cached: bool
    items: list[Movie]


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("CINEMATCH_MODEL", "artifacts/model.pkl"))
    if path.exists():
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        STATE["model"] = blob["model"]
        STATE["dataset"] = blob["dataset"]
    yield
    STATE.clear()


app = FastAPI(title="CineMatch", version="0.1.0", lifespan=lifespan)


def _require_model():
    if "model" not in STATE:
        raise HTTPException(503, "model artifact not loaded; run scripts/train.py")
    return STATE["model"], STATE["dataset"]


@lru_cache(maxsize=4096)
def _cached_recommend(user_idx: int, k: int) -> tuple[int, ...]:
    model, _ = _require_model()
    return tuple(int(i) for i in model.recommend(user_idx, k=k))


@app.get("/health")
def health() -> dict:
    loaded = "model" in STATE
    ds = STATE.get("dataset")
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "n_users": getattr(ds, "n_users", 0),
        "n_items": getattr(ds, "n_items", 0),
    }


@app.get("/metrics")
def metrics() -> dict:
    if not LATENCIES:
        return {"requests": 0}
    arr = np.array(LATENCIES)
    return {
        "requests": len(arr),
        # 4dp: cache hits are sub-microsecond and round to zero at 2dp,
        # which would make the cold/warm comparison unreadable
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "cache": _cached_recommend.cache_info()._asdict(),
    }


@app.get("/recommend/{user_id}", response_model=Recommendations)
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)) -> Recommendations:
    model, ds = _require_model()
    match = np.where(ds.user_ids == user_id)[0]
    if len(match) == 0:
        raise HTTPException(404, f"unknown user {user_id}")
    user_idx = int(match[0])

    before = _cached_recommend.cache_info().hits
    t0 = time.perf_counter()
    item_idxs = _cached_recommend(user_idx, k)
    elapsed = (time.perf_counter() - t0) * 1000
    LATENCIES.append(elapsed)
    cached = _cached_recommend.cache_info().hits > before

    scores = model.score_users(np.array([user_idx]))[0]
    return Recommendations(
        user_id=user_id,
        cold_start=bool(model.user_counts_[user_idx] < model.cold_threshold),
        latency_ms=round(elapsed, 3),
        cached=cached,
        items=[_movie(ds, i, float(scores[i])) for i in item_idxs],
    )


@app.get("/similar/{movie_id}", response_model=list[Movie])
def similar(movie_id: int, k: int = Query(10, ge=1, le=100)) -> list[Movie]:
    _, ds = _require_model()
    model = STATE["model"]
    match = np.where(ds.item_ids == movie_id)[0]
    if len(match) == 0:
        raise HTTPException(404, f"unknown movie {movie_id}")
    idx = int(match[0])
    content = model.components["content"][0]
    sims = np.asarray((content.item_features_[idx] @ content.item_features_.T).todense()).ravel()
    sims[idx] = -np.inf
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [_movie(ds, int(i), float(sims[i])) for i in top]


def _movie(ds, item_idx: int, score: float) -> Movie:
    row = ds.items.iloc[item_idx]
    return Movie(
        movie_id=int(row["movieId"]),
        title=str(row["title"]),
        genres=[g for g in str(row["genres"]).split("|") if g and g != "(no genres listed)"],
        score=round(score, 5),
    )
