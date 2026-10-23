"""Response shapes shared by the recommendation endpoints."""
from __future__ import annotations

from pydantic import BaseModel


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


def movie_from(ds, item_idx: int, score: float) -> Movie:
    row = ds.items.iloc[item_idx]
    return Movie(
        movie_id=int(row["movieId"]),
        title=str(row["title"]),
        genres=[g for g in str(row["genres"]).split("|") if g and g != "(no genres listed)"],
        score=round(score, 5),
    )
