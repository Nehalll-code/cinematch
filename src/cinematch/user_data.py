"""Authenticated write paths: a user's own interactions and watchlist.

These rows are STORED, not learned from: nothing here reaches the trained model, and there is no
scheduled retraining. They exist so a future retrain has data, and so the schema is exercised.
The caller's identity always comes from the token, never from the request body.

/me/recommendations is the exception to "stored, not learned from": it personalises a registered
user by FOLD-IN (cinematch.recommenders.ials.IALSRecommender.fold_in), one least-squares solve
against the fixed item factors. The item factors themselves never learn from these users.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .accounts import current_user, get_session
from .db.dialect import insert_for
from .data import POSITIVE_THRESHOLD
from .db.models import Interaction, Movie, User, WatchlistItem
from .schemas import Recommendations, movie_from
from .state import STATE

router = APIRouter()

LIST_LIMIT = 200


class InteractionIn(BaseModel):
    # forbid: a client sending user_id (or anything else unexpected) gets a 422, not silence
    model_config = ConfigDict(extra="forbid")

    movie_id: int
    type: Literal["rating", "watch", "dismiss"]
    rating: float | None = None

    @model_validator(mode="after")
    def _rating_matches_type(self) -> "InteractionIn":
        # mirrors the DB CHECK constraint so bad input is a 422 with a reason, not a 500
        if self.type == "rating":
            if self.rating is None:
                raise ValueError("rating is required when type is 'rating'")
            if not 0.5 <= self.rating <= 5.0 or (self.rating * 2) % 1 != 0:
                raise ValueError("rating must be between 0.5 and 5.0 in steps of 0.5")
        elif self.rating is not None:
            raise ValueError(f"rating is not allowed when type is {self.type!r}")
        return self


class InteractionOut(BaseModel):
    movie_id: int
    type: str
    rating: float | None
    created_at: datetime


class WatchlistEntry(BaseModel):
    movie_id: int
    title: str
    added_at: datetime


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)  # SQLite drops tzinfo


def _require_movie(session: Session, movie_id: int) -> None:
    if session.get(Movie, movie_id) is None:
        raise HTTPException(404, f"unknown movie {movie_id}")


@router.post("/interactions", response_model=InteractionOut)
def record_interaction(
    body: InteractionIn, response: Response,
    user: User = Depends(current_user), session: Session = Depends(get_session),
) -> InteractionOut:
    _require_movie(session, body.movie_id)
    existing = session.scalar(
        select(Interaction.id).where(
            Interaction.user_id == user.id, Interaction.movie_id == body.movie_id, Interaction.type == body.type
        )
    )
    now = datetime.now(timezone.utc)
    # One current row per (user, movie, type): re-rating overwrites and moves created_at forward,
    # because the latest event time is what the temporal split would cut on.
    stmt = insert_for(session, Interaction).values(
        user_id=user.id, movie_id=body.movie_id, type=body.type, rating=body.rating, created_at=now
    )
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=["user_id", "movie_id", "type"],
            set_={"rating": stmt.excluded.rating, "created_at": stmt.excluded.created_at},
        )
    )
    session.commit()
    response.status_code = 200 if existing is not None else 201
    return InteractionOut(movie_id=body.movie_id, type=body.type, rating=body.rating, created_at=now)


@router.get("/interactions", response_model=list[InteractionOut])
def list_interactions(user: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.scalars(
        select(Interaction).where(Interaction.user_id == user.id)
        .order_by(Interaction.created_at.desc(), Interaction.id.desc()).limit(LIST_LIMIT)
    ).all()
    return [InteractionOut(movie_id=r.movie_id, type=r.type, rating=r.rating, created_at=_utc(r.created_at)) for r in rows]


@router.put("/watchlist/{movie_id}", status_code=204)
def add_to_watchlist(
    movie_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)
) -> Response:
    _require_movie(session, movie_id)
    session.execute(
        insert_for(session, WatchlistItem)
        .values(user_id=user.id, movie_id=movie_id)
        .on_conflict_do_nothing(index_elements=["user_id", "movie_id"])  # PUT is idempotent
    )
    session.commit()
    return Response(status_code=204)


@router.delete("/watchlist/{movie_id}", status_code=204)
def remove_from_watchlist(
    movie_id: int, user: User = Depends(current_user), session: Session = Depends(get_session)
) -> Response:
    session.execute(delete(WatchlistItem).where(WatchlistItem.user_id == user.id, WatchlistItem.movie_id == movie_id))
    session.commit()
    return Response(status_code=204)


@router.get("/watchlist", response_model=list[WatchlistEntry])
def list_watchlist(user: User = Depends(current_user), session: Session = Depends(get_session)):
    rows = session.execute(
        select(WatchlistItem.movie_id, Movie.title, WatchlistItem.added_at)
        .join(Movie, Movie.movie_id == WatchlistItem.movie_id)
        .where(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.added_at.desc(), WatchlistItem.movie_id)
    ).all()
    return [WatchlistEntry(movie_id=r.movie_id, title=r.title, added_at=_utc(r.added_at)) for r in rows]


@router.get("/me/recommendations", response_model=Recommendations)
def my_recommendations(
    k: int = Query(10, ge=1, le=100),
    user: User = Depends(current_user), session: Session = Depends(get_session),
) -> Recommendations:
    """Personalised top-k for the signed-in user, from their own interactions.

    Positives are ratings >= 4.0 (the same rule as training); every interaction of any type or
    rating marks a film as seen and hides it. Below the model's cold threshold of positives the
    user gets popularity, flagged cold_start. Films added to the catalogue after the model was
    trained have no factors and are ignored. `cached` is always False here: there is no result
    cache for this endpoint, only the fold-in vector cache inside the model.
    """
    model, ds = STATE.get("model"), STATE.get("dataset")
    if model is None:
        raise HTTPException(503, "model artifact not loaded; run scripts/train.py")
    t0 = time.perf_counter()

    rows = session.execute(
        select(Movie.item_idx, Interaction.type, Interaction.rating, Interaction.created_at)
        .join(Movie, Movie.movie_id == Interaction.movie_id)
        .where(Interaction.user_id == user.id)
    ).all()
    known = [r for r in rows if r.item_idx < ds.n_items]
    seen = np.array(sorted({r.item_idx for r in known}), dtype=np.int64)
    positives = np.array(
        sorted({r.item_idx for r in known if r.type == "rating" and r.rating >= POSITIVE_THRESHOLD}), dtype=np.int64
    )

    # Version stamp for the cached latent vector. The interaction COUNT alone is not enough:
    # re-rating a film from 5 stars to 1 leaves the count unchanged but removes a positive, and the
    # stale vector would keep recommending as if it were still liked. Every write also moves
    # created_at (upserts overwrite it), so (count, newest timestamp) changes on any write.
    newest = max((r.created_at for r in rows), default=None)
    stamp = (len(rows), newest.isoformat() if newest is not None else None)

    scores = model.score_from_items(positives, cache_key=(user.id, stamp)).astype(np.float32, copy=True)
    scores[seen] = -np.inf
    kk = min(k, len(scores))
    top = np.argpartition(-scores, kk - 1)[:kk]
    top = top[np.argsort(-scores[top])]

    return Recommendations(
        user_id=user.id,
        cold_start=bool(len(positives) < model.cold_threshold),
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        cached=False,
        items=[movie_from(ds, int(i), float(scores[i])) for i in top],
    )
