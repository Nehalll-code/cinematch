"""Read-only catalogue endpoints the frontend uses: film detail, search, taste profile, poster."""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import posters as posters_mod
from .accounts import current_user, get_session
from .db.models import Movie, User
from .state import STATE
from .user_data import load_user_items

router = APIRouter()
_YEAR = re.compile(r"\((\d{4})\)\s*$")
NO_GENRES = "(no genres listed)"


class MovieDetail(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    poster_available: bool


class SearchHit(BaseModel):
    movie_id: int
    title: str
    year: int | None
    genres: list[str]


class ProfileAxis(BaseModel):
    genre: str
    value: float


class TasteProfile(BaseModel):
    user_id: int
    n_positive: int  # liked films the profile was built from; 0 means there is no profile yet
    genre_mass: float  # share of the profile vector's squared norm that sits on genre terms
    axes: list[ProfileAxis]


def _model_and_dataset():
    model, ds = STATE.get("model"), STATE.get("dataset")
    if model is None:
        raise HTTPException(503, "model artifact not loaded; run scripts/train.py")
    return model, ds


def _optional_session():
    factory = STATE.get("session_factory")
    if factory is None:
        yield None
        return
    with factory() as session:
        yield session


def _year(title: str) -> int | None:
    m = _YEAR.search(title)
    return int(m.group(1)) if m else None


def _genres(raw: str) -> list[str]:
    return [g for g in str(raw).split("|") if g and g != NO_GENRES]


def _all_genres(ds) -> list[str]:
    """Every genre label in the catalogue, alphabetically: the radar's fixed axes."""
    if "genres" not in STATE:
        STATE["genres"] = sorted({g for raw in ds.items["genres"] for g in _genres(raw)})
    return STATE["genres"]


def _row_index(ds, movie_id: int) -> int:
    import numpy as np

    match = np.where(ds.item_ids == movie_id)[0]
    if len(match) == 0:
        raise HTTPException(404, f"unknown movie {movie_id}")
    return int(match[0])


@router.get("/movies/{movie_id}", response_model=MovieDetail)
def movie_detail(movie_id: int, session: Session | None = Depends(_optional_session)) -> MovieDetail:
    _, ds = _model_and_dataset()
    row = ds.items.iloc[_row_index(ds, movie_id)]
    # "available" = a poster could be served: TMDB configured AND we know this film's tmdbId
    tmdb_id = session.get(Movie, movie_id).tmdb_id if session is not None and session.get(Movie, movie_id) else None
    return MovieDetail(
        movie_id=movie_id, title=str(row["title"]), year=_year(str(row["title"])), genres=_genres(row["genres"]),
        poster_available=STATE.get("posters") is not None and tmdb_id is not None,
    )


@router.get("/search", response_model=list[SearchHit])
def search(q: str = Query(min_length=2, max_length=100), limit: int = Query(20, ge=1, le=50)) -> list[SearchHit]:
    """Title substring search, case-insensitive; titles that START with the query come first."""
    _, ds = _model_and_dataset()
    if "titles_lower" not in STATE:
        STATE["titles_lower"] = ds.items["title"].str.lower().tolist()
    needle = q.strip().lower()
    hits = [(0 if t.startswith(needle) else 1, t, i) for i, t in enumerate(STATE["titles_lower"]) if needle in t]
    hits.sort()
    out = []
    for _, _, i in hits[:limit]:
        row = ds.items.iloc[i]
        out.append(SearchHit(movie_id=int(row["movieId"]), title=str(row["title"]),
                             year=_year(str(row["title"])), genres=_genres(row["genres"])))
    return out


def _taste_profile(model, ds, user_id: int, n_positive: int, profile_row) -> TasteProfile:
    """Genre axes read off the content model's REAL user profile (a normalised mean TF-IDF vector).

    The profile also has title-word and release-year dimensions; the radar shows only the genre
    components, and `genre_mass` reports how much of the profile that is, so the chart is never
    mistaken for the whole vector. Values are the profile's own weights, not re-scaled: a rarer
    genre carries a higher IDF, so a small cluster of liked rare-genre films can outweigh a larger
    cluster of liked dramas. That is how this model sees taste, and the chart should say so.
    """
    comp = model.components.get("content")
    if comp is None:
        raise HTTPException(404, "the served model has no content component")
    content = comp[0]
    vec = profile_row.toarray().ravel()
    vocab = content.vectorizer_.vocabulary_
    axes = [ProfileAxis(genre=g, value=float(vec[vocab[g.lower()]]) if g.lower() in vocab else 0.0)
            for g in _all_genres(ds)]
    norm2 = float(vec @ vec)
    genre_mass = float(sum(a.value ** 2 for a in axes) / norm2) if norm2 > 0 else 0.0
    return TasteProfile(user_id=user_id, n_positive=n_positive, genre_mass=genre_mass, axes=axes)


@router.get("/profile/{user_id}", response_model=TasteProfile)
def profile(user_id: int) -> TasteProfile:
    """Taste profile of a MovieLens user, from the profile the content model built at training."""
    import numpy as np

    model, ds = _model_and_dataset()
    match = np.where(ds.user_ids == user_id)[0]
    if len(match) == 0:
        raise HTTPException(404, f"unknown user {user_id}")
    uidx = int(match[0])
    content = model.components["content"][0] if "content" in model.components else None
    if content is None:
        raise HTTPException(404, "the served model has no content component")
    return _taste_profile(model, ds, user_id, int(model.user_counts_[uidx]), content.user_profiles_[uidx])


@router.get("/me/profile", response_model=TasteProfile)
def my_profile(user: User = Depends(current_user), session: Session = Depends(get_session)) -> TasteProfile:
    """Taste profile of the signed-in user, built on the fly from their own positive ratings."""
    model, ds = _model_and_dataset()
    content = model.components["content"][0] if "content" in model.components else None
    if content is None:
        raise HTTPException(404, "the served model has no content component")
    _, _, positives = load_user_items(session, user, ds.n_items)
    return _taste_profile(model, ds, user.id, len(positives), content.profile_from_items(positives))


@router.get("/poster/{movie_id}")
def poster(
    movie_id: int, w: int = Query(342), session: Session | None = Depends(_optional_session),
) -> Response:
    """The film's poster image, fetched from TMDB by the server and cached on disk."""
    if w not in posters_mod.ALLOWED_WIDTHS:
        raise HTTPException(422, f"w must be one of {posters_mod.ALLOWED_WIDTHS}")
    service = STATE.get("posters")
    if service is None or session is None:
        raise HTTPException(404, "poster source not configured")
    movie = session.get(Movie, movie_id)
    if movie is None or movie.tmdb_id is None:
        raise HTTPException(404, "no poster for this film")
    try:
        data = service.get(movie.tmdb_id, w)
    except posters_mod.PosterUnavailable:
        raise HTTPException(404, "no poster for this film") from None
    except posters_mod.UpstreamError:
        raise HTTPException(502, "poster source unavailable") from None
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
