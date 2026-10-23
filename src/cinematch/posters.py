"""Poster images from TMDB, fetched by the server and served from our own origin.

WHY A PROXY. A TMDB key placed in a Vite env var is compiled into the public JavaScript bundle, so
anyone can read it. The key therefore lives only in the server's environment (TMDB_API_KEY), the
browser asks *us* for /poster/{movie_id}, and nothing is hotlinked from image.tmdb.org.

CACHING, and its three outcomes. A fetched image is written to disk once and served from there.
  * poster found       -> image cached; later requests never touch TMDB.
  * TMDB has no poster -> a marker file is written (negative cache), so a film that will never have
                          a poster does not cost two upstream calls on every page view.
  * network/5xx error  -> NOT cached, returned as 502: a transient outage must not turn into a
                          permanent "no poster".
Writes go to a temp file then os.replace, so a concurrent reader never sees a half-written image.

Per TMDB's terms, any product using the API must show: "This product uses the TMDB API but is not
endorsed or certified by TMDB." The frontend footer does.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx

API = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p"
ALLOWED_WIDTHS = (185, 342, 500)
TIMEOUT = httpx.Timeout(6.0)


class PosterUnavailable(Exception):
    """The film has no poster (unknown to TMDB, or TMDB lists none)."""


class UpstreamError(Exception):
    """TMDB could not be reached or answered with an error; the answer must not be cached."""


class PosterService:
    def __init__(self, cache_dir: Path, api_key: str, client: httpx.Client | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=TIMEOUT)

    @classmethod
    def from_env(cls) -> "PosterService | None":
        key = os.environ.get("TMDB_API_KEY")
        if not key:
            return None
        return cls(Path(os.environ.get("CINEMATCH_POSTER_CACHE", "artifacts/posters")), key)

    # -- auth: TMDB issues both a short v3 key (query parameter) and a long v4 JWT (Bearer header).
    #    Prefer the header form whenever we have the token: query strings end up in access logs.
    def _auth(self) -> tuple[dict, dict]:
        if self.api_key.startswith("eyJ"):
            return {"Authorization": f"Bearer {self.api_key}"}, {}
        return {}, {"api_key": self.api_key}

    def _paths(self, tmdb_id: int, width: int) -> tuple[Path, Path]:
        return self.cache_dir / f"{tmdb_id}_w{width}.jpg", self.cache_dir / f"{tmdb_id}.none"

    def get(self, tmdb_id: int, width: int) -> bytes:
        image, none_marker = self._paths(tmdb_id, width)
        if image.exists():
            return image.read_bytes()
        if none_marker.exists():
            raise PosterUnavailable(tmdb_id)

        headers, params = self._auth()
        try:
            meta = self.client.get(f"{API}/movie/{tmdb_id}", headers=headers, params=params)
            if meta.status_code == 404:
                self._remember_none(none_marker)
                raise PosterUnavailable(tmdb_id)
            meta.raise_for_status()
            poster_path = meta.json().get("poster_path")
            if not poster_path:
                self._remember_none(none_marker)
                raise PosterUnavailable(tmdb_id)
            img = self.client.get(f"{IMAGE_BASE}/w{width}{poster_path}")
            img.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise UpstreamError(str(exc)) from exc

        self._atomic_write(image, img.content)
        return img.content

    def _remember_none(self, marker: Path) -> None:
        self._atomic_write(marker, b"")

    def _atomic_write(self, target: Path, data: bytes) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
