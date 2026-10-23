from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from cinematch.db.indexing import ITEM_IDX, USER_IDX, peek
from cinematch.db.models import Interaction, Movie, WatchlistItem

SECRET = "s" * 40


@pytest.fixture
def client(db, migrated_engine, tmp_path, monkeypatch):
    db.add_all([Movie(movie_id=10, item_idx=0, title="Alpha (2000)", genres="Drama"),
                Movie(movie_id=20, item_idx=1, title="Beta (2001)", genres="Comedy|Drama")])
    db.commit()
    monkeypatch.setenv("DATABASE_URL", migrated_engine.url.render_as_string(hide_password=False))
    monkeypatch.setenv("CINEMATCH_MODEL", str(tmp_path / "no-model.pkl"))
    monkeypatch.setenv("CINEMATCH_JWT_SECRET", SECRET)
    from cinematch.api import app

    with TestClient(app) as c:
        yield c


def _auth(client, email):
    password = "correct horse battery"
    client.post("/auth/register", json={"email": email, "password": password})
    token = client.post("/auth/login", json={"email": email, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def ada(client):
    return _auth(client, "ada@example.com")


@pytest.fixture
def bob(client):
    return _auth(client, "bob@example.com")


class TestInteractions:
    def test_requires_authentication(self, client):
        assert client.post("/interactions", json={"movie_id": 10, "type": "watch"}).status_code == 401
        assert client.get("/interactions").status_code == 401

    def test_rating_is_stored_for_the_token_holder(self, client, db, ada):
        res = client.post("/interactions", headers=ada, json={"movie_id": 10, "type": "rating", "rating": 4.5})
        assert res.status_code == 201
        row = db.scalars(select(Interaction)).one()
        assert (row.movie_id, row.type, row.rating) == (10, "rating", 4.5)
        created = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=timezone.utc)
        assert abs((datetime.now(timezone.utc) - created).total_seconds()) < 30

    def test_re_rating_updates_in_place_and_advances_the_timestamp(self, client, db, ada):
        first = client.post("/interactions", headers=ada, json={"movie_id": 10, "type": "rating", "rating": 2.0})
        again = client.post("/interactions", headers=ada, json={"movie_id": 10, "type": "rating", "rating": 5.0})
        assert (first.status_code, again.status_code) == (201, 200)
        rows = db.scalars(select(Interaction)).all()
        assert len(rows) == 1 and rows[0].rating == 5.0

    def test_watch_has_no_rating(self, client, ada):
        assert client.post("/interactions", headers=ada, json={"movie_id": 10, "type": "watch"}).status_code == 201

    @pytest.mark.parametrize("body", [
        {"movie_id": 10, "type": "rating"},                       # rating required
        {"movie_id": 10, "type": "watch", "rating": 3.0},         # rating forbidden
        {"movie_id": 10, "type": "rating", "rating": 0.0},        # out of range
        {"movie_id": 10, "type": "rating", "rating": 5.5},
        {"movie_id": 10, "type": "rating", "rating": 3.3},        # not a half-star step
        {"movie_id": 10, "type": "like"},                         # unknown type
        {"movie_id": 10, "type": "watch", "user_id": 999},        # identity comes from the token only
    ])
    def test_invalid_bodies_are_422_not_500(self, client, ada, body):
        assert client.post("/interactions", headers=ada, json=body).status_code == 422

    def test_unknown_movie_is_404(self, client, ada):
        assert client.post("/interactions", headers=ada, json={"movie_id": 999, "type": "watch"}).status_code == 404

    def test_listing_is_scoped_to_the_caller_newest_first(self, client, ada, bob):
        client.post("/interactions", headers=ada, json={"movie_id": 10, "type": "watch"})
        client.post("/interactions", headers=ada, json={"movie_id": 20, "type": "rating", "rating": 3.0})
        client.post("/interactions", headers=bob, json={"movie_id": 10, "type": "rating", "rating": 1.0})
        mine = client.get("/interactions", headers=ada).json()
        assert [i["movie_id"] for i in mine] == [20, 10]
        assert [i["movie_id"] for i in client.get("/interactions", headers=bob).json()] == [10]

    def test_writes_do_not_touch_the_index_spaces(self, client, db, ada):
        before = (peek(db, USER_IDX), peek(db, ITEM_IDX))
        client.post("/interactions", headers=ada, json={"movie_id": 10, "type": "watch"})
        db.expire_all()
        assert (peek(db, USER_IDX), peek(db, ITEM_IDX)) == before


class TestWatchlist:
    def test_requires_authentication(self, client):
        assert client.get("/watchlist").status_code == 401
        assert client.put("/watchlist/10").status_code == 401
        assert client.delete("/watchlist/10").status_code == 401

    def test_add_is_idempotent_and_listed_with_titles(self, client, db, ada):
        assert client.put("/watchlist/10", headers=ada).status_code == 204
        assert client.put("/watchlist/10", headers=ada).status_code == 204
        assert db.scalar(select(func.count()).select_from(WatchlistItem)) == 1
        listing = client.get("/watchlist", headers=ada).json()
        assert [(m["movie_id"], m["title"]) for m in listing] == [(10, "Alpha (2000)")]

    def test_unknown_movie_is_404(self, client, ada):
        assert client.put("/watchlist/999", headers=ada).status_code == 404

    def test_remove_is_idempotent(self, client, ada):
        client.put("/watchlist/10", headers=ada)
        assert client.delete("/watchlist/10", headers=ada).status_code == 204
        assert client.delete("/watchlist/10", headers=ada).status_code == 204
        assert client.get("/watchlist", headers=ada).json() == []

    def test_users_cannot_see_or_remove_each_others_entries(self, client, ada, bob):
        client.put("/watchlist/10", headers=ada)
        assert client.get("/watchlist", headers=bob).json() == []
        client.delete("/watchlist/10", headers=bob)
        assert [m["movie_id"] for m in client.get("/watchlist", headers=ada).json()] == [10]
