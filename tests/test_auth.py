import time

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from cinematch import auth
from cinematch.db.indexing import USER_IDX, peek
from cinematch.db.models import User

SECRET = "s" * 40


# ---- password hashing ------------------------------------------------------------------------

def test_hash_is_argon2id_salted_and_verifies():
    h1, h2 = auth.hash_password("correct horse"), auth.hash_password("correct horse")
    assert h1.startswith("$argon2id$") and h1 != h2  # per-hash random salt
    assert auth.verify_password("correct horse", h1)
    assert not auth.verify_password("correct horsf", h1)


@pytest.mark.parametrize("garbage", ["", "plaintext", "$argon2id$broken", "$2b$12$notargon"])
def test_verify_treats_a_malformed_stored_hash_as_a_failed_login_not_a_crash(garbage):
    assert auth.verify_password("whatever", garbage) is False


# ---- tokens ----------------------------------------------------------------------------------

def test_token_round_trip_and_lifetime_is_sixty_minutes():
    now = int(time.time())
    token = auth.create_access_token(42, SECRET, now=now)
    claims = auth.decode_access_token(token, SECRET)
    assert claims.user_id == 42
    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert payload["exp"] - payload["iat"] == 60 * 60 and payload["iat"] == now


def test_expired_token_is_rejected():
    token = auth.create_access_token(1, SECRET, now=int(time.time()) - 2 * 3600)
    with pytest.raises(auth.InvalidToken):
        auth.decode_access_token(token, SECRET)


def test_tampered_or_foreign_tokens_are_rejected():
    good = auth.create_access_token(1, SECRET)
    with pytest.raises(auth.InvalidToken):
        auth.decode_access_token(good, "t" * 40)  # signed with a different key
    with pytest.raises(auth.InvalidToken):
        auth.decode_access_token(good[:-2] + "xx", SECRET)  # signature altered
    unsigned = jwt.encode({"sub": "1", "exp": int(time.time()) + 600}, key=None, algorithm="none")
    with pytest.raises(auth.InvalidToken):
        auth.decode_access_token(unsigned, SECRET)  # alg=none
    hs512 = jwt.encode({"sub": "1", "exp": int(time.time()) + 600}, SECRET, algorithm="HS512")
    with pytest.raises(auth.InvalidToken):
        auth.decode_access_token(hs512, SECRET)  # only HS256 is accepted


def test_token_without_exp_or_sub_is_rejected():
    for claims in ({"sub": "1"}, {"exp": int(time.time()) + 600}):
        with pytest.raises(auth.InvalidToken):
            auth.decode_access_token(jwt.encode(claims, SECRET, algorithm="HS256"), SECRET)


def test_non_numeric_subject_is_rejected():
    bad = jwt.encode({"sub": "abc", "exp": int(time.time()) + 600}, SECRET, algorithm="HS256")
    with pytest.raises(auth.InvalidToken):
        auth.decode_access_token(bad, SECRET)


def test_secret_has_no_default_and_must_be_long(monkeypatch):
    monkeypatch.delenv("CINEMATCH_JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="CINEMATCH_JWT_SECRET"):
        auth.jwt_secret()
    monkeypatch.setenv("CINEMATCH_JWT_SECRET", "short")
    with pytest.raises(RuntimeError, match="at least 32"):
        auth.jwt_secret()
    monkeypatch.setenv("CINEMATCH_JWT_SECRET", SECRET)
    assert auth.jwt_secret() == SECRET


# ---- HTTP endpoints --------------------------------------------------------------------------

@pytest.fixture
def client(db, migrated_engine, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", migrated_engine.url.render_as_string(hide_password=False))
    monkeypatch.setenv("CINEMATCH_MODEL", str(tmp_path / "no-model.pkl"))  # accounts need no model
    monkeypatch.setenv("CINEMATCH_JWT_SECRET", SECRET)
    from cinematch.api import app

    with TestClient(app) as c:
        yield c


def _register(client, email="Ada@Example.com", password="correct horse battery"):
    return client.post("/auth/register", json={"email": email, "password": password})


def _login(client, email="ada@example.com", password="correct horse battery"):
    return client.post("/auth/login", json={"email": email, "password": password})


def test_register_stores_a_lowercased_email_and_an_argon2_hash_never_the_password(client, db):
    res = _register(client)
    assert res.status_code == 201 and res.json()["email"] == "ada@example.com"
    row = db.scalars(select(User)).one()
    assert row.email == "ada@example.com" and row.ml_user_id is None
    assert row.password_hash.startswith("$argon2id$") and "correct horse" not in row.password_hash


def test_duplicate_email_is_a_conflict_regardless_of_case(client):
    assert _register(client).status_code == 201
    assert _register(client, email="ADA@example.COM").status_code == 409


@pytest.mark.parametrize("payload", [
    {"email": "not-an-email", "password": "correct horse battery"},
    {"email": "ada@example.com", "password": "short"},
    {"email": "ada@example.com", "password": "x" * 129},
    {"email": "ada@example.com"},
])
def test_invalid_registrations_are_422(client, payload):
    assert client.post("/auth/register", json=payload).status_code == 422


def test_new_accounts_get_the_next_append_only_user_idx(client, db):
    assert _register(client, email="a@example.com").status_code == 201
    assert _register(client, email="b@example.com").status_code == 201
    idx = sorted(db.scalars(select(User.user_idx)).all())
    assert idx == [0, 1] and peek(db, USER_IDX) == 2


def test_failed_registration_does_not_burn_an_index(client, db):
    assert _register(client, email="a@example.com").status_code == 201
    assert _register(client, email="a@example.com").status_code == 409
    assert _register(client, email="b@example.com").status_code == 201
    assert sorted(db.scalars(select(User.user_idx)).all()) == [0, 1]  # no hole


def test_login_returns_a_working_bearer_token(client):
    _register(client)
    res = _login(client)
    assert res.status_code == 200
    body = res.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 3600
    me = client.get("/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == "ada@example.com"


def test_login_failures_are_indistinguishable(client, db):
    _register(client)
    db.add(User(user_idx=99, ml_user_id=7))  # imported MovieLens user: no email, no password
    db.commit()
    wrong_pw = _login(client, password="wrong password!!")
    unknown = _login(client, email="nobody@example.com")
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json() == unknown.json()  # same body: no account enumeration


@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Bearer garbage"}, {"Authorization": "Basic abc"},
])
def test_me_requires_a_valid_token(client, headers):
    res = client.get("/me", headers=headers)
    assert res.status_code == 401 and res.headers["www-authenticate"] == "Bearer"


def test_expired_token_is_rejected_by_the_endpoint(client):
    _register(client)
    uid = client.get("/me", headers={"Authorization": "Bearer " + _login(client).json()["access_token"]}).json()["id"]
    stale = auth.create_access_token(uid, SECRET, now=int(time.time()) - 7200)
    assert client.get("/me", headers={"Authorization": f"Bearer {stale}"}).status_code == 401


def test_token_for_a_user_that_no_longer_exists_is_rejected(client):
    ghost = auth.create_access_token(123456, SECRET)
    assert client.get("/me", headers={"Authorization": f"Bearer {ghost}"}).status_code == 401


def test_account_endpoints_are_503_when_no_database_is_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("CINEMATCH_MODEL", str(tmp_path / "no-model.pkl"))
    from cinematch.api import app

    with TestClient(app) as c:
        assert c.post("/auth/login", json={"email": "a@example.com", "password": "x" * 10}).status_code == 503


def test_service_refuses_to_start_with_a_database_but_no_jwt_secret(migrated_engine, tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", migrated_engine.url.render_as_string(hide_password=False))
    monkeypatch.setenv("CINEMATCH_MODEL", str(tmp_path / "no-model.pkl"))
    monkeypatch.delenv("CINEMATCH_JWT_SECRET", raising=False)
    from cinematch.api import app

    with pytest.raises(RuntimeError, match="CINEMATCH_JWT_SECRET"):
        with TestClient(app):
            pass
