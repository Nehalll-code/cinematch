"""Account endpoints: register, login, current user. Requires a configured database."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import auth
from .db.indexing import USER_IDX, allocate
from .db.models import User
from .state import STATE

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)

# Bounds hashing cost: Argon2 time/memory does not grow with length, but unbounded request bodies
# are still an easy way to waste a worker. 8 is the NIST SP 800-63B minimum; no composition rules.
MIN_PASSWORD, MAX_PASSWORD = 8, 128


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD, max_length=MAX_PASSWORD)


class LoginRequest(BaseModel):
    email: EmailStr
    # no min_length here: a too-short password is a failed login (401), not a validation hint
    password: str = Field(max_length=MAX_PASSWORD)


class Account(BaseModel):
    id: int
    email: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def get_session():
    factory = STATE.get("session_factory")
    if factory is None:
        raise HTTPException(503, "database not configured; set DATABASE_URL")
    with factory() as session:
        yield session


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: Session = Depends(get_session),
) -> User:
    if creds is None:
        raise _unauthorized("not authenticated")
    try:
        claims = auth.decode_access_token(creds.credentials, STATE["jwt_secret"])
    except auth.InvalidToken:
        raise _unauthorized("invalid or expired token") from None
    user = session.get(User, claims.user_id)
    if user is None:
        raise _unauthorized("invalid or expired token")
    return user


@router.post("/auth/register", status_code=201, response_model=Account)
def register(body: RegisterRequest, session: Session = Depends(get_session)) -> Account:
    email = body.email.strip().lower()
    # hash BEFORE touching the index counter: allocate() row-locks it until commit, and hashing
    # takes tens of milliseconds that every concurrent registration would otherwise queue behind
    password_hash = auth.hash_password(body.password)
    try:
        user = User(user_idx=allocate(session, USER_IDX), email=email, password_hash=password_hash)
        session.add(user)
        session.commit()
    except IntegrityError:
        # the unique(email) constraint is the arbiter, so two simultaneous registrations of the
        # same address cannot both succeed; the rollback also returns the allocated user_idx
        session.rollback()
        raise HTTPException(409, "email already registered") from None
    return Account(id=user.id, email=user.email)


@router.post("/auth/login", response_model=Token)
def login(body: LoginRequest, session: Session = Depends(get_session)) -> Token:
    user = session.scalar(select(User).where(User.email == body.email.strip().lower()))
    stored = user.password_hash if user is not None and user.password_hash else None
    # always run one verification, against a dummy hash when there is no usable account, so the
    # three failure cases (unknown email, imported user without a password, wrong password) take
    # the same time and return the same response
    ok = auth.verify_password(body.password, stored or auth.dummy_hash())
    if user is None or stored is None or not ok:
        raise _unauthorized("invalid credentials")
    if auth.needs_rehash(stored):
        user.password_hash = auth.hash_password(body.password)
        session.commit()
    return Token(
        access_token=auth.create_access_token(user.id, STATE["jwt_secret"]),
        expires_in=auth.ACCESS_TOKEN_TTL_SECONDS,
    )


@router.get("/me", response_model=Account)
def me(user: User = Depends(current_user)) -> Account:
    return Account(id=user.id, email=user.email or "")
