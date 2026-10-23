"""Password hashing and access tokens. Pure functions: no web framework, no database.

PASSWORDS: Argon2id via argon2-cffi, used directly. Argon2id is the current OWASP first choice; it
is memory-hard (64 MiB with the library's defaults, the RFC 9106 low-memory profile), which is what
makes GPU cracking expensive. The encoded hash carries its own salt and parameters, so tuning the
cost later does not invalidate old hashes: `needs_rehash` lets login upgrade them in place. bcrypt
would also have been defensible, but silently truncates at 72 bytes; Argon2 has no such limit.

TOKENS: HS256 JWT, access-only, 60 minutes. Decoding pins `algorithms=["HS256"]` (never trust the
token's own `alg` header: that is how `alg: none` and RS256-key-as-HMAC-secret forgeries work) and
requires `exp` and `sub`. OUT OF SCOPE, deliberately: refresh tokens, token revocation/logout, and
secret rotation. A stolen token is valid until it expires; rotating the secret logs everyone out.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_SECONDS = 60 * 60
CLOCK_SKEW_LEEWAY_SECONDS = 10
MIN_SECRET_LENGTH = 32

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """True iff `password` matches. A malformed stored hash is a failed login, not a crash."""
    try:
        return _hasher.verify(stored_hash, password)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    return _hasher.check_needs_rehash(stored_hash)


# Verified against when the account does not exist, so "unknown email" costs the same time as
# "wrong password" and response latency does not reveal which emails are registered.
_DUMMY_HASH: str | None = None


def dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("not-a-real-password")
    return _DUMMY_HASH


class InvalidToken(Exception):
    """Token is malformed, expired, mis-signed, or missing required claims."""


@dataclass(frozen=True)
class TokenClaims:
    user_id: int


def jwt_secret() -> str:
    """The signing key, from the environment. No default: a service that falls back to a baked-in
    secret accepts forged tokens from anyone who has read the source."""
    secret = os.environ.get("CINEMATCH_JWT_SECRET")
    if not secret:
        raise RuntimeError(
            "CINEMATCH_JWT_SECRET is not set. Generate one: "
            'python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    if len(secret) < MIN_SECRET_LENGTH:
        raise RuntimeError(f"CINEMATCH_JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters")
    return secret


def create_access_token(user_id: int, secret: str, now: int | None = None) -> str:
    issued = int(time.time()) if now is None else now
    payload = {"sub": str(user_id), "iat": issued, "exp": issued + ACCESS_TOKEN_TTL_SECONDS}
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> TokenClaims:
    try:
        payload = jwt.decode(
            token, secret, algorithms=[ALGORITHM], leeway=CLOCK_SKEW_LEEWAY_SECONDS,
            options={"require": ["exp", "sub"]},
        )
        return TokenClaims(user_id=int(payload["sub"]))
    except (jwt.PyJWTError, ValueError, TypeError) as exc:
        raise InvalidToken(str(exc)) from exc
