"""Allocation from the append-only index spaces (see models.IndexCounter)."""
from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .models import IndexCounter

USER_IDX = "user_idx"
ITEM_IDX = "item_idx"


def allocate(session: Session, name: str, count: int = 1) -> int:
    """Reserve `count` consecutive indices and return the first.

    Runs inside the caller's transaction: the UPDATE row-locks the counter until commit, so
    concurrent allocators queue instead of colliding, and a rollback returns the indices.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    new_next = session.execute(
        update(IndexCounter)
        .where(IndexCounter.name == name)
        .values(next_value=IndexCounter.next_value + count)
        .returning(IndexCounter.next_value)
    ).scalar_one_or_none()
    if new_next is None:
        raise LookupError(f"index counter {name!r} is not seeded; run `alembic upgrade head`")
    return new_next - count


def peek(session: Session, name: str) -> int:
    return session.execute(select(IndexCounter.next_value).where(IndexCounter.name == name)).scalar_one()
