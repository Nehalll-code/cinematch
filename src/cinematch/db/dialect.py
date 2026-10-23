"""Dialect-specific INSERT construction (ON CONFLICT lives in different modules per dialect)."""
from __future__ import annotations

from sqlalchemy.orm import Session


def insert_for(session: Session, model):
    if session.get_bind().dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert(model)
