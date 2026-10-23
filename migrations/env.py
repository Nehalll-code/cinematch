"""Alembic environment. The URL is taken from, in order: a programmatic override
(config.attributes["url"], used by tests), then DATABASE_URL."""
from alembic import context

from cinematch.db.models import Base
from cinematch.db.session import database_url, make_engine

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return config.attributes.get("url") or database_url()


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = make_engine(_url())
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
