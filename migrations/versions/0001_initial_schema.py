"""initial schema: users, movies, interactions, watchlist, index_counters

Revision ID: 0001
Revises: 
Create Date: 2026-10-23 19:06:21.577572
"""
from alembic import op
import sqlalchemy as sa


revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('index_counters',
    sa.Column('name', sa.String(length=32), nullable=False),
    sa.Column('next_value', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('name', name=op.f('pk_index_counters'))
    )
    op.create_table('movies',
    sa.Column('movie_id', sa.Integer(), autoincrement=False, nullable=False),
    sa.Column('item_idx', sa.Integer(), nullable=False),
    sa.Column('title', sa.Text(), nullable=False),
    sa.Column('genres', sa.Text(), server_default='', nullable=False),
    sa.PrimaryKeyConstraint('movie_id', name=op.f('pk_movies')),
    sa.UniqueConstraint('item_idx', name=op.f('uq_movies_item_idx'))
    )
    op.create_table('users',
    sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
    sa.Column('user_idx', sa.Integer(), nullable=False),
    sa.Column('ml_user_id', sa.Integer(), nullable=True),
    sa.Column('email', sa.String(length=320), nullable=True),
    sa.Column('password_hash', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.CheckConstraint('email IS NOT NULL OR ml_user_id IS NOT NULL', name=op.f('ck_users_has_identity')),
    sa.CheckConstraint('email IS NULL OR email = lower(email)', name=op.f('ck_users_email_lowercase')),
    sa.CheckConstraint('password_hash IS NULL OR email IS NOT NULL', name=op.f('ck_users_password_needs_email')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('email', name=op.f('uq_users_email')),
    sa.UniqueConstraint('ml_user_id', name=op.f('uq_users_ml_user_id')),
    sa.UniqueConstraint('user_idx', name=op.f('uq_users_user_idx'))
    )
    op.create_table('interactions',
    sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), nullable=False),
    sa.Column('movie_id', sa.Integer(), nullable=False),
    sa.Column('type', sa.String(length=16), nullable=False),
    sa.Column('rating', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.CheckConstraint("(type = 'rating' AND rating IS NOT NULL AND rating >= 0.5 AND rating <= 5.0) OR (type <> 'rating' AND rating IS NULL)", name=op.f('ck_interactions_rating_matches_type')),
    sa.CheckConstraint("type IN ('rating', 'watch')", name=op.f('ck_interactions_type_known')),
    sa.ForeignKeyConstraint(['movie_id'], ['movies.movie_id'], name=op.f('fk_interactions_movie_id_movies'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_interactions_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_interactions')),
    sa.UniqueConstraint('user_id', 'movie_id', 'type', name='uq_interactions_user_movie_type')
    )
    with op.batch_alter_table('interactions', schema=None) as batch_op:
        batch_op.create_index('ix_interactions_movie_id', ['movie_id'], unique=False)

    op.create_table('watchlist',
    sa.Column('user_id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), nullable=False),
    sa.Column('movie_id', sa.Integer(), nullable=False),
    sa.Column('added_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    sa.ForeignKeyConstraint(['movie_id'], ['movies.movie_id'], name=op.f('fk_watchlist_movie_id_movies'), ondelete='RESTRICT'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_watchlist_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id', 'movie_id', name=op.f('pk_watchlist'))
    )

    # The allocator (cinematch.db.indexing.allocate) raises if a counter row is missing, so the
    # two append-only index spaces start at 0 here rather than being created lazily.
    counters = sa.table('index_counters', sa.column('name', sa.String), sa.column('next_value', sa.Integer))
    op.bulk_insert(counters, [{'name': 'user_idx', 'next_value': 0}, {'name': 'item_idx', 'next_value': 0}])


def downgrade() -> None:
    op.drop_table('watchlist')
    with op.batch_alter_table('interactions', schema=None) as batch_op:
        batch_op.drop_index('ix_interactions_movie_id')

    op.drop_table('interactions')
    op.drop_table('users')
    op.drop_table('movies')
    op.drop_table('index_counters')
