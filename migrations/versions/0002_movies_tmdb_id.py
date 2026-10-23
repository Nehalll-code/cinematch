"""movies.tmdb_id

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-23 22:51:49.820087
"""
from alembic import op
import sqlalchemy as sa


revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('movies', schema=None) as batch_op:
        batch_op.add_column(sa.Column('tmdb_id', sa.Integer(), nullable=True))



def downgrade() -> None:
    with op.batch_alter_table('movies', schema=None) as batch_op:
        batch_op.drop_column('tmdb_id')

