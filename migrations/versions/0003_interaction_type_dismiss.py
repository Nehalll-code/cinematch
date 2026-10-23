"""interactions.type admits 'dismiss' ("not interested": seen, hidden from recommendations, not a positive)

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-23
"""
from alembic import op

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

# A CHECK constraint (not a native Postgres ENUM) is why this is an ordinary drop/create:
# ALTER TYPE .. ADD VALUE cannot run inside a transaction block on older Postgres and cannot be undone.
NAME = op.f('ck_interactions_type_known')  # op.f: already fully named, skip the naming convention


def upgrade() -> None:
    with op.batch_alter_table('interactions') as batch_op:
        batch_op.drop_constraint(NAME, type_='check')
        batch_op.create_check_constraint(NAME, "type IN ('rating', 'watch', 'dismiss')")


def downgrade() -> None:
    # destructive by necessity: rows of the removed type cannot satisfy the old constraint
    op.execute("DELETE FROM interactions WHERE type = 'dismiss'")
    with op.batch_alter_table('interactions') as batch_op:
        batch_op.drop_constraint(NAME, type_='check')
        batch_op.create_check_constraint(NAME, "type IN ('rating', 'watch')")
