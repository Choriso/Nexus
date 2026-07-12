"""add metric weight offset columns to users

Revision ID: add_user_metric_offsets
Revises: add_dynamic_aliases
Create Date: 2026-07-12 20:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "add_user_metric_offsets"
down_revision = "add_dynamic_aliases"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("metric_weight_ocean_offset", sa.Float(), server_default="0.0", nullable=False))
    op.add_column("users", sa.Column("metric_weight_graph_offset", sa.Float(), server_default="0.0", nullable=False))
    op.add_column("users", sa.Column("metric_weight_jaccard_offset", sa.Float(), server_default="0.0", nullable=False))


def downgrade():
    op.drop_column("users", "metric_weight_jaccard_offset")
    op.drop_column("users", "metric_weight_graph_offset")
    op.drop_column("users", "metric_weight_ocean_offset")
