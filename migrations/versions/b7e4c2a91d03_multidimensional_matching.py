"""multidimensional matching schema

Revision ID: b7e4c2a91d03
Revises: 0acf81f89c90
Create Date: 2026-07-05 17:20:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "b7e4c2a91d03"
down_revision = "0acf81f89c90"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "interest_hierarchy_nodes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("slug", sa.String(length=200), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=True),
        sa.Column("match_weight", sa.Float(), nullable=True),
        sa.Column("global_category", sa.String(length=50), nullable=True),
        sa.ForeignKeyConstraint(["parent_id"], ["interest_hierarchy_nodes.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_index("ix_interest_hierarchy_nodes_slug", "interest_hierarchy_nodes", ["slug"])

    op.create_table(
        "user_interest_graph_weights",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.Integer(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("source_tag", sa.String(length=200), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["node_id"], ["interest_hierarchy_nodes.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "node_id", name="uq_user_interest_node"),
    )
    op.create_index("ix_user_interest_graph_weights_user_id", "user_interest_graph_weights", ["user_id"])

    op.create_table(
        "user_behavior_profiles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("avg_char_count", sa.Float(), nullable=True),
        sa.Column("avg_reply_time", sa.Float(), nullable=True),
        sa.Column("avg_emoji_count", sa.Float(), nullable=True),
        sa.Column("avg_hour", sa.Float(), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )

    op.create_table(
        "user_schwartz_profiles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("self_direction", sa.Float(), nullable=True),
        sa.Column("stimulation", sa.Float(), nullable=True),
        sa.Column("hedonism", sa.Float(), nullable=True),
        sa.Column("achievement", sa.Float(), nullable=True),
        sa.Column("power", sa.Float(), nullable=True),
        sa.Column("security", sa.Float(), nullable=True),
        sa.Column("conformity", sa.Float(), nullable=True),
        sa.Column("tradition", sa.Float(), nullable=True),
        sa.Column("benevolence", sa.Float(), nullable=True),
        sa.Column("universalism", sa.Float(), nullable=True),
        sa.Column("values_json", sa.JSON(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )

    op.add_column("messages", sa.Column("char_count", sa.Integer(), nullable=True))
    op.add_column("messages", sa.Column("reply_time", sa.Float(), nullable=True))
    op.add_column("messages", sa.Column("emoji_count", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("messages", "emoji_count")
    op.drop_column("messages", "reply_time")
    op.drop_column("messages", "char_count")
    op.drop_table("user_schwartz_profiles")
    op.drop_table("user_behavior_profiles")
    op.drop_index("ix_user_interest_graph_weights_user_id", "user_interest_graph_weights")
    op.drop_table("user_interest_graph_weights")
    op.drop_index("ix_interest_hierarchy_nodes_slug", "interest_hierarchy_nodes")
    op.drop_table("interest_hierarchy_nodes")
