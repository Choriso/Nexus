from datetime import datetime

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

EMBEDDING_DIM = 384

revision = "add_dynamic_aliases"
down_revision = "add_node_embeddings"  # ← ЗАМЕНИТЕ на ваш актуальный head
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'dynamic_aliases',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('raw_tag', sa.String(500), nullable=False, unique=True, index=True),
        sa.Column('slug', sa.String(200), nullable=False, index=True),
        sa.Column('confidence', sa.Float(), default=0.9),
        sa.Column('enriched_context', sa.Text(), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('access_count', sa.Integer(), default=0),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow),
        sa.PrimaryKeyConstraint('id'),
    )