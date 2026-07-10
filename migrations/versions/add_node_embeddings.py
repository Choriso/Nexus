
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

EMBEDDING_DIM = 384

revision = "add_node_embeddings"
down_revision = "b7e4c2a91d03"  # ← ЗАМЕНИТЕ на ваш актуальный head
branch_labels = None
depends_on = None

def upgrade():
    op.add_column(
        "interest_hierarchy_nodes",
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_interest_hierarchy_nodes_embedding_hnsw
        ON interest_hierarchy_nodes
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_interest_hierarchy_nodes_embedding_hnsw")
    op.drop_column("interest_hierarchy_nodes", "embedding")