"""add pgvector

Revision ID: 0acf81f89c90
Revises: 
Create Date: 2026-06-23 22:10:27.657221

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector  # Импортируем тип векторов
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0acf81f89c90'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # 1. Включаем расширение pgvector в БД
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Добавляем колонку embedding (размерность 5 для OCEAN)
    op.add_column('ai_user_personality_profiles',
        sa.Column('embedding', Vector(5), nullable=True)
    )

    # 3. Создаем HNSW индекс
    op.create_index(
        'idx_user_personality_embedding_hnsw',
        'ai_user_personality_profiles',
        ['embedding'],
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'}
    )

def downgrade():
    op.drop_index('idx_user_personality_embedding_hnsw', table_name='ai_user_personality_profiles')
    op.drop_column('ai_user_personality_profiles', 'embedding')
