"""
Alembic миграция для создания таблиц AI профайлера
Revision ID: add_ai_profiler_tables
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'add_ai_profiler_tables'
down_revision = None  # Замените на ID предыдущей миграции
branch_labels = None
depends_on = None


def upgrade():
    """Создание таблиц для AI профайлера"""
    
    # 1. Таблица профилей личности
    op.create_table(
        'ai_user_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        
        # Big Five scores
        sa.Column('openness', sa.Float(), nullable=True),
        sa.Column('conscientiousness', sa.Float(), nullable=True),
        sa.Column('extraversion', sa.Float(), nullable=True),
        sa.Column('agreeableness', sa.Float(), nullable=True),
        sa.Column('neuroticism', sa.Float(), nullable=True),
        
        # MBTI
        sa.Column('mbti_type', sa.String(length=4), nullable=True),
        
        # JSON поля
        sa.Column('communication_style', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Массивы
        sa.Column('traits', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('values', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('compatible_mbti_types', postgresql.ARRAY(sa.String()), nullable=True),
        
        # Текст
        sa.Column('collaboration_style', sa.Text(), nullable=True),
        
        # Метаданные
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('last_analyzed', sa.DateTime(), nullable=True),
        sa.Column('conversation_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id')
    )
    
    # Индексы для ai_user_profiles
    op.create_index('idx_ai_profiles_user_id', 'ai_user_profiles', ['user_id'])
    op.create_index('idx_ai_profiles_mbti', 'ai_user_profiles', ['mbti_type'])
    op.create_index('idx_traits_gin', 'ai_user_profiles', ['traits'], postgresql_using='gin')
    
    # 2. Таблица извлеченных интересов
    op.create_table(
        'ai_extracted_interests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        
        # Интересы (массивы)
        sa.Column('hobbies', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('topics', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('skills', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('dislikes', postgresql.ARRAY(sa.String()), nullable=True),
        
        # Работа
        sa.Column('occupation', sa.String(length=200), nullable=True),
        sa.Column('work_style', sa.Text(), nullable=True),
        
        # Цели
        sa.Column('short_term_goals', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('long_term_goals', postgresql.ARRAY(sa.String()), nullable=True),
        
        # Предпочтения
        sa.Column('preferences', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Метаданные
        sa.Column('extracted_from_messages', sa.Integer(), nullable=True),
        sa.Column('last_extraction', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id')
    )
    
    # Индексы для ai_extracted_interests
    op.create_index('idx_extracted_interests_user_id', 'ai_extracted_interests', ['user_id'])
    op.create_index('idx_hobbies_gin', 'ai_extracted_interests', ['hobbies'], postgresql_using='gin')
    
    # 3. Таблица кэша совместимости
    op.create_table(
        'user_compatibility_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id_1', sa.Integer(), nullable=False),
        sa.Column('user_id_2', sa.Integer(), nullable=False),
        
        # Оценки
        sa.Column('overall_score', sa.Float(), nullable=True),
        sa.Column('romantic_score', sa.Float(), nullable=True),
        sa.Column('professional_score', sa.Float(), nullable=True),
        sa.Column('creative_score', sa.Float(), nullable=True),
        sa.Column('interest_overlap', sa.Float(), nullable=True),
        
        # Рекомендации
        sa.Column('recommendations', sa.Text(), nullable=True),
        
        # Метаданные
        sa.Column('calculated_at', sa.DateTime(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id_1'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id_2'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id_1', 'user_id_2', name='unique_user_pair')
    )
    
    # Индексы для user_compatibility_cache
    op.create_index('idx_compatibility_users', 'user_compatibility_cache', ['user_id_1', 'user_id_2'])
    
    # 4. Таблица эмбеддингов
    op.create_table(
        'profile_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        
        # Эмбеддинг (JSON или vector если используете pgvector)
        sa.Column('embedding', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        
        # Текстовое представление
        sa.Column('profile_text', sa.Text(), nullable=True),
        
        # Метаданные
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id')
    )
    
    # Индексы для profile_embeddings
    op.create_index('idx_embeddings_user_id', 'profile_embeddings', ['user_id'])
    
    # Если используете pgvector extension:
    # op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    # op.execute('ALTER TABLE profile_embeddings ALTER COLUMN embedding TYPE vector(384)')
    # op.create_index('idx_embedding_vector', 'profile_embeddings', ['embedding'], 
    #                postgresql_using='ivfflat', postgresql_with={'lists': 100})


def downgrade():
    """Удаление таблиц AI профайлера"""
    
    op.drop_index('idx_embeddings_user_id', table_name='profile_embeddings')
    op.drop_table('profile_embeddings')
    
    op.drop_index('idx_compatibility_users', table_name='user_compatibility_cache')
    op.drop_table('user_compatibility_cache')
    
    op.drop_index('idx_hobbies_gin', table_name='ai_extracted_interests')
    op.drop_index('idx_extracted_interests_user_id', table_name='ai_extracted_interests')
    op.drop_table('ai_extracted_interests')
    
    op.drop_index('idx_traits_gin', table_name='ai_user_profiles')
    op.drop_index('idx_ai_profiles_mbti', table_name='ai_user_profiles')
    op.drop_index('idx_ai_profiles_user_id', table_name='ai_user_profiles')
    op.drop_table('ai_user_profiles')
    
    # Если использовали pgvector:
    # op.execute('DROP EXTENSION IF EXISTS vector')
