"""
AI Database Models for Connectify
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime, JSON, Boolean
from sqlalchemy.orm import relationship
from data.session import SqlAlchemyBase


class UserPersonalityProfile(SqlAlchemyBase):
    """
    Профиль личности пользователя на основе Big Five (OCEAN)
    """
    __tablename__ = 'ai_user_personality_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, unique=True)

    # Big Five черты характера (0-1)
    openness = Column(Float, default=0.5)  # Открытость опыту
    conscientiousness = Column(Float, default=0.5)  # Добросовестность
    extraversion = Column(Float, default=0.5)  # Экстраверсия
    agreeableness = Column(Float, default=0.5)  # Доброжелательность
    neuroticism = Column(Float, default=0.5)  # Нейротизм
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # MBTI тип
    mbti_type = Column(String(4), nullable=True)

    # Стиль коммуникации
    communication_style = Column(String(50), nullable=True)  # formal, casual, professional, friendly
    formality = Column(Float, default=0.5)
    enthusiasm = Column(Float, default=0.5)
    detail_oriented = Column(Float, default=0.5)

    # Дополнительные черты (JSON)
    traits = Column(JSON, nullable=True)  # ['analytical', 'creative', 'leadership']
    values = Column(JSON, nullable=True)  # ['family', 'career', 'freedom']

    # Совместимость с MBTI типами
    compatible_mbti_types = Column(JSON, nullable=True)

    # Стиль сотрудничества
    collaboration_style = Column(String(50), nullable=True)

    # Уверенность в анализе
    confidence_score = Column(Float, default=0.0)

    # Метаданные
    last_analyzed = Column(DateTime, default=datetime.utcnow)
    conversation_count = Column(Integer, default=0)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'big_five': {
                'openness': self.openness,
                'conscientiousness': self.conscientiousness,
                'extraversion': self.extraversion,
                'agreeableness': self.agreeableness,
                'neuroticism': self.neuroticism
            },
            'mbti_type': self.mbti_type,
            'communication_style': self.communication_style,
            'traits': self.traits,
            'values': self.values,
            'compatible_mbti_types': self.compatible_mbti_types,
            'collaboration_style': self.collaboration_style,
            'confidence_score': self.confidence_score,
            'last_analyzed': self.last_analyzed.isoformat() if self.last_analyzed else None,
            'conversation_count': self.conversation_count
        }

    def get_big_five_vector(self):
        return [
            self.openness or 0.5,
            self.conscientiousness or 0.5,
            self.extraversion or 0.5,
            self.agreeableness or 0.5,
            self.neuroticism or 0.5
        ]


class AIExtractedInterests(SqlAlchemyBase):
    """
    Извлеченные интересы из диалогов
    """
    __tablename__ = 'ai_extracted_interests'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, unique=True)

    # Интересы
    hobbies = Column(JSON, nullable=True, default=list)
    topics = Column(JSON, nullable=True, default=list)
    skills = Column(JSON, nullable=True, default=list)
    dislikes = Column(JSON, nullable=True, default=list)

    # Работа и карьера
    occupation = Column(String(200), nullable=True)
    work_style = Column(Text, nullable=True)

    # Цели
    short_term_goals = Column(JSON, nullable=True, default=list)
    long_term_goals = Column(JSON, nullable=True, default=list)

    # Предпочтения
    preferences = Column(JSON, nullable=True)

    # Метаданные
    last_extraction = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'hobbies': self.hobbies,
            'topics': self.topics,
            'skills': self.skills,
            'dislikes': self.dislikes,
            'occupation': self.occupation,
            'work_style': self.work_style,
            'short_term_goals': self.short_term_goals,
            'long_term_goals': self.long_term_goals,
            'preferences': self.preferences,
            'last_extraction': self.last_extraction.isoformat() if self.last_extraction else None
        }


class UserCompatibility(SqlAlchemyBase):
    """
    Совместимость между пользователями
    """
    __tablename__ = 'ai_user_compatibility'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_1 = Column(Integer, ForeignKey('users.id'), nullable=False)
    user_id_2 = Column(Integer, ForeignKey('users.id'), nullable=False)

    # Оценки совместимости (0-1)
    overall_score = Column(Float, default=0.0)
    romantic_score = Column(Float, default=0.0)
    professional_score = Column(Float, default=0.0)
    creative_score = Column(Float, default=0.0)
    interest_overlap = Column(Float, default=0.0)

    # Рекомендации
    recommendations = Column(JSON, nullable=True)

    # Метаданные
    calculated_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id_1': self.user_id_1,
            'user_id_2': self.user_id_2,
            'scores': {
                'overall': self.overall_score,
                'romantic': self.romantic_score,
                'professional': self.professional_score,
                'creative': self.creative_score,
                'interests': self.interest_overlap
            },
            'recommendations': self.recommendations,
            'calculated_at': self.calculated_at.isoformat() if self.calculated_at else None
        }


class ConversationAnalysis(SqlAlchemyBase):
    """
    Анализ разговоров для обучения модели
    """
    __tablename__ = 'ai_conversation_analysis'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    # Текст профиля
    profile_text = Column(Text, nullable=True)

    # Метаданные
    updated_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'profile_text': self.profile_text,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class TrainingMetrics(SqlAlchemyBase):
    """
    Метрики обучения модели
    """
    __tablename__ = 'ai_training_metrics'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)

    # Метрики
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)

    # Дополнительные метрики
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)

    # Дата обучения
    trained_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'model_version': self.model_version,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None
        }
