"""
Backward-compatible imports for AI models.

Core runtime AI tables now live in data.ai to keep the ORM model layer unified.
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text

from data.ai import AIExtractedInterests, UserCompatibility, UserPersonalityProfile
from data.session import SqlAlchemyBase


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
