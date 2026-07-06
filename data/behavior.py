"""Поведенческий профиль коммуникации пользователя."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer

from .session import SqlAlchemyBase


class UserBehaviorProfile(SqlAlchemyBase):
    """Агрегированные метрики стиля общения по логам сообщений."""

    __tablename__ = "user_behavior_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    avg_char_count = Column(Float, default=0.0)
    avg_reply_time = Column(Float, default=0.0)
    avg_emoji_count = Column(Float, default=0.0)
    avg_hour = Column(Float, default=12.0)
    message_count = Column(Integer, default=0)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
