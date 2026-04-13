from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class AiChatLog(Base):
    """
    Базовая схема для логирования обращений к AI.
    Это пример схемы поверх существующей БД (Prisma-аналог на SQLAlchemy).
    """

    __tablename__ = "ai_chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True)
    provider = Column(String(32), nullable=False)  # local / external / mock
    prompt = Column(Text, nullable=False)
    reply = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

