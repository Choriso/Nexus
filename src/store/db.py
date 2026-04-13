import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/your_db")

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def get_session() -> Generator[Session, None, None]:
    """
    Базовая точка входа в БД для нового кода.

    Пример использования (в сервисах / фоновых задачах):

        from src.store.db import get_session

        with contextlib.closing(next(get_session())) as db:
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

