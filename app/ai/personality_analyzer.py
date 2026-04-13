import logging
from datetime import datetime, timedelta, timezone

# 1. Сначала внешние библиотеки
try:
    from celery import Celery, Task
except ImportError:
    print("Ошибка: Библиотека 'celery' не найдена. Установите: pip install celery")
    raise

# 2. Твои локальные модули (исправлено под твою структуру)
from data.user import User
from data.message import Message
from app.ai.models import UserPersonalityProfile
from app.ai_profiler.core import AIProfiler
import os
from dotenv import load_dotenv # Убедись, что установлен: pip install python-dotenv
from data.session import create_session, global_init
import random


# 1. Загружаем переменные из .env
load_dotenv()

# 2. Инициализируем БД, используя ту же логику, что и в App.py
db_url = os.environ.get("DATABASE_URL")
if db_url:
    global_init(db_url)
else:
    # Fallback, если что-то пошло не так
    global_init("sqlite:///db/blogs.db")

logger = logging.getLogger(__name__)

# 3. Инициализация Celery
# Теперь берем URL из переменных окружения
broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery = Celery('ai_profiler', broker=broker_url, backend=broker_url)

# ... остальной код (DBTask, analyze_user_profile) ...

logger = logging.getLogger(__name__)

# 3. Инициализация Celery
# Название 'ai_worker' — просто имя процесса
celery = Celery('ai_profiler')


# 4. Класс для работы с БД (чтобы сессии всегда закрывались)
class DBTask(Task):
    """Базовый класс для задач, который управляет сессией SQLAlchemy"""
    _db = None

    def __call__(self, *args, **kwargs):
        self._db = create_session()
        try:
            return self.run(*args, **kwargs)
        finally:
            if self._db:
                self._db.close()

    @property
    def db(self):
        return self._db


# 5. Главная задача анализа
@celery.task(base=DBTask, name='ai.analyze_user_profile', bind=True)
def analyze_user_profile(self, user_id, force=False):
    """Фоновый анализ профиля через твое Core-ядро"""
    logger.info(f"Начинаем анализ пользователя {user_id}")
    db = self.db  # Берем сессию из DBTask

    try:
        # Ищем пользователя
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {'error': 'User not found'}

        # Проверка даты последнего анализа
        profile = db.query(UserPersonalityProfile).filter_by(user_id=user_id).first()
        if profile and profile.updated_at and not force:
            # Убеждаемся, что время в одном формате (с таймзоной)
            last_upd = profile.updated_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last_upd < timedelta(hours=1):
                pass
                # logger.info(f"Профиль {user_id} обновлялся недавно. Пропуск.")
                # return {'status': 'skipped'}

        # Собираем данные: сообщения пользователя
        messages = db.query(Message) \
            .filter_by(author_id=user_id) \
            .order_by(Message.timestamp.desc()) \
            .limit(50) \
            .all()
        # Даем последним сообщениям больше веса (просто дублируем их в строке)
        last_messages = messages[:5]  # берем 5 последних
        # Внутри analyze_user_profile
        full_text = " ".join([m.content for m in messages if m.content])
        # Используй метод очистки, который мы обсуждали


        if not full_text:
            return {'error': 'No text data for analysis'}

        # Запуск нейронки из Core
        profiler = AIProfiler()
        logger.info(f"Анализирую {len(messages)} сообщений для User {user_id}. Текст: {full_text[50:]}...")
        clean_full_text = profiler.clean_text(full_text)
        scores = profiler.analyze_text(clean_full_text)

        # Сохранение OCEAN
        if not profile:
            profile = UserPersonalityProfile(user_id=user_id)
            db.add(profile)

        profile.openness = scores[0]
        profile.conscientiousness = scores[1]
        profile.extraversion = scores[2]
        profile.agreeableness = scores[3]
        profile.neuroticism = scores[4]
        profile.updated_at = datetime.now(timezone.utc)

        db.commit()
        return {'status': 'success', 'user_id': user_id, 'scores': scores}

    except Exception as e:
        logger.exception(f"Ошибка в анализе пользователя {user_id}: {e}")
        db.rollback()
        return {'error': str(e)}


# 6. Задача на совместимость
@celery.task(base=DBTask, name='ai.update_compatibility', bind=True)
def update_compatibility(self, user_id):
    """Обновляет % совпадения с другими людьми"""
    db = self.db
    try:
        profiler = AIProfiler()
        my = db.query(UserPersonalityProfile).filter_by(user_id=user_id).first()
        if not my: return {'error': 'My profile not found'}

        my_vec = [my.openness, my.conscientiousness, my.extraversion, my.agreeableness, my.neuroticism]

        others = db.query(UserPersonalityProfile).filter(UserPersonalityProfile.user_id != user_id).all()

        for other in others:
            other_vec = [other.openness, other.conscientiousness, other.extraversion, other.agreeableness,
                         other.neuroticism]
            score = profiler.calculate_compatibility(my_vec, other_vec)
            logger.info(f"Match {user_id} + {other.user_id} = {score}%")

        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Error in compatibility: {e}")
        return {'error': str(e)}



