import logging
import os
from datetime import datetime, timezone

from celery import Celery, Task
from dotenv import load_dotenv
from sqlalchemy import and_, or_

from app.ai_profiler import get_profiler
from data.ai import AIExtractedInterests, UserCompatibility, UserPersonalityProfile
from data.message import Message
from data.session import create_session, global_init
from data.user import User

load_dotenv()

db_url = os.environ.get("DATABASE_URL", "sqlite:///chat.db")
global_init(db_url)

logger = logging.getLogger(__name__)

broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.environ.get("CELERY_RESULT_BACKEND", broker_url)
celery = Celery("ai_profiler", broker=broker_url, backend=result_backend)
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


class DBTask(Task):
    """
    Базовый Celery Task с автоматическим созданием и закрытием сессии базы данных.

    Атрибуты:
        _db (Session): Текущая сессия SQLAlchemy (private, для служебного использования)
    """

    _db = None

    def __call__(self, *args, **kwargs):
        """
        Автоматически создаёт сессию БД перед запуском задачи и освобождает её после выполнения.

        Args:
            *args: Позиционные аргументы задачи.
            **kwargs: Именованные аргументы задачи.

        Returns:
            object: Результат выполнения задачи.
        """
        self._db = create_session()
        try:
            return self.run(*args, **kwargs)
        finally:
            if self._db:
                self._db.close()

    @property
    def db(self):
        """
        Возвращает текущую сессию SQLAlchemy для взаимодействия с БД.

        Returns:
            Session: Сессия базы данных.
        """
        return self._db


@celery.task(base=DBTask, name="ai.analyze_user_profile", bind=True)
def analyze_user_profile(self, user_id: int, force: bool = True) -> dict:
    """
    Анализирует профиль пользователя: вычисляет OCEAN-профиль, MBTI, коммуникационный стиль и извлекает интересы.

    Args:
        self (DBTask): Инстанс задачи celery, с доступом к базе данных.
        user_id (int): Идентификатор пользователя.
        force (bool, optional): Форсировать запуск анализа даже если уже был ранее. По умолчанию True.

    Returns:
        dict: Статус выполнения анализа и ключевые результаты (или сообщение об ошибке).
    """
    logger.info("Starting user profile analysis for user_id=%s", user_id)
    db = self.db
    now_utc = datetime.now(timezone.utc)
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"error": "User not found"}

        profile = db.query(UserPersonalityProfile).filter_by(user_id=user_id).first()

        messages = (
            db.query(Message)
            .filter_by(author_id=user_id)
            .order_by(Message.timestamp.desc())
            .limit(50)
            .all()
        )
        full_text = " ".join([m.content for m in messages if m.content])
        if not full_text:
            return {"error": "No text data for analysis"}

        profiler = get_profiler()
        analysis = profiler.analyze_profile(full_text)
        ocean = analysis["ocean"]
        communication = analysis["communication"]
        extracted = analysis["interests"]

        if not profile:
            profile = UserPersonalityProfile(user_id=user_id)
            db.add(profile)

        # Обновление данных профиля по результатам анализа
        profile.openness = ocean[0]
        profile.conscientiousness = ocean[1]
        profile.extraversion = ocean[2]
        profile.agreeableness = ocean[3]
        profile.neuroticism = ocean[4]
        profile.mbti_type = analysis["mbti_type"]
        profile.communication_style = communication["communication_style"]
        profile.formality = communication["formality"]
        profile.enthusiasm = communication["enthusiasm"]
        profile.detail_oriented = communication["detail_oriented"]
        profile.traits = analysis["traits"]
        profile.values = analysis["values"]
        profile.compatible_mbti_types = analysis["compatible_mbti_types"]
        profile.collaboration_style = communication["collaboration_style"]
        profile.confidence_score = analysis["confidence_score"]
        profile.last_analyzed = now_utc
        profile.updated_at = now_utc
        profile.conversation_count = len(messages)

        extracted_interests = db.query(AIExtractedInterests).filter_by(user_id=user_id).first()
        if not extracted_interests:
            extracted_interests = AIExtractedInterests(user_id=user_id)
            db.add(extracted_interests)
        extracted_interests.hobbies = extracted.get("hobbies", [])
        extracted_interests.topics = extracted.get("topics", [])
        extracted_interests.skills = extracted.get("skills", [])
        extracted_interests.dislikes = extracted.get("dislikes", [])
        extracted_interests.occupation = extracted.get("occupation")
        extracted_interests.work_style = extracted.get("work_style")
        extracted_interests.short_term_goals = extracted.get("short_term_goals", [])
        extracted_interests.long_term_goals = extracted.get("long_term_goals", [])
        extracted_interests.preferences = extracted.get("preferences", {})
        extracted_interests.last_extraction = now_utc

        db.commit()
        update_compatibility.delay(user_id)
        return {
            "status": "success",
            "user_id": user_id,
            "scores": ocean,
            "mbti_type": profile.mbti_type,
        }
    except Exception as exc:
        logger.exception("Error while analyzing profile for user_id=%s", user_id)
        db.rollback()
        return {"error": str(exc)}


@celery.task(base=DBTask, name="ai.update_compatibility", bind=True)
def update_compatibility(self, user_id: int) -> dict:
    """
    Пересчитывает совместимость (compatibility) текущего пользователя с другими по типу личности и признакам OCEAN.

    Args:
        self (DBTask): Инстанс задачи celery, с доступом к базе данных.
        user_id (int): Идентификатор пользователя, для которого происходит расчет.

    Returns:
        dict: Статус выполнения операции или сообщение об ошибке.
    """
    db = self.db
    try:
        profiler = get_profiler()
        my = db.query(UserPersonalityProfile).filter_by(user_id=user_id).first()
        if not my:
            return {"error": "My profile not found"}

        my_vec = my.get_big_five_vector()
        others = db.query(UserPersonalityProfile).filter(UserPersonalityProfile.user_id != user_id).all()
        for other in others:
            other_vec = other.get_big_five_vector()
            score = profiler.calculate_compatibility(my_vec, other_vec)

            compat = (
                db.query(UserCompatibility)
                .filter(
                    or_(
                        and_(
                            UserCompatibility.user_id_1 == user_id,
                            UserCompatibility.user_id_2 == other.user_id,
                        ),
                        and_(
                            UserCompatibility.user_id_1 == other.user_id,
                            UserCompatibility.user_id_2 == user_id,
                        ),
                    )
                )
                .first()
            )
            if not compat:
                compat = UserCompatibility(user_id_1=user_id, user_id_2=other.user_id)
                db.add(compat)

            compat.overall_score = round(score / 100.0, 4)
            compat.romantic_score = compat.overall_score
            compat.professional_score = compat.overall_score
            compat.creative_score = compat.overall_score
            compat.interest_overlap = compat.overall_score
            compat.recommendations = {
                "summary": f"Compatibility between {user_id} and {other.user_id}",
                "mbti_pair": [my.mbti_type, other.mbti_type],
            }
            compat.calculated_at = datetime.now(timezone.utc)

        db.commit()
        return {"status": "success", "updated": len(others)}
    except Exception as exc:
        logger.exception("Error in compatibility update for user_id=%s", user_id)
        db.rollback()
        return {"error": str(exc)}
