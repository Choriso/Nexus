import logging
from datetime import datetime, timezone

from celery import Celery, Task
from sqlalchemy import and_, or_
from sqlalchemy import text
from app.ai_profiler import get_profiler
from app.ai_profiler.behavior_analyzer import refresh_user_behavior_profile
from app.ai_profiler.interest_graph import register_user_tags, refresh_all_node_embeddings
from app.ai_profiler.schwartz_analyzer import (
    analyze_schwartz_values,
    extract_onboarding_text,
    upsert_schwartz_profile,
)
from config import config
from data.ai import AIExtractedInterests, UserCompatibility, UserPersonalityProfile, UserSchwartzProfile
from data.message import Message
from data.session import create_session, global_init
from data.user import User

global_init(config.DATABASE_URL)

logger = logging.getLogger(__name__)

celery = Celery(
    "ai_profiler",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
)
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "adjust-global-weights-every-24h": {
            "task": "ai.adjust_global_weights",
            "schedule": config.WEIGHT_ADJUSTMENT_INTERVAL,
            "args": (),
        },
    },
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
            .limit(config.MAX_MESSAGES_PER_ANALYSIS)
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
        profile.embedding = ocean
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

        tag_list: list[str] = []
        for field in ("hobbies", "skills", "topics"):
            for item in extracted.get(field) or []:
                if isinstance(item, dict) and item.get("subcategory"):
                    tag_list.append(str(item["subcategory"]))
                elif isinstance(item, str):
                    tag_list.append(item)
        
        # WRITE PHASE: Resolve all tags through dynamic enrichment
        if tag_list:
            from app.ai_profiler.dynamic_enrichment import get_tag_enricher
            enricher = get_tag_enricher()
            
            resolved_slugs = []
            for raw_tag in tag_list:
                slug = enricher.resolve_tag_to_slug(db, raw_tag, fallback_to_enrichment=True)
                if slug:
                    resolved_slugs.append(slug)
                    logger.debug(f"[analyze_profile] Resolved '{raw_tag}' -> '{slug}'")
                else:
                    logger.warning(f"[analyze_profile] Could not resolve '{raw_tag}'")
            
            # Register only successfully resolved slugs
            if resolved_slugs:
                register_user_tags(db, user_id, resolved_slugs)

        refresh_user_behavior_profile(db, user_id)

        db.commit()
        update_compatibility.delay(user_id)
        analyze_schwartz_values_task.delay(user_id)
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


@celery.task(base=DBTask, name="ai.analyze_schwartz_values", bind=True)
def analyze_schwartz_values_task(self, user_id: int) -> dict:
    """
    Фоновая задача: извлекает 10 ценностей Schwartz через Ollama (phi3:medium).
    При ошибке или таймауте — безопасный no-op, HTTP-поток не затрагивается.
    """
    db = self.db
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"error": "User not found"}

        messages = (
            db.query(Message)
            .filter_by(author_id=user_id)
            .order_by(Message.timestamp.desc())
            .limit(config.MAX_MESSAGES_PER_ANALYSIS)
            .all()
        )
        messages_text = " ".join(m.content for m in messages if m.content)
        onboarding_text = extract_onboarding_text(user, messages_text)

        if not onboarding_text.strip():
            return {"status": "skipped", "reason": "no text"}

        values = analyze_schwartz_values(onboarding_text)
        if not values:
            return {"status": "fallback", "reason": "ollama unavailable or parse failed"}

        upsert_schwartz_profile(db, user_id, values)
        return {"status": "success", "user_id": user_id}
    except Exception as exc:
        logger.exception("Schwartz analysis failed for user_id=%s", user_id)
        db.rollback()
        return {"error": str(exc)}


@celery.task(base=DBTask, name="ai.adjust_global_weights", bind=True)
def adjust_global_weights(self) -> dict:
    """
    Медленный глобальный контур (Block 2): собирает персональные смещения
    весов метрик активных пользователей, усредняет тренд и плавно корректирует
    глобальные базовые константы в GlobalWeightsConfig.
    """
    from app.ai_profiler.search_ranking import aggregate_global_trend

    db = self.db
    try:
        result = aggregate_global_trend(db)
        logger.info("[adjust_global_weights] Result: %s", result)
        return result
    except Exception as exc:
        logger.exception("[adjust_global_weights] Error: %s", exc)
        db.rollback()
        return {"error": str(exc)}


@celery.task(base=DBTask, name="ai.refresh_interest_embeddings", bind=True)
def refresh_interest_embeddings(self) -> dict:
    """
    Полный пересчёт SBERT-эмбеддингов всех узлов графа интересов
    (InterestHierarchyNode.embedding, pgvector). Нужен после смены SBERT-модели
    или массового обновления _HIERARCHY_SEED / SEMANTIC_ONTOLOGY — то есть
    редкая административная операция, а не часть пользовательского пайплайна.
    Резолвинг тегов пользователей (register_user_tags / resolve_tags_batch)
    продолжает работать во время пересчёта — просто до его завершения
    новые/изменённые узлы временно матчатся хуже.
    """
    db = self.db
    try:
        updated = refresh_all_node_embeddings(db)
        logger.info("Refreshed embeddings for %s interest graph nodes", updated)
        return {"status": "success", "nodes_reset": updated}
    except Exception as exc:
        logger.exception("Error refreshing interest graph embeddings")
        db.rollback()
        return {"error": str(exc)}


@celery.task(base=DBTask, name="ai.update_compatibility", bind=True)
def update_compatibility(self, user_id: int) -> dict:
    """
    Пересчитывает совместимость текущего пользователя с другими, используя
    встроенный косинусный поиск pgvector на уровне СУБД.
    """
    db = self.db
    try:
        my = db.query(UserPersonalityProfile).filter_by(user_id=user_id).first()
        if not my or my.embedding is None:
            return {"error": "My profile or embedding not found"}

        # 1. Используем pgvector для расчета расстояний прямо в запросе.
        # Метод cosine_distance возвращает расстояние (0 - идентичны, 2 - противоположны).
        # Совместимость = 1 - (расстояние / 2).
        # То есть если расстояние 0, совместимость 1 (100%).

        # Получаем всех остальных пользователей и сразу вычисляем косинусное расстояние
        similar_profiles = db.query(
            UserPersonalityProfile.user_id,
            UserPersonalityProfile.mbti_type,
            UserPersonalityProfile.embedding.cosine_distance(my.embedding).label("distance")
        ).filter(
            UserPersonalityProfile.user_id != user_id,
            UserPersonalityProfile.embedding.isnot(None)  # Игнорируем тех, у кого еще нет вектора
        ).all()

        if not similar_profiles:
            return {"status": "success", "updated": 0}

        # 2. Пакетная загрузка существующих связей (оставляем вашу логику маппинга)
        existing_compats = db.query(UserCompatibility).filter(
            or_(
                UserCompatibility.user_id_1 == user_id,
                UserCompatibility.user_id_2 == user_id,
            )
        ).all()

        compat_map = {}
        for c in existing_compats:
            target_id = c.user_id_2 if c.user_id_1 == user_id else c.user_id_1
            compat_map[target_id] = c

        # 3. Обновление объектов SQLAlchemy
        new_compats = []
        for other in similar_profiles:
            # Расстояние от 0 до 2. Переводим в процент сходства [0.0, 1.0]
            # Чем меньше расстояние, тем больше сходство.
            score_normalized = round(1.0 - (float(other.distance) / 2.0), 4)
            # Защита от отрицательных значений на всякий случай
            score_normalized = max(0.0, score_normalized)

            compat = compat_map.get(other.user_id)

            if not compat:
                compat = UserCompatibility(user_id_1=user_id, user_id_2=other.user_id)
                new_compats.append(compat)

            compat.overall_score = score_normalized
            compat.romantic_score = score_normalized
            compat.professional_score = score_normalized
            compat.creative_score = score_normalized
            compat.interest_overlap = score_normalized
            compat.recommendations = {
                "summary": f"Compatibility calculated via pgvector cosine distance",
                "mbti_pair": [my.mbti_type, other.mbti_type],
            }
            compat.calculated_at = datetime.now(timezone.utc)

        if new_compats:
            db.add_all(new_compats)

        db.commit()
        return {"status": "success", "updated": len(similar_profiles)}
    except Exception as exc:
        logger.exception("Error in compatibility update for user_id=%s", user_id)
        db.rollback()
        return {"error": str(exc)}