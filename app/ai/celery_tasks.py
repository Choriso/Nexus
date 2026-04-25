"""
Celery задачи для фонового анализа профилей (Оптимизировано с Singleton)
"""

from celery import Celery, Task
from app import create_app, db
from app.models import User, Message
from app.ai.models import UserPersonalityProfile, UserCompatibility
from datetime import datetime, timedelta
import logging
import click
import json
from app.ai.profiler_singleton import get_profiler

logger = logging.getLogger(__name__)

celery = Celery('ai_profiler')

# Конфигурация Celery
celery.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

class FlaskTask(Task):
    """Базовый класс для задач с Flask app context"""
    def __call__(self, *args, **kwargs):
        with create_app().app_context():
            return self.run(*args, **kwargs)

@celery.task(base=FlaskTask, name='ai.analyze_user_profile')
def analyze_user_profile(user_id, force=False):
    """Фоновый анализ профиля пользователя через Singleton"""
    # ПОЛУЧАЕМ ПРОФАЙЛЕР ИЗ СИНГЛТОНА (Модель грузится 1 раз на воркер)
    profiler = get_profiler()
    logger.info(f"Starting profile analysis for user {user_id}")

    try:
        user = User.query.get(user_id)
        if not user:
            return {'error': 'User not found'}

        # Проверка на частоту анализа
        if not force:
            profile = UserPersonalityProfile.query.filter_by(user_id=user_id).first()
            if profile and profile.last_analyzed:
                if datetime.utcnow() - profile.last_analyzed < timedelta(hours=1):
                    return {'status': 'skipped', 'reason': 'recent_analysis'}

        # Собираем сообщения
        messages = Message.query.filter_by(user_id=user_id).order_by(Message.created_at.desc()).limit(50).all()
        user_text = " ".join([msg.text for msg in messages if msg.text])

        if not user_text:
            return {'status': 'skipped', 'reason': 'no_messages'}

        # ГЛАВНОЕ ИЗМЕНЕНИЕ: Используем метод из core.py
        # analyze_profile возвращает словарь с 'ocean', 'mbti' и 'confidence'
        result = profiler.analyze_profile(user_text)

        existing_profile = UserPersonalityProfile.query.filter_by(user_id=user_id).first()

        if existing_profile:
            # Обновляем OCEAN
            existing_profile.openness = result['ocean']['openness']
            existing_profile.conscientiousness = result['ocean']['conscientiousness']
            existing_profile.extraversion = result['ocean']['extraversion']
            existing_profile.agreeableness = result['ocean']['agreeableness']
            existing_profile.neuroticism = result['ocean']['neuroticism']
            # Обновляем MBTI
            existing_profile.mbti_type = result['mbti']
            existing_profile.confidence_score = result['confidence']
            existing_profile.last_analyzed = datetime.utcnow()
            existing_profile.conversation_count = (existing_profile.conversation_count or 0) + len(messages)
            db.session.add(existing_profile)
        else:
            # Создаем новый профиль
            new_profile = UserPersonalityProfile(
                user_id=user_id,
                openness=result['ocean']['openness'],
                conscientiousness=result['ocean']['conscientiousness'],
                extraversion=result['ocean']['extraversion'],
                agreeableness=result['ocean']['agreeableness'],
                neuroticism=result['ocean']['neuroticism'],
                mbti_type=result['mbti'],
                confidence_score=result['confidence'],
                last_analyzed=datetime.utcnow(),
                conversation_count=len(messages)
            )
            db.session.add(new_profile)

        db.session.commit()
        update_compatibility_matrix.delay(user_id)

        return {'status': 'success', 'mbti': result['mbti']}

    except Exception as e:
        logger.exception(f"Error analyzing user {user_id}")
        return {'error': str(e)}

@celery.task(base=FlaskTask, name='ai.update_compatibility_matrix')
def update_compatibility_matrix(user_id):
    """Обновляет матрицу совместимости через Singleton"""
    profiler = get_profiler()
    logger.info(f"Updating compatibility matrix for user {user_id}")

    try:
        profile = UserPersonalityProfile.query.filter_by(user_id=user_id).first()
        if not profile:
            return {'error': 'Profile not found'}

        other_profiles = UserPersonalityProfile.query.filter(UserPersonalityProfile.user_id != user_id).all()

        for other in other_profiles:
            # Используем расчет из profiler
            score = profiler.calculate_compatibility(profile, other)

            existing = UserCompatibility.query.filter(
                ((UserCompatibility.user_id_1 == user_id) & (UserCompatibility.user_id_2 == other.user_id)) |
                ((UserCompatibility.user_id_1 == other.user_id) & (UserCompatibility.user_id_2 == user_id))
            ).first()

            if existing:
                existing.overall_score = score
                existing.calculated_at = datetime.utcnow()
            else:
                new_c = UserCompatibility(
                    user_id_1=user_id,
                    user_id_2=other.user_id,
                    overall_score=score,
                    romantic_score=score, # Можно добавить разную логику позже
                    professional_score=score,
                    creative_score=score
                )
                db.session.add(new_c)

        db.session.commit()
        return {'status': 'success', 'updated_for': user_id}

    except Exception as e:
        logger.exception(f"Error in compatibility matrix for {user_id}")
        return {'error': str(e)}

# Остальные задачи (batch_analyze_users, nightly_analysis) остаются без изменений,
# так как они внутри вызывают analyze_user_profile, который уже исправлен.