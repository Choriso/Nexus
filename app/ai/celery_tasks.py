"""
Celery задачи для фонового анализа профилей
"""

from celery import Celery, Task
from app import create_app, db
from app.models import User, Message 
from app.ai.models import UserPersonalityProfile, UserCompatibility, AIExtractedInterests 
from app.ai.personality_analyzer import PersonalityAnalyzer
from app.ai.data_processor import TextDataProcessor
from datetime import datetime, timedelta
import logging
import click # Для CLI команд
import json # Для вывода метрик тренировки

logger = logging.getLogger(__name__)

# Создаем Celery app
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
    """
    Фоновый анализ профиля пользователя
    
    Args:
        user_id: ID пользователя
        force: Принудительный анализ даже если был недавно
    
    Returns:
        dict: Результат анализа
    """
    logger.info(f"Starting profile analysis for user {user_id}")
    
    try:
        user = User.query.get(user_id)
        if not user:
            logger.error(f"User {user_id} not found")
            return {'error': 'User not found'}
        
        # Проверяем последний анализ
        if not force:
            profile = UserPersonalityProfile.query.filter_by(user_id=user_id).first()
            if profile and profile.last_analyzed:
                time_since = datetime.utcnow() - profile.last_analyzed
                if time_since < timedelta(hours=1):
                    logger.info(f"Profile {user_id} was analyzed recently, skipping")
                    return {'status': 'skipped', 'reason': 'recent_analysis'}
        
        processor = TextDataProcessor()
        analyzer = PersonalityAnalyzer(processor)

        messages = Message.query.filter_by(user_id=user_id).order_by(Message.created_at.desc()).limit(50).all()
        user_text = " ".join([msg.text for msg in messages if msg.text])

        if not user_text:
            logger.info(f"User {user_id} has no messages to analyze. Skipping analysis.")
            return {'status': 'skipped', 'reason': 'no_messages'}

        
        personality_profile = analyzer.predict_personality(user_text)

        if personality_profile is None:
            logger.error(f"Personality analysis returned None for user {user_id}. Model might not be trained.")
            return {'error': 'Personality analysis failed or model not trained'}

        existing_profile = UserPersonalityProfile.query.filter_by(user_id=user_id).first()
        if existing_profile:
            existing_profile.openness = personality_profile.openness
            existing_profile.conscientiousness = personality_profile.conscientiousness
            existing_profile.extraversion = personality_profile.extraversion
            existing_profile.agreeableness = personality_profile.agreeableness
            existing_profile.neuroticism = personality_profile.neuroticism
            existing_profile.mbti_type = personality_profile.mbti_type
            existing_profile.confidence_score = personality_profile.confidence_score
            existing_profile.last_analyzed = datetime.utcnow()
            existing_profile.conversation_count = (existing_profile.conversation_count or 0) + len(messages)
            db.session.add(existing_profile)
            profile_id = existing_profile.id
        else:
            personality_profile.user_id = user_id
            personality_profile.conversation_count = len(messages)
            db.session.add(personality_profile)
            db.session.flush()
            profile_id = personality_profile.id
        
        db.session.commit()

        logger.info(f"Successfully analyzed and saved profile for user {user_id}. Profile ID: {profile_id}")
        
        # Триггер для пересчета совместимости
        update_compatibility_matrix.delay(user_id)

        return {
            'status': 'success',
            'user_id': user_id,
            'profile_id': profile_id,
            'confidence': personality_profile.confidence_score
        }
        
    except Exception as e:
        logger.exception(f"Error analyzing user {user_id}")
        return {'error': str(e)}


@celery.task(base=FlaskTask, name='ai.update_compatibility_matrix')
def update_compatibility_matrix(user_id):
    """
    Обновляет совместимость между пользователем и всеми остальными
    Запускается после обновления профиля
    
    Args:
        user_id: ID пользователя
    """
    logger.info(f"Updating compatibility matrix for user {user_id}")
    
    try:
        processor = TextDataProcessor()
        analyzer = PersonalityAnalyzer(processor)

        profile = UserPersonalityProfile.query.filter_by(user_id=user_id).first()
        
        if not profile:
            logger.error(f"Personality profile not found for user {user_id}")
            return {'error': 'Profile not found'}
        
        all_profiles = UserPersonalityProfile.query.filter(
            UserPersonalityProfile.user_id != user_id
        ).all()
        
        updated_count = 0
        
        for other_profile in all_profiles:
            compatibility_score = analyzer.calculate_compatibility(profile, other_profile)
            
            existing_compat = UserCompatibility.query.filter(\
                ((UserCompatibility.user_id_1 == user_id) & (UserCompatibility.user_id_2 == other_profile.user_id)) |\
                ((UserCompatibility.user_id_1 == other_profile.user_id) & (UserCompatibility.user_id_2 == user_id))\
            ).first()

            if existing_compat:
                existing_compat.overall_score = compatibility_score
                existing_compat.romantic_score = compatibility_score 
                existing_compat.professional_score = compatibility_score
                existing_compat.creative_score = compatibility_score
                existing_compat.calculated_at = datetime.utcnow()
                db.session.add(existing_compat)
            else:
                new_compat = UserCompatibility(
                    user_id_1=user_id,
                    user_id_2=other_profile.user_id,
                    overall_score=compatibility_score,
                    romantic_score=compatibility_score,
                    professional_score=compatibility_score,
                    creative_score=compatibility_score
                )
                db.session.add(new_compat)
            db.session.commit()
            updated_count += 1
        
        logger.info(f"Updated {updated_count} compatibility records for user {user_id}")
        
        return {
            'status': 'success',
            'user_id': user_id,
            'updated_count': updated_count
        }
        
    except Exception as e:
        logger.exception(f"Error updating compatibility matrix for user {user_id}")
        return {'error': str(e)}


@celery.task(base=FlaskTask, name='ai.batch_analyze_users')
def batch_analyze_users(user_ids_to_analyze): # Изменено имя аргумента для ясности
    """
    Пакетный анализ нескольких пользователей
    Полезно для первоначального анализа или периодического обновления
    
    Args:
        user_ids_to_analyze: Список ID пользователей
    """
    logger.info(f"Batch analyzing {len(user_ids_to_analyze)} users")
    
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    for user_id in user_ids_to_analyze:
        result = analyze_user_profile(user_id, force=False)
        
        if 'error' in result:
            results['failed'].append({'user_id': user_id, 'error': result['error']})
        elif result.get('status') == 'skipped':
            results['skipped'].append(user_id)
        else:
            results['success'].append(user_id)
    
    logger.info(f"Batch analysis complete: {len(results['success'])} success, "
                f"{len(results['failed'])} failed, {len(results['skipped'])} skipped")
    
    return results


@celery.task(base=FlaskTask, name='ai.nightly_analysis')
def nightly_analysis():
    """
    Ночной анализ активных пользователей
    Запускается по расписанию (например, в 3:00 каждую ночь)
    """
    logger.info("Starting nightly analysis")
    
    try:
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        active_users = db.session.query(Message.user_id)\
            .filter(Message.created_at >= week_ago)\
            .distinct()\
            .all()
        
        user_ids_for_analysis = [user_id for (user_id,) in active_users]
        
        logger.info(f"Found {len(user_ids_for_analysis)} active users for analysis")
        
        result = batch_analyze_users(user_ids_for_analysis)
        
        return {
            'status': 'complete',
            'analyzed': len(user_ids_for_analysis),
            'results': result
        }
        
    except Exception as e:
        logger.exception("Error in nightly analysis")
        return {'error': str(e)}


# ==================== Периодические задачи ====================

# Настройка Celery Beat для периодических задач
celery.conf.beat_schedule = {
    'nightly-analysis': {
        'task': 'ai.nightly_analysis',
        'schedule': 3600 * 24,  # Каждые 24 часа
        'options': {'expires': 3600 * 2}  # Истекает через 2 часа
    },
}


# ==================== Хуки для автоматического анализа ====================

def trigger_analysis_on_message_count(user_id, message_count):
    """
    Триггер для автоматического анализа каждые N сообщений
    Вызывайте эту функцию при создании нового сообщения
    
    Args:
        user_id: ID пользователя
        message_count: Текущее количество сообщений
    """
    # Анализировать каждые 10 сообщений
    if message_count % 10 == 0:
        logger.info(f"Triggering analysis for user {user_id} (message count: {message_count})")
        analyze_user_profile.delay(user_id, force=False)


# ==================== CLI команды для управления ====================

def setup_cli_commands(app):
    """
    Добавляет CLI команды для управления AI системой
    Вызовите в вашем create_app()
    """
    
    @app.cli.command('ai-analyze-all')
    def analyze_all():
        """Анализировать всех пользователей"""
        with app.app_context():
            all_users = User.query.all()
            user_ids = [user.id for user in all_users]
            
            click.echo(f"Starting analysis of {len(user_ids)} users...")
            result = batch_analyze_users(user_ids)
            
            click.echo(f"Success: {len(result['success'])} ")
            click.echo(f"Failed: {len(result['failed'])} ")
            click.echo(f"Skipped: {len(result['skipped'])} ")
    
    @app.cli.command('ai-analyze-user')
    @click.argument('user_id', type=int)
    def analyze_user(user_id):
        """
        Анализировать конкретного пользователя
        """
        with app.app_context():
            click.echo(f"Analyzing user {user_id}...")
            result = analyze_user_profile(user_id, force=True)
            
            if 'error' in result:
                click.echo(f"Error: {result['error']}", err=True)
            else:
                click.echo(f"Success! Confidence: {result.get('confidence', 0):.2f}")
    
    # Команда ai-update-embeddings удалена, так как этот функционал изменен
    
    @app.cli.command('ai-train-personality-model')
    @click.option('--dataset_path', default='data/ai_personality_dataset.json', help='Path to the dataset JSON file.')
    @click.option('--model_version', default='1.0', help='Version of the model being trained.')
    def train_personality_model(dataset_path, model_version):
        """
        Обучает модель анализа личности на основе предоставленного датасета.
        Пример датасета:
        [
            {"text": "Какой-то текст пользователя", "label": "INTJ"},
            {"text": "Другой текст пользователя", "label": "ESFP"}
        ]
        """
        with app.app_context():
            click.echo(f"Starting personality model training with dataset: {dataset_path}")
            
            try:
                import json
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)

                if not dataset:
                    click.echo("Dataset is empty. Skipping training.")
                    return

                texts = [item['text'] for item in dataset]
                labels = [item['label'] for item in dataset]

                processor = TextDataProcessor()
                # Сначала обучаем векторизатор
                processor.fit_vectorizer(texts) 

                analyzer = PersonalityAnalyzer(processor)
                metrics = analyzer.train(texts, labels, model_version=model_version)

                if metrics:
                    click.echo("Training complete. Metrics:")
                    click.echo(json.dumps(metrics.to_dict(), indent=2))
                    # Сохранение метрик в БД (реализовано в PersonalityAnalyzer.train)
                else:
                    click.echo("Training failed or returned no metrics.")

            except FileNotFoundError:
                click.echo(f"Error: Dataset file not found at {dataset_path}", err=True)
            except json.JSONDecodeError:
                click.echo(f"Error: Invalid JSON in dataset file {dataset_path}", err=True)
            except Exception as e:
                click.echo(f"An unexpected error occurred during training: {e}", err=True)
                logger.exception("Error during personality model training CLI command")
