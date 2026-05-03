"""
Конфигурация для AI Profiler
Добавьте эти настройки в ваш config.py
"""

import os
from datetime import timedelta

class Config:
    """Базовая конфигурация"""
    
    # ==================== Существующие настройки ====================
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///ai_profiler.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # ==================== AI Profiler настройки ====================
    
    # Anthropic API
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    
    # Режим работы: True = локальные модели, False = API
    USE_LOCAL_AI_MODELS = os.environ.get('USE_LOCAL_AI_MODELS', 'False').lower() == 'true'
    
    # Модель для эмбеддингов (Sentence Transformers)
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
    
    # Путь к локальным моделям (если используются)
    LOCAL_MODELS_PATH = os.environ.get('LOCAL_MODELS_PATH', './models')
    
    # ==================== Celery настройки ====================
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # ==================== Redis для кэширования ====================
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/1')
    CACHE_TYPE = 'RedisCache'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 3600  # 1 час
    
    # ==================== Анализ профилей ====================
    
    # Минимальное количество сообщений для анализа
    MIN_MESSAGES_FOR_ANALYSIS = int(os.environ.get('MIN_MESSAGES_FOR_ANALYSIS', '5'))
    
    # Интервал между анализами (в секундах)
    ANALYSIS_COOLDOWN = int(os.environ.get('ANALYSIS_COOLDOWN', '3600'))  # 1 час
    
    # Триггер анализа: каждые N сообщений
    ANALYSIS_MESSAGE_TRIGGER = int(os.environ.get('ANALYSIS_MESSAGE_TRIGGER', '10'))
    
    # Максимальное количество сообщений для анализа за раз
    MAX_MESSAGES_PER_ANALYSIS = int(os.environ.get('MAX_MESSAGES_PER_ANALYSIS', '50'))
    
    # ==================== Совместимость ====================
    
    # Время жизни кэша совместимости (в секундах)
    COMPATIBILITY_CACHE_TTL = int(os.environ.get('COMPATIBILITY_CACHE_TTL', '86400'))  # 24 часа
    
    # Минимальный порог совместимости для рекомендаций
    MIN_COMPATIBILITY_SCORE = float(os.environ.get('MIN_COMPATIBILITY_SCORE', '0.6'))
    
    # ==================== Поиск ====================
    
    # Порог для семантического поиска
    SEMANTIC_SEARCH_THRESHOLD = float(os.environ.get('SEMANTIC_SEARCH_THRESHOLD', '0.5'))
    
    # Максимальное количество результатов поиска
    MAX_SEARCH_RESULTS = int(os.environ.get('MAX_SEARCH_RESULTS', '20'))
    
    # ==================== Приватность и безопасность ====================
    
    # Требовать согласие пользователя на анализ
    REQUIRE_ANALYSIS_CONSENT = os.environ.get('REQUIRE_ANALYSIS_CONSENT', 'True').lower() == 'true'
    
    # Анонимизировать данные при обучении моделей
    ANONYMIZE_TRAINING_DATA = os.environ.get('ANONYMIZE_TRAINING_DATA', 'True').lower() == 'true'
    
    # Максимальное время хранения истории чата для анализа (дней)
    MAX_CHAT_HISTORY_DAYS = int(os.environ.get('MAX_CHAT_HISTORY_DAYS', '90'))
    
    # ==================== Rate Limiting ====================
    
    # API rate limits
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_DEFAULT = "100/hour"
    RATELIMIT_STRATEGY = "fixed-window"
    
    # Специфичные лимиты для AI endpoints
    AI_ANALYSIS_RATE_LIMIT = "5/hour"  # Анализ профиля
    AI_COMPATIBILITY_RATE_LIMIT = "20/hour"  # Расчет совместимости
    AI_SEARCH_RATE_LIMIT = "30/hour"  # Семантический поиск
    
    # ==================== Мониторинг ====================
    
    # Логирование AI операций
    LOG_AI_OPERATIONS = os.environ.get('LOG_AI_OPERATIONS', 'True').lower() == 'true'
    
    # Уровень логирования
    AI_LOG_LEVEL = os.environ.get('AI_LOG_LEVEL', 'INFO')


class DevelopmentConfig(Config):
    """Конфигурация для разработки"""
    DEBUG = True
    TESTING = False
    
    # В разработке можно использовать меньше сообщений для анализа
    MIN_MESSAGES_FOR_ANALYSIS = 3
    ANALYSIS_MESSAGE_TRIGGER = 5
    
    # Более частые анализы
    ANALYSIS_COOLDOWN = 300  # 5 минут
    
    # Отключить rate limiting
    RATELIMIT_ENABLED = False


class ProductionConfig(Config):
    """Конфигурация для production"""
    DEBUG = False
    TESTING = False
    
    # В production более строгие настройки
    MIN_MESSAGES_FOR_ANALYSIS = 10
    ANALYSIS_MESSAGE_TRIGGER = 15
    ANALYSIS_COOLDOWN = 7200  # 2 часа
    
    # Включить rate limiting
    RATELIMIT_ENABLED = True
    
    # Требовать HTTPS
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class TestingConfig(Config):
    """Конфигурация для тестирования"""
    TESTING = True
    DEBUG = True
    
    # Использовать in-memory SQLite для тестов
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Отключить rate limiting
    RATELIMIT_ENABLED = False
    
    # Использовать mock для AI операций
    USE_LOCAL_AI_MODELS = True
    
    # Минимальные требования для тестов
    MIN_MESSAGES_FOR_ANALYSIS = 1
    ANALYSIS_COOLDOWN = 0


# ==================== Выбор конфигурации ====================

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


# ==================== Пример .env файла ====================
"""
# Скопируйте это в файл .env

# Flask
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# Database
DATABASE_URL=postgresql://user:password@localhost/ai_profiler_db

# Anthropic API
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# AI Settings
USE_LOCAL_AI_MODELS=False
MIN_MESSAGES_FOR_ANALYSIS=5
ANALYSIS_MESSAGE_TRIGGER=10

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0

# Rate Limiting (optional)
RATELIMIT_ENABLED=True
AI_ANALYSIS_RATE_LIMIT=5/hour
"""
