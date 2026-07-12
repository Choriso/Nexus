"""
Единая точка конфигурации проекта Nexus.

Все переменные окружения считываются только здесь.
Остальные модули импортируют Config или объект config из этого файла.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


def _env_bool(key: str, default: str = "False") -> bool:
    """
    Возвращает значение переменной окружения как булево.

    Args:
        key (str): Ключ переменной окружения.
        default (str, optional): Значение по умолчанию (строка 'True' или 'False'). По умолчанию "False".

    Returns:
        bool: True, если значение переменной окружения (регистронезависимо) равно "true", иначе False.
    """
    return os.environ.get(key, default).lower() == "true"


def _env_int(key: str, default: int) -> int:
    """
    Преобразует значение переменной окружения в целое число.

    Args:
        key (str): Ключ переменной окружения.
        default (int): Значение по умолчанию.

    Returns:
        int: Целое значение переменной окружения либо значение по умолчанию.
    """
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    """
    Преобразует значение переменной окружения в число с плавающей точкой.

    Args:
        key (str): Ключ переменной окружения.
        default (float): Значение по умолчанию.

    Returns:
        float: Значение переменной окружения как float либо значение по умолчанию.
    """
    return float(os.environ.get(key, str(default)))


def _join_path(base: str, filename: str) -> str:
    """
    Соединяет базовый путь и имя файла в нормализованный путь.

    Args:
        base (str): Базовый путь.
        filename (str): Имя файла.

    Returns:
        str: Объединённый путь.
    """
    return os.path.normpath(os.path.join(base, filename))


class Config:
    """
    Базовая конфигурация приложения, включая настройки Flask, базы данных, ML, Celery и инфраструктуры.

    Все переменные берутся из окружения либо используются значения по умолчанию.
    """

    # Настройки Flask
    SECRET_KEY: str = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"
    FLASK_APP: str = os.environ.get("FLASK_APP", "App.py")
    FLASK_ENV: str = os.environ.get("FLASK_ENV", "development")
    FLASK_HOST: str = os.environ.get("FLASK_HOST", "127.0.0.1")
    FLASK_PORT: int = _env_int("FLASK_PORT", 3000)
    DEBUG: bool = FLASK_ENV == "development"

    UPLOAD_FOLDER: str = os.environ.get("UPLOAD_FOLDER", "static/uploads/")
    MAX_CONTENT_LENGTH: int = _env_int("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)
    ALLOWED_ORIGINS: str | None = os.environ.get("ALLOWED_ORIGINS")

    # Настройки базы данных
    DATABASE_URL: str = os.environ.get("DATABASE_URL") or "sqlite:///ai_profiler.db"
    SQLALCHEMY_DATABASE_URI: str = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    # Конфигурация для начального наполнения базы (seed db)
    SEED_DB_NAME: str = os.environ.get("SEED_DB_NAME", "ai_profiler_db")
    SEED_DB_USER: str = os.environ.get("SEED_DB_USER", "my_app_user")
    SEED_DB_PASSWORD: str = os.environ.get("SEED_DB_PASSWORD", "")
    SEED_DB_HOST: str = os.environ.get("SEED_DB_HOST", "127.0.0.1")
    SEED_DB_PORT: str = os.environ.get("SEED_DB_PORT", "5432")

    # Настройки AI/ML
    USE_LOCAL_AI_MODELS: bool = _env_bool("USE_LOCAL_AI_MODELS", "False")
    EMBEDDING_MODEL: str = os.environ.get(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    SBERT_MODEL_NAME: str = os.environ.get("SBERT_MODEL_NAME", EMBEDDING_MODEL)
    SBERT_ALLOW_DOWNLOAD: bool = _env_bool("SBERT_ALLOW_DOWNLOAD", "True")
    LOCAL_MODELS_PATH: str = os.environ.get("LOCAL_MODELS_PATH", "./models")
    LOCAL_ARTIFACTS_DIR: str = os.environ.get(
        "LOCAL_ARTIFACTS_DIR", str(BASE_DIR / "ml" / "artifacts")
    )
    PERSONALITY_MODEL_FILENAME: str = os.environ.get(
        "PERSONALITY_MODEL_FILENAME", "personality_model_best.pth"
    )
    MBTI_MODEL_FILENAME: str = os.environ.get("MBTI_MODEL_FILENAME", "mbti_model.pth")
    PERSONALITY_MODEL_PATH: str = _join_path(
        LOCAL_ARTIFACTS_DIR, PERSONALITY_MODEL_FILENAME
    )
    MBTI_MODEL_PATH: str = _join_path(LOCAL_ARTIFACTS_DIR, MBTI_MODEL_FILENAME)
    MBTI_NEURAL_BLEND_WEIGHT: float = _env_float("NEXUS_MBTI_NEURAL_BLEND_WEIGHT", 0.7)
    TFIDF_VECTORIZER_PATH: str = os.environ.get(
        "TFIDF_VECTORIZER_PATH", "app/ai/models/tfidf_vectorizer.pkl"
    )

    # Настройки Celery
    CELERY_BROKER_URL: str = os.environ.get(
        "CELERY_BROKER_URL", "redis://localhost:6379/0"
    )
    CELERY_RESULT_BACKEND: str = os.environ.get(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
    )

    # Настройки Redis/Кэша
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
    CACHE_TYPE: str = "RedisCache"
    CACHE_REDIS_URL: str = REDIS_URL
    CACHE_DEFAULT_TIMEOUT: int = 3600

    # Анализ профилей
    MIN_MESSAGES_FOR_ANALYSIS: int = _env_int("MIN_MESSAGES_FOR_ANALYSIS", 5)
    ANALYSIS_COOLDOWN: int = _env_int("ANALYSIS_COOLDOWN", 3600)
    ANALYSIS_MESSAGE_TRIGGER: int = _env_int("ANALYSIS_MESSAGE_TRIGGER", 10)
    MAX_MESSAGES_PER_ANALYSIS: int = _env_int("MAX_MESSAGES_PER_ANALYSIS", 50)

    # Совместимость
    COMPATIBILITY_CACHE_TTL: int = _env_int("COMPATIBILITY_CACHE_TTL", 86400)
    MIN_COMPATIBILITY_SCORE: float = _env_float("MIN_COMPATIBILITY_SCORE", 0.6)

    # Поиск
    SEMANTIC_SEARCH_THRESHOLD: float = _env_float("SEMANTIC_SEARCH_THRESHOLD", 0.5)
    MAX_SEARCH_RESULTS: int = _env_int("MAX_SEARCH_RESULTS", 20)

    # Контекстуальный адаптер (обогащение сленга перед SBERT)
    CONTEXTUAL_ADAPTER_ENABLED: bool = _env_bool("CONTEXTUAL_ADAPTER_ENABLED", "True")

    # Локальная LLM (Ollama) для мини-докладов о совпадениях
    OLLAMA_ENABLED: bool = _env_bool("OLLAMA_ENABLED", "False")
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct-q4_K_M")
    OLLAMA_TIMEOUT: int = _env_int("OLLAMA_TIMEOUT", 90)
    MATCH_REPORT_LLM_LIMIT: int = _env_int("MATCH_REPORT_LLM_LIMIT", 5)

    # Приватность и безопасность
    REQUIRE_ANALYSIS_CONSENT: bool = _env_bool("REQUIRE_ANALYSIS_CONSENT", "True")
    ANONYMIZE_TRAINING_DATA: bool = _env_bool("ANONYMIZE_TRAINING_DATA", "True")
    MAX_CHAT_HISTORY_DAYS: int = _env_int("MAX_CHAT_HISTORY_DAYS", 90)

    # =========================================================================
    # Block 1: Two-Stage LLM Classifier — Multi-Provider Failover Cascade
    # =========================================================================
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE: str = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    YANDEX_GPT_API_KEY: str = os.environ.get("YANDEX_GPT_API_KEY", "")
    YANDEX_GPT_API_BASE: str = os.environ.get("YANDEX_GPT_API_BASE", "https://llm.api.cloud.yandex.net")
    YANDEX_GPT_MODEL: str = os.environ.get("YANDEX_GPT_MODEL", "gpt://b1gd7uvpjf1qlla85o97/yandexgpt-5.1/latest")

    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GROQ_API_BASE: str = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
    GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    LLM_CLASSIFIER_TIMEOUT: int = _env_int("LLM_CLASSIFIER_TIMEOUT", 15)
    LLM_CLASSIFIER_TOP_K: int = _env_int("LLM_CLASSIFIER_TOP_K", 10)
    LLM_CLASSIFIER_FALLBACK_TO_OLLAMA: bool = _env_bool("LLM_CLASSIFIER_FALLBACK_TO_OLLAMA", "True")
    LLM_CLASSIFIER_CONFIDENCE_THRESHOLD: float = _env_float("LLM_CLASSIFIER_CONFIDENCE_THRESHOLD", 0.6)

    # =========================================================================
    # Block 2: Hybrid Search Ranking — Two-Contour Optimization
    # =========================================================================
    RANKING_WEIGHT_OCEAN: float = _env_float("RANKING_WEIGHT_OCEAN", 0.35)
    RANKING_WEIGHT_GRAPH: float = _env_float("RANKING_WEIGHT_GRAPH", 0.40)
    RANKING_WEIGHT_JACCARD: float = _env_float("RANKING_WEIGHT_JACCARD", 0.25)

    GLOBAL_LEARNING_RATE: float = _env_float("GLOBAL_LEARNING_RATE", 0.01)
    WEIGHT_ADJUSTMENT_INTERVAL: int = _env_int("WEIGHT_ADJUSTMENT_INTERVAL", 86400)
    MICRO_GRADIENT_LEARNING_RATE: float = _env_float("MICRO_GRADIENT_LEARNING_RATE", 0.05)
    MIN_ACTIVE_USERS_FOR_GLOBAL_ADJUSTMENT: int = _env_int("MIN_ACTIVE_USERS_FOR_GLOBAL_ADJUSTMENT", 5)

    # Root Personality Archetype Blend
    ROOT_PERSONALITY_BLEND_WEIGHT: float = _env_float("ROOT_PERSONALITY_BLEND_WEIGHT", 0.30)
    PERSONALITY_OCEAN_WEIGHT: float = _env_float("PERSONALITY_OCEAN_WEIGHT", 0.60)
    PERSONALITY_SCHWARTZ_WEIGHT: float = _env_float("PERSONALITY_SCHWARTZ_WEIGHT", 0.40)

    # Rate Limiting
    RATELIMIT_STORAGE_URL: str = REDIS_URL
    RATELIMIT_DEFAULT: str = "100/hour"
    RATELIMIT_STRATEGY: str = "fixed-window"
    RATELIMIT_ENABLED: bool = _env_bool("RATELIMIT_ENABLED", "True")
    AI_ANALYSIS_RATE_LIMIT: str = os.environ.get("AI_ANALYSIS_RATE_LIMIT", "5/hour")
    AI_COMPATIBILITY_RATE_LIMIT: str = os.environ.get(
        "AI_COMPATIBILITY_RATE_LIMIT", "20/hour"
    )
    AI_SEARCH_RATE_LIMIT: str = os.environ.get("AI_SEARCH_RATE_LIMIT", "30/hour")

    # Мониторинг
    LOG_AI_OPERATIONS: bool = _env_bool("LOG_AI_OPERATIONS", "True")
    AI_LOG_LEVEL: str = os.environ.get("AI_LOG_LEVEL", "INFO")

    @classmethod
    def is_production(cls) -> bool:
        """
        Определяет, работает ли приложение в production-окружении.

        Returns:
            bool: True, если FLASK_ENV равен "production", иначе False.
        """
        return cls.FLASK_ENV == "production"

    @classmethod
    def allowed_origins_list(cls) -> list[str] | str:
        """
        Возвращает список разрешённых origins для CORS, либо "*" для всех.

        Returns:
            list[str] | str: Список строк origins или "*" (разрешено всё).
        """
        if cls.ALLOWED_ORIGINS:
            return [origin.strip() for origin in cls.ALLOWED_ORIGINS.split(",")]
        return "*"

    @classmethod
    def seed_db_config(cls) -> dict[str, str]:
        """
        Возвращает словарь с параметрами для инициализации базы данных (seed).

        Returns:
            dict[str, str]: Параметры подключения к базе.
        """
        return {
            "dbname": cls.SEED_DB_NAME,
            "user": cls.SEED_DB_USER,
            "password": cls.SEED_DB_PASSWORD,
            "host": cls.SEED_DB_HOST,
            "port": cls.SEED_DB_PORT,
        }


class DevelopmentConfig(Config):
    """
    Конфигурация для разработки.

    Все значения выставлены для облегчения разработки (меньшие лимиты, отключен rate-limiting).
    """
    DEBUG = True
    TESTING = False
    MIN_MESSAGES_FOR_ANALYSIS = 3
    ANALYSIS_MESSAGE_TRIGGER = 5
    ANALYSIS_COOLDOWN = 300
    RATELIMIT_ENABLED = False


class ProductionConfig(Config):
    """
    Конфигурация для production.

    Значения выставлены для боевого режима: более строгие лимиты,
    включённое rate-limiting и защищённые куки.
    """
    DEBUG = False
    TESTING = False
    MIN_MESSAGES_FOR_ANALYSIS = 10
    ANALYSIS_MESSAGE_TRIGGER = 15
    ANALYSIS_COOLDOWN = 7200
    RATELIMIT_ENABLED = True
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


class TestingConfig(Config):
    """
    Конфигурация для тестирования.

    Использует SQLite в памяти, максимально "разрешающая": нет лимитов, включены локальные модели.
    """
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    DATABASE_URL = SQLALCHEMY_DATABASE_URI
    RATELIMIT_ENABLED = False
    USE_LOCAL_AI_MODELS = True
    MIN_MESSAGES_FOR_ANALYSIS = 1
    ANALYSIS_COOLDOWN = 0


config_by_name: dict[str, type[Config]] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config() -> type[Config]:
    """
    Возвращает класс конфигурации на основе значения FLASK_ENV.

    Returns:
        type[Config]: Класс конфигурации для указанного окружения.
    """
    return config_by_name.get(Config.FLASK_ENV, config_by_name["default"])


config = get_config()
