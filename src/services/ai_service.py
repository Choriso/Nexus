import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAIService(ABC):
    """
    Базовый интерфейс для всех AI-провайдеров.
    Код приложения должен зависеть только от этого интерфейса.
    """

    @abstractmethod
    def generate_reply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        ...


class DisabledAIService(BaseAIService):
    """
    Стратегия по умолчанию, когда AI отключен.
    Возвращает детерминированный мок, чтобы не ломать приложение.
    """

    def generate_reply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        return "[AI отключен] Этот ответ сгенерирован мок-сервисом. " \
               "Проверьте переменные окружения AI_ENABLED и AI_PROVIDER."


class LocalAIService(BaseAIService):
    """
    Стратегия для локальных моделей (Ollama, llama-cpp, локальный HTTP-сервис и т.п.).
    Здесь оставлена заготовка, чтобы вы могли подставить свой клиент.
    """

    def __init__(self, endpoint: Optional[str] = None) -> None:
        # Пример: локальный HTTP endpoint, можно брать из .env
        self._endpoint = endpoint or os.getenv("LOCAL_AI_ENDPOINT", "http://localhost:11434")

    def generate_reply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        # TODO: заменить на реальный вызов локальной модели / прокси.
        # Сейчас — безопасный мок с эхо-поведением.
        return f"[LOCAL AI MOCK] prompt='{prompt[:120]}' endpoint='{self._endpoint}'"


class ExternalAIService(BaseAIService):
    """
    Стратегия для внешних провайдеров (Anthropic, OpenAI и т.п.).
    Конкретная реализация зависит от того, какие SDK вы хотите использовать.
    """

    def __init__(self, provider_name: str = "anthropic") -> None:
        self._provider_name = provider_name
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    def generate_reply(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        # TODO: интегрировать выбранный SDK (anthropic / openai и т.п.).
        # Сейчас — безопасный заглушечный ответ.
        return f"[EXTERNAL AI MOCK:{self._provider_name}] prompt='{prompt[:120]}'"


def get_ai_service() -> BaseAIService:
    """
    Фабрика стратегий.

    Управляется переменными окружения:
    - AI_ENABLED: "true"/"false"
    - AI_PROVIDER: "local" | "external"
    """
    enabled_raw = os.getenv("AI_ENABLED", "false").strip().lower()
    is_enabled = enabled_raw in {"1", "true", "yes", "on"}

    if not is_enabled:
        return DisabledAIService()

    provider = os.getenv("AI_PROVIDER", "local").strip().lower()

    if provider == "external":
        return ExternalAIService()

    # Значение по умолчанию — локальный провайдер
    return LocalAIService()

