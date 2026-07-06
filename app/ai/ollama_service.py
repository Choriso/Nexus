"""
Изолированный клиент для локальной Ollama (http://localhost:11434/api/generate).

Не бросает исключений наружу — при недоступности LLM возвращает None,
чтобы вызывающий код мог отдать дефолтный текст без 500-й ошибки.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from config import config

logger = logging.getLogger(__name__)


class OllamaService:
    """
    Тонкий HTTP-клиент Ollama API.

    Конфигурация через config.py:
        OLLAMA_ENABLED, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.enabled = enabled if enabled is not None else config.OLLAMA_ENABLED
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT

    @property
    def generate_url(self) -> str:
        return f"{self.base_url}/api/generate"

    def is_available(self) -> bool:
        """Быстрая проверка доступности Ollama (GET /api/tags)."""
        if not self.enabled:
            return False
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        system: str = "",
        *,
        temperature: float = 0.3,
        max_tokens: int = 200,
    ) -> str | None:
        """
        Вызов POST /api/generate.

        Args:
            prompt: Пользовательский промпт.
            system: Системный промпт (передаётся в поле system Ollama).
            temperature: Низкая температура снижает галлюцинации.
            max_tokens: Лимит токенов ответа.

        Returns:
            str | None: Сгенерированный текст или None при ошибке/отключении.
        """
        if not self.enabled:
            logger.debug("Ollama отключена в конфиге (OLLAMA_ENABLED=False)")
            return None

        if not prompt or not prompt.strip():
            return None

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = (data.get("response") or "").strip()
            return text if text else None
        except requests.Timeout:
            logger.warning("Ollama timeout (%ss) для модели %s", self.timeout, self.model)
            return None
        except requests.RequestException as exc:
            logger.warning("Ollama недоступна: %s", exc)
            return None


_ollama_instance: OllamaService | None = None


def get_ollama_service() -> OllamaService:
    """Ленивый singleton клиента Ollama."""
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaService()
    return _ollama_instance
