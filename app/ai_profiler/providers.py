from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import aiohttp

from config import config

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT_TEMPLATE = """
You are an expert tag classification system. Given a RAW user tag and a list of candidate categories, select the SINGLE best matching category slug.

RAW TAG: "{raw_tag}"

CANDIDATE CATEGORIES:
{candidates_text}

RULES:
1. Match the raw tag to the MOST semantically appropriate category from the list above.
2. Return ONLY a valid JSON object with no additional text.
3. If a category fits: {{"status": "matched", "slug": "chosen_slug", "confidence": 0.95, "reason": "brief explanation"}}
4. If NONE of the candidates fit (the tag is completely new/unrelated): {{"status": "create", "suggested_slug": "new_lowercase_slug", "confidence": 0.8, "reason": "why none fit"}}
5. The suggested_slug must be lowercase with underscores, descriptive, and unique.
6. Confidence must be between 0.0 and 1.0.
7. "reason" must be very brief (under 10 words).
""".strip()


def _format_candidates(candidates: list[dict]) -> str:
    """Форматирует список кандидатов для включения в промпт LLM-классификатора.

    Args:
        candidates: Список словарей с ключами slug, name, path.

    Returns:
        Строка с нумерованным списком кандидатов.
    """
    lines = []
    for i, c in enumerate(candidates, 1):
        name = c.get("name", c.get("slug", "unknown"))
        slug = c.get("slug", "")
        path = c.get("path", slug)
        lines.append(f"{i}. slug={slug} | name={name} | path={path}")
    return "\n".join(lines)


def _parse_llm_response(raw: str) -> Optional[dict]:
    """Парсит JSON-ответ LLM-классификатора в структурированный результат.

    Args:
        raw: Сырая строка ответа от LLM.

    Returns:
        Словарь с ключами status, slug/suggested_slug, confidence, reason
        или None при ошибке парсинга.
    """
    import json
    import re

    if not raw or not raw.strip():
        return None

    cleaned = raw.strip()
    json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            cleaned = cleaned.replace("'", '"')
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    status = data.get("status")
    if status == "matched":
        slug = data.get("slug", "").strip()
        if not slug:
            return None
        return {
            "status": "matched",
            "slug": slug,
            "confidence": float(data.get("confidence", 0.8)),
            "reason": data.get("reason", ""),
        }
    if status == "create":
        suggested = data.get("suggested_slug", "").strip()
        if not suggested:
            return None
        return {
            "status": "create",
            "suggested_slug": suggested,
            "confidence": float(data.get("confidence", 0.6)),
            "reason": data.get("reason", ""),
        }

    return None


class LLMProvider(ABC):
    """Абстрактный базовый класс для провайдеров LLM-классификации тегов."""

    def __init__(self, api_key: str, api_base: str, model: str, timeout: int):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    async def _request(self, payload: dict, headers: dict, url: str) -> Optional[str]:
        ...

    async def classify(self, raw_tag: str, candidates: list[dict]) -> Optional[dict]:
        """Отправляет сырой тег и кандидаты в LLM для классификации.

        Args:
            raw_tag: Исходный тег пользователя.
            candidates: Список кандидатов из иерархии.

        Returns:
            Результат классификации или None при ошибке.
        """
        try:
            prompt = CLASSIFIER_PROMPT_TEMPLATE.format(
                raw_tag=raw_tag,
                candidates_text=_format_candidates(candidates),
            )
            payload = self._build_payload(prompt)
            headers = self._build_headers()
            url = self._build_url()

            raw_response = await self._request(payload, headers, url)
            if not raw_response:
                return None

            result = _parse_llm_response(raw_response)
            if result:
                result["provider"] = self.provider_name
            return result

        except Exception as e:
            logger.warning(f"[{self.provider_name}] classify error: {e}")
            return None

    def _build_payload(self, prompt: str) -> dict:
        """Собирает тело запроса для LLM API.

        Args:
            prompt: Текст промпта.

        Returns:
            Словарь с телом запроса.
        """
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 256,
        }

    def _build_headers(self) -> dict:
        """Собирает заголовки HTTP-запроса для LLM API.

        Returns:
            Словарь с заголовками.
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_url(self) -> str:
        """Формирует URL эндпоинта LLM API.

        Returns:
            Строка URL.
        """
        return f"{self.api_base}/chat/completions"


class DeepSeekProvider(LLMProvider):
    """Провайдер DeepSeek для LLM-классификации тегов."""

    @property
    def provider_name(self) -> str:
        return "deepseek"


class OpenAIProvider(LLMProvider):
    """Провайдер OpenAI для LLM-классификации тегов."""

    @property
    def provider_name(self) -> str:
        return "openai"

    def _build_url(self) -> str:
        return f"{self.api_base}/chat/completions"


class GroqProvider(LLMProvider):
    """Провайдер Groq для LLM-классификации тегов."""

    @property
    def provider_name(self) -> str:
        return "groq"

    def _build_url(self) -> str:
        return f"{self.api_base}/chat/completions"


class YandexGPTProvider(LLMProvider):
    """Провайдер YandexGPT для LLM-классификации тегов."""

    @property
    def provider_name(self) -> str:
        return "yandexgpt"

    def _build_headers(self) -> dict:
        return {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_url(self) -> str:
        return f"{self.api_base}/foundationModels/v1/completion"

    def _build_payload(self, prompt: str) -> dict:
        return {
            "modelUri": self.model,
            "completionOptions": {"temperature": 0.1, "maxTokens": 256},
            "messages": [{"role": "user", "text": prompt}],
        }

    async def _request(self, payload: dict, headers: dict, url: str) -> Optional[str]:
        """Выполняет асинхронный запрос к YandexGPT API.

        Args:
            payload: Тело запроса.
            headers: Заголовки HTTP.
            url: URL эндпоинта.

        Returns:
            Текст ответа или None при ошибке.
        """
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning(f"[yandexgpt] HTTP {resp.status}: {text[:200]}")
                    return None
                data = await resp.json()
                result = data.get("result", {})
                alternatives = result.get("alternatives", [])
                if alternatives:
                    message = alternatives[0].get("message", {})
                    return message.get("text", "")
                return None


class OllamaProvider(LLMProvider):
    """Провайдер Ollama для LLM-классификации тегов."""

    def __init__(self, base_url: str, model: str, timeout: int):
        super().__init__("", base_url, model, timeout)

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _build_headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _build_url(self) -> str:
        return f"{self.api_base}/api/chat"

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        }

    async def _request(self, payload: dict, headers: dict, url: str) -> Optional[str]:
        """Выполняет асинхронный запрос к Ollama API.

        Args:
            payload: Тело запроса.
            headers: Заголовки HTTP.
            url: URL эндпоинта.

        Returns:
            Текст ответа или None при ошибке.
        """
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning(f"[ollama] HTTP {resp.status}: {text[:200]}")
                    return None
                data = await resp.json()
                msg = data.get("message", {})
                return msg.get("content", "")


class FailoverCascade:
    """Каскадный отказоустойчивый классификатор с перебором провайдеров."""

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers

    async def classify(self, raw_tag: str, candidates: list[dict]) -> Optional[dict]:
        """Пытается классифицировать тег через каждого провайдера по очереди.

        Args:
            raw_tag: Исходный тег пользователя.
            candidates: Список кандидатов из иерархии.

        Returns:
            Результат первого успешного провайдера или None.
        """
        last_error = None
        for i, provider in enumerate(self.providers):
            try:
                logger.info(
                    "[cascade] Trying %s provider (%d/%d) for '%s'",
                    provider.provider_name, i + 1, len(self.providers), raw_tag,
                )
                result = await provider.classify(raw_tag, candidates)
                if result is not None:
                    logger.info(
                        "[cascade] %s succeeded for '%s' (status=%s, slug=%s)",
                        provider.provider_name, raw_tag,
                        result.get("status"), result.get("slug") or result.get("suggested_slug"),
                    )
                    result["provider"] = provider.provider_name
                    return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "[cascade] %s failed for '%s': %s",
                    provider.provider_name, raw_tag, e,
                )

        logger.error("[cascade] All providers failed for '%s': %s", raw_tag, last_error)
        return None


def build_cascade() -> FailoverCascade:
    """Собирает каскад провайдеров LLM из конфигурации проекта.

    Порядок: DeepSeek, OpenAI, YandexGPT, Groq, Ollama (fallback).

    Returns:
        Экземпляр FailoverCascade с настроенными провайдерами.
    """
    timeout = config.LLM_CLASSIFIER_TIMEOUT
    providers: list[LLMProvider] = []

    if config.DEEPSEEK_API_KEY:
        providers.append(DeepSeekProvider(
            config.DEEPSEEK_API_KEY, config.DEEPSEEK_API_BASE,
            config.DEEPSEEK_MODEL, timeout,
        ))

    if config.OPENAI_API_KEY:
        providers.append(OpenAIProvider(
            config.OPENAI_API_KEY, config.OPENAI_API_BASE,
            config.OPENAI_MODEL, timeout,
        ))

    if config.YANDEX_GPT_API_KEY:
        providers.append(YandexGPTProvider(
            config.YANDEX_GPT_API_KEY, config.YANDEX_GPT_API_BASE,
            config.YANDEX_GPT_MODEL, timeout,
        ))

    if config.GROQ_API_KEY:
        providers.append(GroqProvider(
            config.GROQ_API_KEY, config.GROQ_API_BASE,
            config.GROQ_MODEL, timeout,
        ))

    fallback_to_ollama = getattr(config, "LLM_CLASSIFIER_FALLBACK_TO_OLLAMA", True)
    if fallback_to_ollama and getattr(config, "OLLAMA_ENABLED", False):
        providers.append(OllamaProvider(
            config.OLLAMA_BASE_URL, config.OLLAMA_MODEL, timeout,
        ))

    if not providers:
        logger.warning("[cascade] No external providers configured, using Ollama as emergency fallback")
        if getattr(config, "OLLAMA_ENABLED", False):
            providers.append(OllamaProvider(
                config.OLLAMA_BASE_URL, config.OLLAMA_MODEL, timeout,
            ))

    return FailoverCascade(providers)
