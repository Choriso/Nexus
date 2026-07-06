"""
Извлечение ценностей Schwartz через локальную Ollama (phi3:medium).
"""

from __future__ import annotations

import json
import logging
import re

from sqlalchemy.orm import Session

from app.ai.ollama_service import OllamaService, get_ollama_service
from config import config
from data.ai import UserSchwartzProfile
from data.user import User

logger = logging.getLogger(__name__)

SCHWARTZ_SYSTEM_PROMPT = """Ты — психологический аналитик. Оцени текст пользователя по 10 базовым ценностям теории Schwartz.
Верни ТОЛЬКО валидный JSON без markdown и пояснений.
Ключи (все обязательны, значения float от 0.0 до 1.0):
self_direction, stimulation, hedonism, achievement, power,
security, conformity, tradition, benevolence, universalism"""

SCHWARTZ_KEYS = UserSchwartzProfile.SCHWARTZ_KEYS


def _parse_schwartz_json(raw: str) -> dict[str, float] | None:
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    result: dict[str, float] = {}
    for key in SCHWARTZ_KEYS:
        val = data.get(key)
        if val is None:
            return None
        try:
            result[key] = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return None
    return result


def analyze_schwartz_values(
    text: str,
    *,
    ollama: OllamaService | None = None,
) -> dict[str, float] | None:
    """Вызывает Ollama и парсит JSON с 10 ценностями Schwartz."""
    if not text or not text.strip():
        return None

    service = ollama or get_ollama_service()
    if not service.enabled:
        return None

    prompt = (
        f"Текст пользователя для анализа ценностей:\n\n{text[:3000]}\n\n"
        "Верни JSON с 10 ключами Schwartz (0.0–1.0)."
    )

    try:
        raw = service.generate(
            prompt=prompt,
            system=SCHWARTZ_SYSTEM_PROMPT,
            temperature=0.1,
            max_tokens=300,
        )
        return _parse_schwartz_json(raw or "")
    except Exception as exc:
        logger.warning("Schwartz Ollama analysis failed: %s", exc)
        return None


def upsert_schwartz_profile(
    db: Session,
    user_id: int,
    values: dict[str, float],
    *,
    confidence: float = 0.8,
) -> UserSchwartzProfile:
    profile = db.query(UserSchwartzProfile).filter_by(user_id=user_id).first()
    if not profile:
        profile = UserSchwartzProfile(user_id=user_id)
        db.add(profile)

    for key in SCHWARTZ_KEYS:
        setattr(profile, key, values[key])
    profile.values_json = values
    profile.confidence_score = confidence
    db.commit()
    return profile


def extract_onboarding_text(user: User, messages_text: str = "") -> str:
    parts = []
    if user.information:
        parts.append(user.information.strip())
    if messages_text:
        parts.append(messages_text.strip())
    return " ".join(parts)


def schwartz_cosine_similarity(
    profile_a: UserSchwartzProfile | None,
    profile_b: UserSchwartzProfile | None,
) -> float | None:
    if profile_a is None or profile_b is None:
        return None
    if not profile_a.is_populated() or not profile_b.is_populated():
        return None

    vec_a = profile_a.to_vector()
    vec_b = profile_b.to_vector()
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return None
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))
