"""
Генерация мини-докладов (обоснований) для совпавших пользователей.

Использует локальную Ollama (llama3/gemma2) с промптом, ориентированным
на глубокий психологический анализ и уникальность пользователей.
При недоступности LLM — вариативный, динамический fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy.orm import Session

from app.ai.ollama_service import get_ollama_service
from app.ai_profiler.contextual_adapter import get_contextual_adapter
from config import config  # Используем глобальный объект config
from data.ai import AIExtractedInterests, UserPersonalityProfile

logger = logging.getLogger(__name__)

OCEAN_LABELS_RU = {
    "openness": "Открытость",
    "conscientiousness": "Добросовестность",
    "extraversion": "Экстраверсия",
    "agreeableness": "Доброжелательность",
    "neuroticism": "Нейротизм",
}

# ИСПРАВЛЕНО: Убраны жесткие рамки единого шаблона строки.
# Теперь модель фокусируется на поиске связей между характерами и интересами.
MATCH_REPORT_SYSTEM_PROMPT = """Ты — эксперт-психолог и AI-аналитик социальной платформы Nexus. Твоя задача — написать краткое, живое и максимально персонализированное обоснование того, почему два пользователя подходят друг другу.

СТРОГИЕ ПРАВИЛА:
1. Используй ТОЛЬКО факты из блока «ДАННЫЕ» ниже. Не выдумывай хобби, навыки или черты характера, которых нет в описании профилей.
2. Категорически запрещены любые шаблонные фразы вроде "Мы подобрали тебе людей...". Пиши сразу по существу взаимодействия.
3. Объём: ровно 2-3 предложения на русском языке.
4. Стиль: интеллектуальный, вовлекающий, дружелюбный. Покажи, как именно их пересекающиеся интересы или взаимодополняющие типы личности (MBTI и OCEAN) помогут им в общении или совместных проектах.
5. Не используй markdown (звёздочки, решётки), списки, кавычки вокруг всего ответа, эмодзи.
6. Не упоминай ИИ, алгоритмы, эмбеддинги или нейросети — пиши естественным языком от лица платформы."""


def _serialize_interests(row: AIExtractedInterests | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        "hobbies": row.hobbies or [],
        "topics": row.topics or [],
        "skills": row.skills or [],
        "occupation": row.occupation,
        "semantic_categories": getattr(row, "semantic_categories", None) or {},
    }


def _dominant_ocean_trait(profile: UserPersonalityProfile | None) -> str:
    """Определяет доминирующую психологическую черту пользователя для кастомного fallback ответа"""
    if profile is None:
        return "сбалансированный профиль"
    scores = {
        "Открытость": profile.openness or 0.5,
        "Добросовестность": profile.conscientiousness or 0.5,
        "Экстраверсия": profile.extraversion or 0.5,
        "Доброжелательность": profile.agreeableness or 0.5,
    }
    trait, value = max(scores.items(), key=lambda x: x[1])
    if value >= 0.65:
        return f"высокая {trait.lower()}"
    if value <= 0.35:
        return f"низкая {trait.lower()}"
    return f"умеренная {trait.lower()}"


def _get_detailed_ocean_str(profile: UserPersonalityProfile | None) -> str:
    if profile is None:
        return "Нет данных"
    return (
        f"Открытость: {profile.openness:.2f}, Добросовестность: {profile.conscientiousness:.2f}, "
        f"Экстраверсия: {profile.extraversion:.2f}, Доброжелательность: {profile.agreeableness:.2f}, "
        f"Нейротизм: {profile.neuroticism:.2f}"
    )


def _collect_directions(interests: dict[str, Any]) -> list[str]:
    directions: list[str] = []
    for field in ("hobbies", "skills", "topics"):
        for item in interests.get(field) or []:
            label = item["subcategory"] if isinstance(item, dict) else str(item)
            if label and label not in directions:
                directions.append(label)
    return directions[:5]


def _find_shared_directions(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    dirs_a = {d.lower() for d in _collect_directions(a)}
    dirs_b = {d.lower() for d in _collect_directions(b)}
    shared = dirs_a & dirs_b
    return [d for d in _collect_directions(a) if d.lower() in shared]


def _build_prompt_payload(
    user_a_profile: UserPersonalityProfile | None,
    user_b_profile: UserPersonalityProfile | None,
    user_a_interests: dict[str, Any],
    user_b_interests: dict[str, Any],
    matched_tags: list[str] | None = None,
) -> str:
    adapter = get_contextual_adapter(
        enabled=getattr(config, "CONTEXTUAL_ADAPTER_ENABLED", True)
    )

    a_summary = adapter.build_profile_summary(user_a_interests)
    b_summary = adapter.build_profile_summary(user_b_interests)

    a_mbti = (user_a_profile.mbti_type if user_a_profile else None) or "не определён"
    b_mbti = (user_b_profile.mbti_type if user_b_profile else None) or "не определён"

    a_ocean = _get_detailed_ocean_str(user_a_profile)
    b_ocean = _get_detailed_ocean_str(user_b_profile)

    shared = matched_tags or _find_shared_directions(user_a_interests, user_b_interests)
    shared_str = ", ".join(shared) if shared else "общее стремление к развитию"

    return f"""ДАННЫЕ ПОЛЬЗОВАТЕЛЕЙ:

Текущий пользователь (Кому мы пишем разбор):
  Тип личности MBTI: {a_mbti}
  Метрики OCEAN: {a_ocean}
  Профиль интересов и контекст: {a_summary}

Найденный кандидат:
  Тип личности MBTI: {b_mbti}
  Метрики OCEAN: {b_ocean}
  Профиль интересов и контекст: {b_summary}

Точки пересечения в графе знаний: {shared_str}

Задание: напиши краткое индивидуальное обоснование (2-3 предложения), раскрывающее уникальный потенциал этого союза."""


def _sanitize_llm_output(text: str) -> str:
    """Очищает вывод нейросети от лишних артефактов форматирования."""
    cleaned = re.sub(r"[*#_`]", "", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:3])


def build_default_match_report(
    user_a_profile: UserPersonalityProfile | None,
    user_b_profile: UserPersonalityProfile | None,
    user_a_interests: dict[str, Any],
    user_b_interests: dict[str, Any],
    matched_tags: list[str] | None = None,
) -> str:
    """
    ИСПРАВЛЕНО: Новый динамический алгоритмический fallback.
    Формирует естественное предложение, отражающее реальные черты, даже если нейросеть отключена.
    """
    shared = matched_tags or _find_shared_directions(user_a_interests, user_b_interests)
    direction = shared[0] if shared else None

    if not direction:
        dirs_a = _collect_directions(user_a_interests)
        dirs_b = _collect_directions(user_b_interests)
        direction = (dirs_a + dirs_b)[0] if (dirs_a or dirs_b) else "общих интересов"

    a_mbti = (user_a_profile.mbti_type if user_a_profile else None) or "Гибкий"
    b_mbti = (user_b_profile.mbti_type if user_b_profile else None) or "Сбалансированный"
    a_trait = _dominant_ocean_trait(user_a_profile)

    # Кастомная динамическая сборка строки вместо жесткого шаблона "Мы подобрали..."
    return (
        f"Ваш пересекающийся интерес к сфере «{direction}» открывает отличные возможности для взаимодействия. "
        f"Ваш профиль личности ({a_mbti}, для которого характерна {a_trait}) гармонично дополняет "
        f"подход пользователя со структурой {b_mbti}. Это создает прекрасную почву для обмена уникальным опытом."
    )


def generate_match_report(
    user_id_1: int,
    user_id_2: int,
    db_session: Session,
    *,
    matched_tags: list[str] | None = None,
    use_llm: bool | None = None,
) -> str:
    """
    Генерирует уникальный мини-доклад о совпадении двух пользователей.
    """
    profile_a = db_session.query(UserPersonalityProfile).filter_by(user_id=user_id_1).first()
    profile_b = db_session.query(UserPersonalityProfile).filter_by(user_id=user_id_2).first()

    interests_row_a = db_session.query(AIExtractedInterests).filter_by(user_id=user_id_1).first()
    interests_row_b = db_session.query(AIExtractedInterests).filter_by(user_id=user_id_2).first()

    interests_a = _serialize_interests(interests_row_a)
    interests_b = _serialize_interests(interests_row_b)

    fallback = build_default_match_report(
        profile_a, profile_b, interests_a, interests_b, matched_tags
    )

    # Работаем строго с инстансом config из config.py
    llm_enabled = use_llm if use_llm is not None else getattr(config, "OLLAMA_ENABLED", False)
    if not llm_enabled:
        return fallback

    try:
        ollama = get_ollama_service()
        prompt = _build_prompt_payload(
            profile_a, profile_b, interests_a, interests_b, matched_tags
        )

        llm_text = ollama.generate(
            prompt=prompt,
            system=MATCH_REPORT_SYSTEM_PROMPT,
            temperature=0.7,  # Повысили температуру для большего разнообразия фраз и свободы формулировок
            max_tokens=250,
        )

        if not llm_text:
            logger.debug("Ollama не вернула текст, используем fallback")
            return fallback

        sanitized = _sanitize_llm_output(llm_text)
        return sanitized if sanitized else fallback

    except Exception as e:
        logger.error("Ошибка при генерации отчета через Ollama: %s", e)
        return fallback