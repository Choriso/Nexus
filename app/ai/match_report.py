"""
Генерация мини-докладов (обоснований) для совпавших пользователей.

Приоритет LLM:
1. YandexGPT (если настроен API-ключ)
2. Ollama (если включена)
3. Динамический шаблонный fallback
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy.orm import Session

from app.ai.ollama_service import get_ollama_service
from app.ai_profiler.contextual_adapter import get_contextual_adapter
from config import config
from data.ai import AIExtractedInterests, UserPersonalityProfile, UserSchwartzProfile
from data.behavior import UserBehaviorProfile

logger = logging.getLogger(__name__)

OCEAN_LABELS_RU = {
    "openness": "Открытость",
    "conscientiousness": "Добросовестность",
    "extraversion": "Экстраверсия",
    "agreeableness": "Доброжелательность",
    "neuroticism": "Нейротизм",
}

MATCH_REPORT_SYSTEM_PROMPT = """Ты — эксперт-психолог и AI-аналитик социальной платформы Nexus. Твоя задача — написать краткое, живое и максимально персонализированное обоснование того, почему два пользователя подходят друг другу.

СТРОГИЕ ПРАВИЛА:
1. Используй ТОЛЬКО факты из блока «ДАННЫЕ» ниже. Не выдумывай хобби, навыки или черты характера, которых нет в описании профилей.
2. Категорически запрещены любые шаблонные фразы вроде "Мы подобрали тебе людей...". Пиши сразу по существу взаимодействия.
3. Объём: ровно 2-3 предложения на русском языке.
4. Стиль: интеллектуальный, вовлекающий, дружелюбный. Покажи, как именно их пересекающиеся интересы или взаимодополняющие типы личности (MBTI и OCEAN) помогут им в общении или совместных проектах.
5. Не используй markdown (звёздочки, решётки), списки, кавычки вокруг всего ответа, эмодзи.
6. Не упоминай ИИ, алгоритмы, эмбеддинги или нейросети — пиши естественным языком от лица платформы."""


def _yandexgpt_generate(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 250,
) -> str | None:
    """Выполняет синхронный запрос к YandexGPT API.

    Args:
        prompt: Текст запроса для генерации.
        system: Системный промпт для задания контекста модели.
        temperature: Температура генерации (степень случайности).
        max_tokens: Максимальное количество токенов в ответе.

    Returns:
        Сгенерированный текст ответа или None при ошибке/отсутствии ключа.
    """
    api_key = config.YANDEX_GPT_API_KEY
    if not api_key:
        return None

    import requests

    url = f"{config.YANDEX_GPT_API_BASE.rstrip('/')}/foundationModels/v1/completion"
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    payload = {
        "modelUri": config.YANDEX_GPT_MODEL,
        "completionOptions": {"temperature": temperature, "maxTokens": max_tokens},
        "messages": [{"role": "user", "text": full_prompt}],
    }
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code >= 400:
            logger.warning("[yandexgpt] HTTP %d: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        alternatives = data.get("result", {}).get("alternatives", [])
        if alternatives:
            return alternatives[0].get("message", {}).get("text", "")
        return None
    except Exception as e:
        logger.warning("[yandexgpt] Request failed: %s", e)
        return None


def _serialize_interests(row: AIExtractedInterests | None) -> dict[str, Any]:
    """Преобразует ORM-строку AIExtractedInterests в плоский словарь.

    Args:
        row: ORM-объект с извлечёнными интересами пользователя.

    Returns:
        Словарь с полями hobbies, topics, skills, occupation, semantic_categories.
    """
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
    """Определяет доминирующую психологическую черту пользователя для кастомного fallback ответа.

    Args:
        profile: Профиль личности пользователя (OCEAN).

    Returns:
        Строка с описанием доминирующей черты (высокая/низкая/умеренная).
    """
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
    """Формирует строку со всеми метриками OCEAN для включения в промпт.

    Args:
        profile: Профиль личности пользователя.

    Returns:
        Строка с пятью значениями OCEAN через запятую или "Нет данных".
    """
    if profile is None:
        return "Нет данных"
    return (
        f"Открытость: {profile.openness:.2f}, Добросовестность: {profile.conscientiousness:.2f}, "
        f"Экстраверсия: {profile.extraversion:.2f}, Доброжелательность: {profile.agreeableness:.2f}, "
        f"Нейротизм: {profile.neuroticism:.2f}"
    )


def _collect_directions(interests: dict[str, Any]) -> list[str]:
    """Собирает уникальные направления (подкатегории) из полей интересов.

    Args:
        interests: Словарь с интересами пользователя.

    Returns:
        Список уникальных названий направлений (не более 5).
    """
    directions: list[str] = []
    for field in ("hobbies", "skills", "topics"):
        for item in interests.get(field) or []:
            label = item["subcategory"] if isinstance(item, dict) else str(item)
            if label and label not in directions:
                directions.append(label)
    return directions[:5]


def _find_shared_directions(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Находит пересекающиеся направления интересов между двумя пользователями.

    Args:
        a: Словарь интересов первого пользователя.
        b: Словарь интересов второго пользователя.

    Returns:
        Список общих направлений (регистронезависимое сравнение).
    """
    dirs_a = {d.lower() for d in _collect_directions(a)}
    dirs_b = {d.lower() for d in _collect_directions(b)}
    shared = dirs_a & dirs_b
    return [d for d in _collect_directions(a) if d.lower() in shared]


SCHWARTZ_KEYS_RU = {
    "self_direction": "Самостоятельность", "stimulation": "Стимуляция",
    "hedonism": "Гедонизм", "achievement": "Достижения",
    "power": "Власть", "security": "Безопасность",
    "conformity": "Конформность", "tradition": "Традиции",
    "benevolence": "Доброта", "universalism": "Универсализм",
}


def _schwartz_to_str(profile: UserSchwartzProfile | None) -> str:
    """Формирует строку с тремя главными ценностями по Шварцу для включения в промпт.

    Args:
        profile: Профиль ценностей пользователя (Шварц).

    Returns:
        Строка с топ-3 ценностями и их баллами или "Нет данных".
    """
    if profile is None:
        return "Нет данных"
    items = [(SCHWARTZ_KEYS_RU[k], getattr(profile, k, 0.5)) for k in SCHWARTZ_KEYS_RU]
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[:3]
    return ", ".join(f"{label} ({val:.2f})" for label, val in top)


def _behavior_to_str(profile: UserBehaviorProfile | None) -> str:
    """Формирует строку с поведенческими метриками пользователя для промпта.

    Args:
        profile: Профиль поведения пользователя.

    Returns:
        Строка с описанием поведения или "Нет данных".
    """
    if profile is None:
        return "Нет данных"
    return (
        f"Средняя длина сообщения: {profile.avg_char_count:.0f} символов, "
        f"Среднее время ответа: {profile.avg_reply_time:.1f} мин, "
        f"Эмодзи/сообщение: {profile.avg_emoji_count:.2f}, "
        f"Активность: {profile.message_count} сообщений"
    )


def _goals_str(interests: dict[str, Any]) -> str:
    """Извлекает цели пользователя (краткосрочные и долгосрочные) в строку.

    Args:
        interests: Словарь с интересами пользователя.

    Returns:
        Строка с целями через точку с запятой или "Нет данных".
    """
    parts = []
    stg = interests.get("short_term_goals") or []
    ltg = interests.get("long_term_goals") or []
    if stg:
        parts.append("Ближайшие цели: " + ", ".join(stg))
    if ltg:
        parts.append("Долгосрочные цели: " + ", ".join(ltg))
    return "; ".join(parts) or "Нет данных"


def _build_prompt_payload(
    user_a_profile: UserPersonalityProfile | None,
    user_b_profile: UserPersonalityProfile | None,
    user_a_interests: dict[str, Any],
    user_b_interests: dict[str, Any],
    user_a_schwartz: UserSchwartzProfile | None = None,
    user_b_schwartz: UserSchwartzProfile | None = None,
    user_a_behavior: UserBehaviorProfile | None = None,
    user_b_behavior: UserBehaviorProfile | None = None,
    matched_tags: list[str] | None = None,
) -> str:
    """Собирает полный промпт с данными двух пользователей для LLM.

    Args:
        user_a_profile: Профиль личности текущего пользователя.
        user_b_profile: Профиль личности найденного кандидата.
        user_a_interests: Интересы текущего пользователя.
        user_b_interests: Интересы кандидата.
        user_a_schwartz: Ценности Шварца текущего пользователя.
        user_b_schwartz: Ценности Шварца кандидата.
        user_a_behavior: Поведенческий профиль текущего пользователя.
        user_b_behavior: Поведенческий профиль кандидата.
        matched_tags: Теги найденных пересечений.

    Returns:
        Строка промпта для передачи в LLM.
    """
    adapter = get_contextual_adapter(
        enabled=getattr(config, "CONTEXTUAL_ADAPTER_ENABLED", True)
    )

    a_summary = adapter.build_profile_summary(user_a_interests)
    b_summary = adapter.build_profile_summary(user_b_interests)

    a_mbti = (user_a_profile.mbti_type if user_a_profile else None) or "не определён"
    b_mbti = (user_b_profile.mbti_type if user_b_profile else None) or "не определён"

    a_ocean = _get_detailed_ocean_str(user_a_profile)
    b_ocean = _get_detailed_ocean_str(user_b_profile)

    a_schwartz = _schwartz_to_str(user_a_schwartz)
    b_schwartz = _schwartz_to_str(user_b_schwartz)

    a_behavior = _behavior_to_str(user_a_behavior)
    b_behavior = _behavior_to_str(user_b_behavior)

    a_goals = _goals_str(user_a_interests)
    b_goals = _goals_str(user_b_interests)

    shared = matched_tags or _find_shared_directions(user_a_interests, user_b_interests)
    shared_str = ", ".join(shared) if shared else "общее стремление к развитию"

    return f"""ДАННЫЕ ПОЛЬЗОВАТЕЛЕЙ:

Текущий пользователь (Кому мы пишем разбор):
  Тип личности MBTI: {a_mbti}
  Метрики OCEAN: {a_ocean}
  Ценности (Шварц): {a_schwartz}
  Стиль общения: {user_a_profile.communication_style or 'Стандартный'}, Сотрудничество: {user_a_profile.collaboration_style or 'Стандартный'}
  Поведение: {a_behavior}
  Цели: {a_goals}
  Профиль интересов и контекст: {a_summary}

Найденный кандидат:
  Тип личности MBTI: {b_mbti}
  Метрики OCEAN: {b_ocean}
  Ценности (Шварц): {b_schwartz}
  Стиль общения: {user_b_profile.communication_style or 'Стандартный'}, Сотрудничество: {user_b_profile.collaboration_style or 'Стандартный'}
  Поведение: {b_behavior}
  Цели: {b_goals}
  Профиль интересов и контекст: {b_summary}

Точки пересечения в графе знаний: {shared_str}

Задание: напиши краткое индивидуальное обоснование (2-3 предложения), раскрывающее уникальный потенциал этого союза."""


def _sanitize_llm_output(text: str) -> str:
    """Очищает вывод нейросети от лишних артефактов форматирования.

    Args:
        text: Сырой текст от LLM.

    Returns:
        Очищенный текст без markdown-символов, не более 3 предложений.
    """
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
    """Формирует шаблонное обоснование совместимости без использования LLM.

    Args:
        user_a_profile: Профиль личности текущего пользователя.
        user_b_profile: Профиль личности найденного кандидата.
        user_a_interests: Интересы текущего пользователя.
        user_b_interests: Интересы кандидата.
        matched_tags: Теги найденных пересечений.

    Returns:
        Строка с динамически собранным обоснованием совместимости.
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
    """Генерирует уникальный мини-доклад о совпадении двух пользователей.

    Приоритет:
    1. YandexGPT (если API-ключ настроен)
    2. Ollama (если enabled)
    3. Шаблонный fallback

    Args:
        user_id_1: ID текущего пользователя.
        user_id_2: ID найденного кандидата.
        db_session: Сессия SQLAlchemy для запросов к БД.
        matched_tags: Теги найденных пересечений интересов.
        use_llm: Флаг принудительного использования/отключения LLM.

    Returns:
        Строка с текстом мини-доклада о совместимости.
    """
    profile_a = db_session.query(UserPersonalityProfile).filter_by(user_id=user_id_1).first()
    profile_b = db_session.query(UserPersonalityProfile).filter_by(user_id=user_id_2).first()

    schwartz_a = db_session.query(UserSchwartzProfile).filter_by(user_id=user_id_1).first()
    schwartz_b = db_session.query(UserSchwartzProfile).filter_by(user_id=user_id_2).first()

    behavior_a = db_session.query(UserBehaviorProfile).filter_by(user_id=user_id_1).first()
    behavior_b = db_session.query(UserBehaviorProfile).filter_by(user_id=user_id_2).first()

    interests_row_a = db_session.query(AIExtractedInterests).filter_by(user_id=user_id_1).first()
    interests_row_b = db_session.query(AIExtractedInterests).filter_by(user_id=user_id_2).first()

    interests_a = _serialize_interests(interests_row_a)
    interests_b = _serialize_interests(interests_row_b)

    fallback = build_default_match_report(
        profile_a, profile_b, interests_a, interests_b, matched_tags
    )

    prompt = _build_prompt_payload(
        profile_a, profile_b, interests_a, interests_b,
        user_a_schwartz=schwartz_a, user_b_schwartz=schwartz_b,
        user_a_behavior=behavior_a, user_b_behavior=behavior_b,
        matched_tags=matched_tags,
    )

    if config.YANDEX_GPT_API_KEY:
        try:
            llm_text = _yandexgpt_generate(
                prompt=prompt,
                system=MATCH_REPORT_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=250,
            )
            if llm_text:
                sanitized = _sanitize_llm_output(llm_text)
                if sanitized:
                    logger.info("[match_report] YandexGPT успешно сгенерировал отчёт")
                    return sanitized
        except Exception as e:
            logger.error("YandexGPT report failed: %s", e)

    llm_enabled = use_llm if use_llm is not None else getattr(config, "OLLAMA_ENABLED", False)
    if llm_enabled:
        try:
            ollama = get_ollama_service()
            llm_text = ollama.generate(
                prompt=prompt,
                system=MATCH_REPORT_SYSTEM_PROMPT,
                temperature=0.7,
                max_tokens=250,
            )
            if llm_text:
                sanitized = _sanitize_llm_output(llm_text)
                if sanitized:
                    logger.info("[match_report] Ollama успешно сгенерировал отчёт")
                    return sanitized
        except Exception as e:
            logger.error("Ollama report failed: %s", e)

    return fallback
