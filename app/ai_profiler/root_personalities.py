from __future__ import annotations

import logging
import math

from data.ai import UserPersonalityProfile, UserSchwartzProfile

logger = logging.getLogger(__name__)

ROOT_ARCHETYPES = {
    "work": {
        "name_ru": "Работа",
        "ocean": [0.55, 0.85, 0.40, 0.55, 0.25],
        "ocean_labels": [
            "Открыт новому", "Максимально добросовестный", "Сдержанный",
            "Коллегиальный", "Эмоционально стабильный",
        ],
        "schwartz": {
            "self_direction": 0.80, "stimulation": 0.30, "hedonism": 0.20,
            "achievement": 0.90, "power": 0.55, "security": 0.65,
            "conformity": 0.60, "tradition": 0.40, "benevolence": 0.50,
            "universalism": 0.45,
        },
        "description": "Целеустремлённый профессионал, ценящий порядок, результат и стабильность",
        "match_reason": "Профессиональный интерес",
    },
    "entertainment": {
        "name_ru": "Хобби",
        "ocean": [0.80, 0.35, 0.70, 0.50, 0.40],
        "ocean_labels": [
            "Открытый новому", "Гибкий", "Общительный",
            "Нейтральный", "Умеренно реактивный",
        ],
        "schwartz": {
            "self_direction": 0.70, "stimulation": 0.85, "hedonism": 0.80,
            "achievement": 0.40, "power": 0.30, "security": 0.30,
            "conformity": 0.20, "tradition": 0.20, "benevolence": 0.60,
            "universalism": 0.55,
        },
        "description": "Творческая, импульсивная личность, ищущая новых впечатлений",
        "match_reason": "Совпадение по интересам",
    },
    "life": {
        "name_ru": "Психология",
        "ocean": [0.75, 0.50, 0.40, 0.80, 0.55],
        "ocean_labels": [
            "Глубоко рефлексирующий", "Умеренный", "Интровертный",
            "Эмпатичный", "Чувствительный",
        ],
        "schwartz": {
            "self_direction": 0.75, "stimulation": 0.40, "hedonism": 0.30,
            "achievement": 0.30, "power": 0.20, "security": 0.50,
            "conformity": 0.30, "tradition": 0.30, "benevolence": 0.85,
            "universalism": 0.90,
        },
        "description": "Эмпатичный и рефлексирующий, ценящий глубокие связи и самопознание",
        "match_reason": "Общие интересы",
    },
}

OCEAN_WEIGHT = 0.6
SCHWARTZ_WEIGHT = 0.4
OCEAN_DIMS = 5

KNOWLEDGE_CATEGORY_TO_ROOT = {
    "work": "work",
    "hobby": "entertainment",
    "entertainment": "entertainment",
    "psychology": "life",
    "life": "life",
}

_ROOT_SLUG_TO_CATEGORY = {
    "it_development": "work",
    "science_education": "work",
    "finance_business": "work",
    "gaming": "entertainment",
    "music_audio": "entertainment",
    "cinema_video": "entertainment",
    "literature_reading": "entertainment",
    "creativity_art": "life",
    "self_development": "life",
    "sports_active_life": "life",
    "psychology_relations": "life",
    "home_lifestyle": "life",
}


def find_root_category(target_node, hierarchy_cache: dict) -> str:
    """Определяет корневую категорию узла иерархии, поднимаясь до самого верха.

    Args:
        target_node: Целевой узел InterestHierarchyNode.
        hierarchy_cache: Предзагруженный кэш иерархии {node_id: data}.

    Returns:
        Строка с названием корневой категории (work/entertainment/life).
    """
    node_id = target_node.id
    visited = set()
    while node_id in hierarchy_cache and node_id not in visited:
        visited.add(node_id)
        parent_id = hierarchy_cache[node_id].get("parent_id")
        if parent_id is None:
            root_slug = hierarchy_cache[node_id].get("slug", "")
            return _ROOT_SLUG_TO_CATEGORY.get(root_slug, "life")
        node_id = parent_id
    return "life"


def compute_ocean_similarity(
    user_ocean: list[float], archetype_ocean: list[float],
) -> float:
    """Вычисляет косинусную/евклидову близость OCEAN-профиля пользователя к архетипу.

    Args:
        user_ocean: Вектор OCEAN пользователя (5 значений).
        archetype_ocean: Эталонный вектор OCEAN архетипа.

    Returns:
        Балл схожести от 0.0 до 1.0.
    """
    if not user_ocean or len(user_ocean) != OCEAN_DIMS:
        return 0.0
    sq_sum = sum((u - a) ** 2 for u, a in zip(user_ocean, archetype_ocean))
    euclidean = math.sqrt(sq_sum)
    max_dist = math.sqrt(OCEAN_DIMS)
    return max(0.0, 1.0 - euclidean / max_dist)


def compute_schwartz_similarity(
    user_values: dict[str, float], archetype_values: dict[str, float],
) -> float:
    """Вычисляет косинусную близость ценностного профиля пользователя к архетипу.

    Args:
        user_values: Словарь ценностей пользователя по Шварцу.
        archetype_values: Эталонный словарь ценностей архетипа.

    Returns:
        Балл схожести от 0.0 до 1.0.
    """
    keys = list(archetype_values.keys())
    u_vec = [user_values.get(k, 0.5) for k in keys]
    a_vec = [archetype_values.get(k, 0.5) for k in keys]

    dot = sum(u * a for u, a in zip(u_vec, a_vec))
    u_norm = math.sqrt(sum(v * v for v in u_vec))
    a_norm = math.sqrt(sum(v * v for v in a_vec))

    if u_norm == 0 or a_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (u_norm * a_norm)))


def compute_root_personality_score(
    personality_profile: UserPersonalityProfile | None,
    schwartz_profile: UserSchwartzProfile | None,
    root_category: str,
) -> float:
    """Вычисляет итоговый балл соответствия личности пользователя корневому архетипу.

    Комбинирует OCEAN-схожесть (60%) и ценностную схожесть по Шварцу (40%).

    Args:
        personality_profile: Профиль OCEAN пользователя.
        schwartz_profile: Профиль ценностей Шварца пользователя.
        root_category: Корневая категория (work/entertainment/life).

    Returns:
        Балл от 0.0 до 1.0.
    """
    archetype = ROOT_ARCHETYPES.get(root_category)
    if not archetype:
        return 0.0

    ocean_sim = 0.0
    if personality_profile:
        user_ocean = [
            personality_profile.openness or 0.5,
            personality_profile.conscientiousness or 0.5,
            personality_profile.extraversion or 0.5,
            personality_profile.agreeableness or 0.5,
            personality_profile.neuroticism or 0.5,
        ]
        ocean_sim = compute_ocean_similarity(user_ocean, archetype["ocean"])

    schwartz_sim = 0.0
    if schwartz_profile and schwartz_profile.is_populated():
        user_values = {
            k: getattr(schwartz_profile, k, 0.5) or 0.5
            for k in UserSchwartzProfile.SCHWARTZ_KEYS
        }
        schwartz_sim = compute_schwartz_similarity(
            user_values, archetype["schwartz"],
        )

    if schwartz_sim > 0:
        return OCEAN_WEIGHT * ocean_sim + SCHWARTZ_WEIGHT * schwartz_sim
    return ocean_sim
