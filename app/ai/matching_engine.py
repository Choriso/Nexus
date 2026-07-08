"""
Многомерный движок совместимости Nexus.

Compatibility = 0.50 * BigFive + 0.25 * GraphInterest + 0.15 * Schwartz + 0.10 * Behavioral

При cold-start веса динамически перераспределяются между доступными компонентами.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.ai_profiler.behavior_analyzer import calculate_behavioral_score
from app.ai_profiler.interest_graph import (
    build_query_weights,
    calculate_graph_interest_score,
    get_user_graph_weights,
    register_user_tags,
    resolve_tags_batch,
)
from app.ai_profiler.schwartz_analyzer import schwartz_cosine_similarity
from data.ai import UserSchwartzProfile
from data.behavior import UserBehaviorProfile
from data.interest_hierarchy import InterestHierarchyNode

BASE_WEIGHTS = {
    "big_five": 0.5,
    "graph_interest": 0.25,
    "schwartz": 0.15,
    "behavioral": 0.10,
}


def _effective_weights(
    has_schwartz: bool,
    has_behavior: bool,
) -> dict[str, float]:
    weights = dict(BASE_WEIGHTS)
    missing_keys: list[str] = []
    if not has_schwartz:
        missing_keys.append("schwartz")
    if not has_behavior:
        missing_keys.append("behavioral")

    if not missing_keys:
        return weights

    redistributed = sum(weights[k] for k in missing_keys)
    for key in missing_keys:
        weights[key] = 0.0

    remaining = [k for k in weights if k not in missing_keys]
    remaining_sum = sum(weights[k] for k in remaining) or 1.0
    for key in remaining:
        weights[key] += redistributed * (weights[key] / remaining_sum)

    return weights


def calculate_big_five_score(
    ocean_score_normalized: float,
) -> float:
    return max(0.0, min(1.0, ocean_score_normalized))


import logging

logger = logging.getLogger(__name__)


def _calculate_match_type_score(
    db: Session,
    query_tags: set[str],
    matched_tags: list[str],
    hierarchy_node_names: dict[int, str] | None = None,
) -> float:
    """
    ✅ ДОБАВЛЕНО: Вычисляет скор интереса с различением прямого и косвенного совпадения.
    
    - Прямое совпадение (slug точно совпадает с запросом): вес 1.0
    - Косвенное совпадение (через родителей/потомков): вес 0.6
    - Нет совпадения: 0.0
    
    Args:
        db: SQLAlchemy сессия
        query_tags: Запрошенные теги
        matched_tags: Список совпавших названий узлов из calculate_graph_interest_score
        hierarchy_node_names: Маппинг ID узла → имя (используется для обратного преобразования)
    
    Returns:
        float: Скор в диапазоне [0, 1]
    """
    if not matched_tags or not query_tags:
        return 0.0
    
    # Резолвим запрошенные теги в слаги
    _query_resolved = resolve_tags_batch(db, list(query_tags))
    requested_slugs = {s for s in _query_resolved.values() if s}
    
    if not requested_slugs:
        return 0.0
    
    direct_match_count = 0
    indirect_match_count = 0
    
    # Загружаем иерархию
    all_nodes = db.query(InterestHierarchyNode).all()
    node_id_to_node = {n.id: n for n in all_nodes}
    
    # Обратный маппинг: имя узла → ID
    name_to_node_id = {}
    if hierarchy_node_names:
        for nid, name in hierarchy_node_names.items():
            name_to_node_id[name] = nid
    
    for matched_name in matched_tags:
        nid = name_to_node_id.get(matched_name)
        if nid is None:
            continue
        
        node = node_id_to_node.get(nid)
        if not node:
            continue
        
        # Проверяем, прямое ли это совпадение
        if node.slug in requested_slugs:
            direct_match_count += 1
        else:
            # Проверяем, косвенное ли (через иерархию)
            is_indirect = False
            current = node
            while current.parent_id:
                parent = node_id_to_node.get(current.parent_id)
                if not parent:
                    break
                if parent.slug in requested_slugs:
                    is_indirect = True
                    break
                current = parent
            
            if is_indirect:
                indirect_match_count += 1
    
    # Взвешенный скор: прямые совпадения сильнее
    match_score = (direct_match_count * 1.0 + indirect_match_count * 0.6) / max(len(matched_tags), 1)
    return min(1.0, max(0.0, match_score))


def calculate_multidimensional_compatibility(
        db: Session,
        *,
        ocean_score_normalized: float,
        query_tags: set[str],
        current_user_id: int,
        other_user_id: int,
        other_extracted: dict[str, Any] | None,
        my_schwartz: UserSchwartzProfile | None,
        other_schwartz: UserSchwartzProfile | None,
        my_behavior: UserBehaviorProfile | None,
        other_behavior: UserBehaviorProfile | None,
        query_graph_weights: dict[int, float] | None = None,
        other_graph_weights: dict[int, float] | None = None,
        hierarchy_node_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Возвращает итоговый score и разбивку по компонентам.
    Read-only операция, не пишет в БД.
    
    ✅ ИСПРАВЛЕНО: Теперь различает прямое и косвенное совпадение интересов.
    """
    # УБРАНО: register_user_tags, так как это read-only эндпоинт

    graph_score, matched_tags = calculate_graph_interest_score(
        db,
        query_tags,
        other_user_id,
        other_extracted,
        query_weights=query_graph_weights,
        other_weights=other_graph_weights,
        hierarchy_node_names=hierarchy_node_names,
    )

    # ✅ НОВОЕ: Переанализируем скор с учетом типа совпадения
    if matched_tags and hierarchy_node_names:
        match_type_score = _calculate_match_type_score(db, query_tags, matched_tags, hierarchy_node_names)
        logger.debug(
            "Диагностика кандидата %s | tags: %s | graph_score: %s | match_type_score: %s | matched: %s",
            other_user_id, query_tags, graph_score, match_type_score, matched_tags
        )
        # Используем более строгий скор match_type_score если есть совпадения
        graph_score = match_type_score if match_type_score > 0 else graph_score
    else:
        logger.debug(
            "Диагностика кандидата %s | tags: %s | score: %s | matched: %s",
            other_user_id, query_tags, graph_score, matched_tags
        )

    schwartz_score = schwartz_cosine_similarity(my_schwartz, other_schwartz)
    behavioral_score = calculate_behavioral_score(my_behavior, other_behavior)

    has_schwartz = schwartz_score is not None
    has_behavior = behavioral_score is not None
    weights = _effective_weights(has_schwartz, has_behavior)

    big_five = calculate_big_five_score(ocean_score_normalized)
    schwartz_val = schwartz_score if has_schwartz else 0.0
    behavior_val = behavioral_score if has_behavior else 0.0

    final_score = (
            weights["big_five"] * big_five
            + weights["graph_interest"] * graph_score
            + weights["schwartz"] * schwartz_val
            + weights["behavioral"] * behavior_val
    )

    return {
        "final_score": round(float(final_score), 4),
        "big_five_score": round(big_five, 4),
        "graph_interest_score": round(graph_score, 4),
        "schwartz_score": round(schwartz_val, 4) if has_schwartz else None,
        "behavioral_score": round(behavior_val, 4) if has_behavior else None,
        "weights_applied": {k: round(v, 4) for k, v in weights.items()},
        "matched_tags": matched_tags,
        "cold_start": {
            "schwartz": not has_schwartz,
            "behavioral": not has_behavior,
        },
    }
