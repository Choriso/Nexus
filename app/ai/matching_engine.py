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
