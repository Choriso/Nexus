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
)
from app.ai_profiler.schwartz_analyzer import schwartz_cosine_similarity
from data.ai import UserSchwartzProfile
from data.behavior import UserBehaviorProfile

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

    # ЗАМЕНЕНО: print на logger.debug для продакшен-безопасности
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
