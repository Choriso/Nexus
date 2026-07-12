from __future__ import annotations

import logging

from sqlalchemy.orm import Session
from sqlalchemy import or_

from config import config
from data.ai import GlobalWeightsConfig, UserPersonalityProfile
from data.user import User
from app.ai_profiler.interest_graph import (
    compute_hierarchical_overlap_score,
    compute_jaccard_interest_similarity,
)

logger = logging.getLogger(__name__)

METRIC_KEYS = ["ocean", "graph", "jaccard"]

MAX_OFFSET_SUM = 1.0


def _get_global_weights(db: Session) -> dict[str, float]:
    """Загружает глобальные веса метрик совместимости из БД или конфига.

    Args:
        db: Сессия SQLAlchemy.

    Returns:
        Словарь с весами для ocean, graph, jaccard.
    """
    try:
        gwc = GlobalWeightsConfig.get_or_create(db)
        return {
            "ocean": gwc.weight_ocean,
            "graph": gwc.weight_graph,
            "jaccard": gwc.weight_jaccard,
        }
    except Exception as e:
        logger.warning(f"[global_weights] Could not load from DB, using config defaults: {e}")
        return {
            "ocean": config.RANKING_WEIGHT_OCEAN,
            "graph": config.RANKING_WEIGHT_GRAPH,
            "jaccard": config.RANKING_WEIGHT_JACCARD,
        }


def _get_personal_weights(db: Session, user_id: int) -> dict[str, float]:
    """Загружает персональные смещения весов метрик для пользователя.

    Args:
        db: Сессия SQLAlchemy.
        user_id: ID пользователя.

    Returns:
        Словарь с нулевыми смещениями или значениями из БД.
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            return {k: 0.0 for k in METRIC_KEYS}
        return {
            "ocean": user.metric_weight_ocean_offset or 0.0,
            "graph": user.metric_weight_graph_offset or 0.0,
            "jaccard": user.metric_weight_jaccard_offset or 0.0,
        }
    except Exception as e:
        logger.warning(f"[personal_weights] Error for user {user_id}: {e}")
        return {k: 0.0 for k in METRIC_KEYS}


def _compute_semantic_similarity(
    db: Session, user_id_1: int, user_id_2: int,
) -> float:
    """Вычисляет семантическую близость между пользователями через OCEAN-эмбеддинги.

    Args:
        db: Сессия SQLAlchemy.
        user_id_1: ID первого пользователя.
        user_id_2: ID второго пользователя.

    Returns:
        Балл схожести от 0.0 до 1.0.
    """
    try:
        p1 = db.query(UserPersonalityProfile).filter_by(user_id=user_id_1).first()
        p2 = db.query(UserPersonalityProfile).filter_by(user_id=user_id_2).first()
        if p1 is None or p2 is None or p1.embedding is None or p2.embedding is None:
            return 0.0
        row = db.query(
            p1.embedding.cosine_distance(p2.embedding).label("distance")
        ).first()
        if row is None:
            return 0.0
        distance_value = float(row.distance)
        return max(0.0, 1.0 - distance_value / 2.0)
    except Exception as e:
        logger.warning(f"[semantic] Error comparing {user_id_1} / {user_id_2}: {e}")
        return 0.0


def compute_user_compatibility_score(
    db: Session,
    user_id_1: int,
    user_id_2: int,
) -> dict:
    """Вычисляет итоговый балл совместимости двух пользователей.

    Комбинирует три метрики: семантическую (OCEAN), иерархическую (graph),
    и Jaccard-схожесть интересов с учётом глобальных и персональных весов.

    Args:
        db: Сессия SQLAlchemy.
        user_id_1: ID первого пользователя.
        user_id_2: ID второго пользователя.

    Returns:
        Словарь с итоговым баллом, разбивкой метрик и эффективными весами.
    """
    global_weights = _get_global_weights(db)
    personal_offsets = _get_personal_weights(db, user_id_1)

    metric_1 = _compute_semantic_similarity(db, user_id_1, user_id_2)
    metric_2 = compute_hierarchical_overlap_score(db, user_id_1, user_id_2)
    metric_3 = compute_jaccard_interest_similarity(db, user_id_1, user_id_2)

    w_ocean = max(0.0, global_weights.get("ocean", 0.35) + personal_offsets.get("ocean", 0.0))
    w_graph = max(0.0, global_weights.get("graph", 0.40) + personal_offsets.get("graph", 0.0))
    w_jaccard = max(0.0, global_weights.get("jaccard", 0.25) + personal_offsets.get("jaccard", 0.0))

    total_w = w_ocean + w_graph + w_jaccard
    if total_w > 0.0:
        w_ocean /= total_w
        w_graph /= total_w
        w_jaccard /= total_w

    final_score = w_ocean * metric_1 + w_graph * metric_2 + w_jaccard * metric_3

    return {
        "final_score": round(float(final_score), 4),
        "metrics": {
            "ocean": round(float(metric_1), 4),
            "graph": round(float(metric_2), 4),
            "jaccard": round(float(metric_3), 4),
        },
        "effective_weights": {
            "ocean": round(float(w_ocean), 4),
            "graph": round(float(w_graph), 4),
            "jaccard": round(float(w_jaccard), 4),
        },
    }


def micro_gradient_step(
    db: Session,
    user_id: int,
    successful_metric: str,
    learning_rate: float | None = None,
) -> None:
    """Применяет микро-градиентный шаг: увеличивает вес успешной метрики.

    При успешном взаимодействии по метрике её вес увеличивается,
    а веса остальных метрик уменьшаются для персонализации ранжирования.

    Args:
        db: Сессия SQLAlchemy.
        user_id: ID пользователя.
        successful_metric: Название метрики, показавшей успех.
        learning_rate: Скорость обучения (шаг изменения весов).
    """
    if learning_rate is None:
        learning_rate = config.MICRO_GRADIENT_LEARNING_RATE

    if successful_metric not in METRIC_KEYS:
        logger.warning(f"[micro_gradient] Unknown metric '{successful_metric}', ignoring")
        return

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        logger.warning(f"[micro_gradient] User {user_id} not found")
        return

    offsets = {
        "ocean": user.metric_weight_ocean_offset or 0.0,
        "graph": user.metric_weight_graph_offset or 0.0,
        "jaccard": user.metric_weight_jaccard_offset or 0.0,
    }

    remaining = [k for k in METRIC_KEYS if k != successful_metric]
    penalty = learning_rate / len(remaining) if remaining else 0.0

    offsets[successful_metric] = offsets[successful_metric] + learning_rate
    for key in remaining:
        offsets[key] = offsets[key] - penalty

    for key in METRIC_KEYS:
        setattr(user, f"metric_weight_{key}_offset", round(offsets[key], 6))

    db.commit()
    logger.info(
        "[micro_gradient] User %s: +%.3f on '%s', new offsets: %s",
        user_id, learning_rate, successful_metric,
        {k: round(offsets[k], 4) for k in METRIC_KEYS},
    )


def aggregate_global_trend(db: Session) -> dict[str, float]:
    """Агрегирует персональные смещения всех активных пользователей в глобальные веса.

    Если количество активных пользователей превышает порог, обновляет
    глобальные веса метрик как среднее персональных смещений.

    Args:
        db: Сессия SQLAlchemy.

    Returns:
        Словарь со статусом обновления и новыми глобальными весами.
    """
    min_active = config.MIN_ACTIVE_USERS_FOR_GLOBAL_ADJUSTMENT
    learning_rate = config.GLOBAL_LEARNING_RATE

    active_users = (
        db.query(User)
        .filter(
            or_(
                User.metric_weight_ocean_offset != 0.0,
                User.metric_weight_graph_offset != 0.0,
                User.metric_weight_jaccard_offset != 0.0,
            )
        )
        .all()
    )

    if len(active_users) < min_active:
        logger.info(
            "[global_trend] Only %d active users (need %d), skipping",
            len(active_users), min_active,
        )
        return {"status": "skipped", "active_users": len(active_users)}

    avg_offsets = {k: 0.0 for k in METRIC_KEYS}
    for user in active_users:
        avg_offsets["ocean"] += user.metric_weight_ocean_offset or 0.0
        avg_offsets["graph"] += user.metric_weight_graph_offset or 0.0
        avg_offsets["jaccard"] += user.metric_weight_jaccard_offset or 0.0

    n = len(active_users)
    for key in METRIC_KEYS:
        avg_offsets[key] /= n

    gwc = GlobalWeightsConfig.get_or_create(db)
    gwc.weight_ocean = max(0.05, gwc.weight_ocean + avg_offsets["ocean"] * learning_rate)
    gwc.weight_graph = max(0.05, gwc.weight_graph + avg_offsets["graph"] * learning_rate)
    gwc.weight_jaccard = max(0.05, gwc.weight_jaccard + avg_offsets["jaccard"] * learning_rate)

    total_w = gwc.weight_ocean + gwc.weight_graph + gwc.weight_jaccard
    if total_w > 0.0:
        gwc.weight_ocean /= total_w
        gwc.weight_graph /= total_w
        gwc.weight_jaccard /= total_w

    db.commit()
    logger.info(
        "[global_trend] Updated global weights from %d users: ocean=%.4f graph=%.4f jaccard=%.4f",
        n, gwc.weight_ocean, gwc.weight_graph, gwc.weight_jaccard,
    )

    return {
        "status": "updated",
        "active_users": n,
        "avg_offsets": {k: round(v, 6) for k, v in avg_offsets.items()},
        "new_global_weights": gwc.to_dict(),
    }
