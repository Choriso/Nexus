# diagnose_matching.py
"""
Глубокая диагностика поиска интересов.

Для каждого запроса:
- Резолвинг в слаг
- Построение весов запроса
- Поиск кандидатов с общими узлами
- Разбор прямых/косвенных совпадений
- Итоговый многомерный скор с разбивкой по компонентам (graph, big_five, …)
Запуск: python diagnose_matching.py
"""
from __future__ import annotations

import logging
from typing import Any

from data.session import global_init, create_session
from config import config
from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight
from app.ai_profiler.interest_graph import (
    resolve_tags_batch,
    build_query_weights,
    get_user_graph_weights,
    calculate_graph_interest_score,
)
from app.ai.matching_engine import calculate_multidimensional_compatibility

# Чтобы видеть подробные логи, можно включить debug, но здесь мы делаем вывод сами
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("diagnose")

# ID пользователей для проверки (должны существовать в БД)
# Возьмём одного с IT‑интересами и одного пустого.
TEST_USER_WITH_INTERESTS = 1035   # у него есть backend_python и т.д.
TEST_USER_EMPTY = 1036            # или любой другой без весов, можно оставить как есть

QUERIES = [
    "программирование",
    "написание кода",
    "python",
    "3d моделирование",
    "музыка",
    "спорт",
]

def main():
    global_init(config.DATABASE_URL)
    db = create_session()

    # 1. Проверим веса у тестовых пользователей
    for uid in (TEST_USER_WITH_INTERESTS, TEST_USER_EMPTY):
        weights = get_user_graph_weights(db, uid)
        print(f"User {uid}: {len(weights)} весов в графе")
        if weights:
            sample = dict(list(weights.items())[:5])
            nodes = db.query(InterestHierarchyNode).filter(InterestHierarchyNode.id.in_(sample.keys())).all()
            node_names = {n.id: n.name for n in nodes}
            for nid, w in sample.items():
                print(f"  - {node_names.get(nid, nid)}: {w:.3f}")
        else:
            print("  (нет весов)")

    print("\n" + "=" * 70)
    print("ПРОГОН ЗАПРОСОВ")
    print("=" * 70)

    for query in QUERIES:
        print(f"\n--- Запрос: {query!r} ---")

        # Шаг 1: резолвинг
        resolved = resolve_tags_batch(db, [query])
        slug = resolved.get(query)
        print(f"Резолвинг: {query!r} → {slug}")

        # Шаг 2: веса запроса
        q_weights = build_query_weights(db, {query})
        print(f"Query weights: {len(q_weights)} узлов")
        for nid, w in list(q_weights.items())[:5]:
            node = db.query(InterestHierarchyNode).get(nid)
            name = node.name if node else str(nid)
            print(f"  {name} ({nid}): {w:.3f}")

        # Шаг 3: анализ для каждого тестового пользователя
        for uid in (TEST_USER_WITH_INTERESTS, TEST_USER_EMPTY):
            print(f"\n  Анализ для пользователя {uid}:")
            o_weights = get_user_graph_weights(db, uid)
            shared = set(q_weights) & set(o_weights)
            print(f"  Общих узлов: {len(shared)}")

            if shared:
                nodes = {n.id: n for n in db.query(InterestHierarchyNode).filter(InterestHierarchyNode.id.in_(shared)).all()}
                requested_slugs = {slug} if slug else set()
                for nid in shared:
                    node = nodes.get(nid)
                    if node:
                        match_type = "DIRECT" if node.slug in requested_slugs else "INDIRECT"
                        print(f"    {match_type}: {node.name} (slug={node.slug}, q_weight={q_weights[nid]:.3f}, u_weight={o_weights[nid]:.3f})")

            # Шаг 4: многомерный скор (используем оставшиеся компоненты по умолчанию 0.5)
            # Чтобы оценить вклад графа, посчитаем graph_score отдельно через calculate_graph_interest_score
            hierarchy_nodes = db.query(InterestHierarchyNode).all()
            hierarchy_node_names = {n.id: n.name for n in hierarchy_nodes}

            graph_score, matched_tags = calculate_graph_interest_score(
                db,
                query_tags={query},
                other_user_id=uid,
                other_extracted=None,  # не используется, т.к. other_weights уже получены
                query_weights=q_weights,
                other_weights=o_weights,
                hierarchy_node_names=hierarchy_node_names,
            )
            print(f"  Graph score: {graph_score:.4f}")
            if matched_tags:
                print(f"  Matched tags: {matched_tags}")

            # Теперь полный многомерный скор (big_five=0.5, schwartz/behavioral=None)
            try:
                result = calculate_multidimensional_compatibility(
                    db,
                    ocean_score_normalized=0.5,
                    query_tags={query},
                    current_user_id=1,          # не важно, главное чтобы не совпадал с other_user_id
                    other_user_id=uid,
                    other_extracted=None,
                    my_schwartz=None,
                    other_schwartz=None,
                    my_behavior=None,
                    other_behavior=None,
                    query_graph_weights=q_weights,
                    other_graph_weights=o_weights,
                    hierarchy_node_names=hierarchy_node_names,
                )
                print(f"  Final score: {result['final_score']:.4f}")
                print(f"  Breakdown: big_five={result.get('big_five_score', '?')}, "
                      f"graph_interest={result.get('graph_interest_score', '?')}, "
                      f"schwartz={result.get('schwartz_score', '?')}, "
                      f"behavioral={result.get('behavioral_score', '?')}")
                if result.get('cold_start', {}).get('schwartz', True):
                    print("  (schwartz/behavioral в cold start)")
            except Exception as e:
                logger.error(f"Multidimensional compatibility failed: {e}")

    db.close()
    print("\nГотово.")


if __name__ == "__main__":
    main()