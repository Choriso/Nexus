"""Тестер поиска пользователей по ноде интереса.

Имитирует: нажали на ноду -> получили список пользователей с % совпадения.

Замеряет 3 метрики:
  - graph_interest -- пересечение по графу интересов (прямое/косвенное)
  - ocean_similarity -- семантическая близость OCEAN-профилей (pgvector)
  - jaccard -- пересечение множеств узлов графа

Финальный скор -- взвешенная сумма трёх метрик (как в search_ranking.py).
"""
import os
import sys
import json
import math
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from data.session import global_init, create_session
from data.user import User
from data.ai import UserPersonalityProfile, GlobalWeightsConfig
from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight
from app.ai_profiler.interest_graph import (
    compute_hierarchical_overlap_score,
    compute_jaccard_interest_similarity,
)


def _build_hierarchy_cache(db):
    nodes = db.query(InterestHierarchyNode).all()
    return {
        n.id: {
            "slug": n.slug,
            "name": n.name,
            "depth": n.depth or 0,
            "parent_id": n.parent_id,
            "match_weight": n.match_weight or 1.0,
        }
        for n in nodes
    }


def _get_related_node_ids(db, node_id):
    target = db.query(InterestHierarchyNode).filter_by(id=node_id).first()
    if not target:
        return {node_id}
    ids = {node_id}
    parent = target
    while parent.parent_id:
        ids.add(parent.parent_id)
        parent = db.query(InterestHierarchyNode).filter_by(id=parent.parent_id).first()
        if not parent:
            break
    children = db.query(InterestHierarchyNode).filter_by(parent_id=node_id).all()
    ids.update(c.id for c in children)
    return ids


def _graph_interest_score(target_node, matched_weights, hierarchy_cache):
    if not matched_weights:
        return 0.0, []
    score = 0.0
    max_possible = 0.0
    tags = []
    for matched_nid, weight in matched_weights.items():
        data = hierarchy_cache.get(matched_nid)
        if not data:
            continue
        if matched_nid == target_node.id:
            coeff = 1.0
            tags.append(f"{data['slug']} (точное)")
        else:
            depth_diff = abs(data["depth"] - target_node.depth)
            coeff = max(0.4 - 0.05 * depth_diff, 0.1)
            tags.append(f"{data['slug']} (похоже)")
        score += weight * coeff
        max_possible += coeff
    normalized = score / max_possible if max_possible > 0 else 0.0
    return min(normalized, 1.0), tags


def _ocean_similarity(db, user_id_1, user_id_2):
    p1 = db.query(UserPersonalityProfile).filter_by(user_id=user_id_1).first()
    p2 = db.query(UserPersonalityProfile).filter_by(user_id=user_id_2).first()
    if p1 is None or p2 is None or p1.embedding is None or p2.embedding is None:
        return 0.0
    try:
        row = db.query(
            p1.embedding.cosine_distance(p2.embedding).label("distance")
        ).first()
        if row is None:
            return 0.0
        return max(0.0, 1.0 - float(row.distance) / 2.0)
    except Exception:
        return 0.0


def _ocean_similarity_manual(db, user_id, avg_ocean_vector):
    """Compare a user's OCEAN profile against a raw avg vector."""
    p = db.query(UserPersonalityProfile).filter_by(user_id=user_id).first()
    if p is None or p.embedding is None:
        return 0.0
    user_vec = p.get_big_five_vector()
    dot = sum(a * b for a, b in zip(user_vec, avg_ocean_vector))
    norm_u = sum(a * a for a in user_vec) ** 0.5
    norm_a = sum(b * b for b in avg_ocean_vector) ** 0.5
    if norm_u == 0 or norm_a == 0:
        return 0.0
    cosine = dot / (norm_u * norm_a)
    return max(0.0, (cosine + 1.0) / 2.0)


def _global_weights(db):
    try:
        gwc = GlobalWeightsConfig.get_or_create(db)
        return {
            "ocean": gwc.weight_ocean,
            "graph": gwc.weight_graph,
            "jaccard": gwc.weight_jaccard,
        }
    except Exception:
        return {"ocean": 0.35, "graph": 0.40, "jaccard": 0.25}


def list_nodes(db):
    nodes = db.query(InterestHierarchyNode).order_by(InterestHierarchyNode.depth, InterestHierarchyNode.slug).all()
    print(f"\n{'ID':>4} {'Slug':40s} {'Name':30s} {'Depth':>5} {'Category':20s}")
    print("-" * 105)
    for n in nodes:
        indent = "  " * (n.depth or 0)
        print(f"{n.id:>4} {n.slug:40s} {indent}{n.name:30s} {n.depth:>5} {(n.global_category or ''):20s}")
    print(f"\nВсего нод: {len(nodes)}")


def match_by_node(db, node_id, top_k=20, user_id_filter=None):
    target = db.query(InterestHierarchyNode).filter_by(id=node_id).first()
    if not target:
        print(f"[ERROR] Нода id={node_id} не найдена")
        return

    print(f"\n{'='*80}")
    print(f"Поиск по ноде: {target.name} (slug={target.slug}, id={target.id})")
    print(f"{'='*80}\n")

    related_ids = _get_related_node_ids(db, target.id)
    hierarchy_cache = _build_hierarchy_cache(db)
    g_weights = _global_weights(db)

    candidates_raw = (
        db.query(
            UserInterestGraphWeight.user_id,
            UserInterestGraphWeight.node_id,
            UserInterestGraphWeight.weight,
            User.name,
        )
        .join(User, UserInterestGraphWeight.user_id == User.id)
        .filter(
            UserInterestGraphWeight.node_id.in_(related_ids),
            UserInterestGraphWeight.weight > 0.0,
        )
        .all()
    )

    if not candidates_raw:
        print("Нет пользователей с пересекающимися интересами")
        return

    candidates = {}
    for uid, nid, w, name in candidates_raw:
        if uid not in candidates:
            candidates[uid] = {"name": name, "matched_weights": {}}
        candidates[uid]["matched_weights"][nid] = w

    results = []
    for uid, cdata in candidates.items():
        g_score, matched_tags = _graph_interest_score(target, cdata["matched_weights"], hierarchy_cache)
        if not matched_tags or g_score < 0.1:
            continue
        results.append({
            "user_id": uid,
            "user_name": cdata["name"],
            "graph_score": round(g_score, 4),
            "matched_tags": matched_tags,
            "tag_count": len(matched_tags),
        })

    results.sort(key=lambda x: x["graph_score"], reverse=True)

    # ─── Таблица 1: Graph Score ─────────────────────────────────
    print(f"{'#':>3} {'User ID':>8} {'Имя':30s} {'Graph':>7} {'Теги':50s}")
    print("-" * 105)
    for i, r in enumerate(results[:top_k], 1):
        tags_short = ", ".join(r["matched_tags"][:3])
        if len(r["matched_tags"]) > 3:
            tags_short += "..."
        print(f"{i:>3} {r['user_id']:>8} {r['user_name']:30s} {r['graph_score']:>7.2%} {tags_short:50s}")

    # ─── Таблица 2: 3-метричный скор (vs среднестатистический пользователь этой ноды) ─
    # Найти всех пользователей с direct match на target
    direct_user_ids = set()
    for uid, cdata in candidates.items():
        if target.id in cdata["matched_weights"]:
            direct_user_ids.add(uid)

    if direct_user_ids:
        # Вычислить средний OCEAN-вектор direct-match пользователей
        avg_ocean = [0.0, 0.0, 0.0, 0.0, 0.0]
        count = 0
        for uid in direct_user_ids:
            p = db.query(UserPersonalityProfile).filter_by(user_id=uid).first()
            if p and p.embedding is not None:
                vec = p.get_big_five_vector()
                for j in range(5):
                    avg_ocean[j] += vec[j]
                count += 1
        if count > 0:
            for j in range(5):
                avg_ocean[j] /= count

            # Создать synthetic user_id=-1 с этим средним профилем для сравнения
            # Вычислить 3-метричный скор каждого кандидата vs этот средний профиль
            ref_user_id = -1
            from sqlalchemy import text
            full_results = []
            for r in results:
                uid = r["user_id"]
                ocean = _ocean_similarity_manual(db, uid, avg_ocean)
                graph_ov = compute_hierarchical_overlap_score(db, ref_user_id, uid)
                jaccard = compute_jaccard_interest_similarity(db, ref_user_id, uid)

                w_o = g_weights["ocean"]
                w_g = g_weights["graph"]
                w_j = g_weights["jaccard"]
                total_w = w_o + w_g + w_j
                if total_w > 0:
                    w_o /= total_w
                    w_g /= total_w
                    w_j /= total_w

                final = w_o * ocean + w_g * graph_ov + w_j * jaccard
                full_results.append({
                    "user_id": uid,
                    "user_name": r["user_name"],
                    "final": round(final, 4),
                    "ocean": round(ocean, 4),
                    "graph_ov": round(graph_ov, 4),
                    "jaccard": round(jaccard, 4),
                })

            full_results.sort(key=lambda x: x["final"], reverse=True)

            print(f"\n--- OCEAN-совместимость (vs средний OCEAN по ноде) ---")
            print(f"{'#':>3} {'User ID':>8} {'Имя':25s} {'OCEAN':>7}")
            print("-" * 50)
            for i, r in enumerate(full_results[:top_k], 1):
                print(f"{i:>3} {r['user_id']:>8} {r['user_name']:25s} {r['ocean']:>7.2%}")

    # ─── Таблица 3: 3-метричный скор vs конкретный пользователь (если -u) ─
    if user_id_filter is not None:
        ruid = user_id_filter
        print(f"\n--- 3-метричный скор совместимости с user_id={ruid} ---")
        full_results = []
        for r in results:
            uid = r["user_id"]
            if uid == ruid:
                continue
            ocean = _ocean_similarity(db, ruid, uid)
            graph_ov = compute_hierarchical_overlap_score(db, ruid, uid)
            jaccard = compute_jaccard_interest_similarity(db, ruid, uid)

            w_o = g_weights["ocean"]
            w_g = g_weights["graph"]
            w_j = g_weights["jaccard"]
            total_w = w_o + w_g + w_j
            if total_w > 0:
                w_o /= total_w
                w_g /= total_w
                w_j /= total_w

            final = w_o * ocean + w_g * graph_ov + w_j * jaccard

            full_results.append({
                "user_id": uid,
                "user_name": r["user_name"],
                "final": round(final, 4),
                "ocean": round(ocean, 4),
                "graph_ov": round(graph_ov, 4),
                "jaccard": round(jaccard, 4),
            })

        full_results.sort(key=lambda x: x["final"], reverse=True)

        print(f"\n{'#':>3} {'User ID':>8} {'Имя':25s} {'Финальный':>10} {'OCEAN':>7} {'GraphOv':>8} {'Jaccard':>8}")
        print("-" * 75)
        for i, r in enumerate(full_results[:top_k], 1):
            print(f"{i:>3} {r['user_id']:>8} {r['user_name']:25s} {r['final']:>10.2%} {r['ocean']:>7.2%} {r['graph_ov']:>8.2%} {r['jaccard']:>8.2%}")

        if full_results:
            print(f"\nGlobal weights: ocean={g_weights['ocean']:.3f}, graph={g_weights['graph']:.3f}, jaccard={g_weights['jaccard']:.3f}")

    return results


def match_by_slug(db, slug, **kwargs):
    node = db.query(InterestHierarchyNode).filter_by(slug=slug).first()
    if not node:
        print(f"[ERROR] Нода со слагом '{slug}' не найдена")
        return
    return match_by_node(db, node.id, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Тестер поиска пользователей по ноде интереса")
    parser.add_argument("node_id", nargs="?", type=int, default=None, help="ID ноды для поиска")
    parser.add_argument("-s", "--slug", type=str, default=None, help="Slug ноды для поиска")
    parser.add_argument("-n", "--top", type=int, default=20, help="Количество результатов (default: 20)")
    parser.add_argument("-u", "--user", type=int, default=None, help="ID пользователя для 3-метричного скора")
    parser.add_argument("-l", "--list", action="store_true", help="Показать все доступные ноды")
    args = parser.parse_args()

    global_init(config.DATABASE_URL)
    db = create_session()

    try:
        if args.list:
            list_nodes(db)
            return

        if args.slug:
            match_by_slug(db, args.slug, top_k=args.top, user_id_filter=args.user)
        elif args.node_id is not None:
            match_by_node(db, args.node_id, top_k=args.top, user_id_filter=args.user)
        else:
            list_nodes(db)
            print("\nУкажите --slug или node_id для поиска. Примеры:")
            print("  python tools/test_node_match.py -l")
            print("  python tools/test_node_match.py --slug backend_python -n 10")
            print("  python tools/test_node_match.py 5 -n 15 -u 1")
    finally:
        db.close()


if __name__ == "__main__":
    main()
