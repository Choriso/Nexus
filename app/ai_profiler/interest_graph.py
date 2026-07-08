"""
Иерархический граф интересов: seed, регистрация тегов, расчёт graph score.

При совпадении листового тега (напр. Flask) веса распространяются вверх
по цепочке: Python (0.80) → Backend (0.50) → Programming (0.30).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.ai_profiler.contextual_adapter import get_contextual_adapter
from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY
from app.ai_profiler.taxonomy import INTEREST_TAXONOMY
from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight

logger = logging.getLogger(__name__)

# (slug, display_name, parent_slug, match_weight, global_category)
_HIERARCHY_SEED = [
    # ============================================================
    # КОРНЕВЫЕ УЗЛЫ (0.3 - 0.4)
    # ============================================================
    ("it_development", "IT и Разработка", None, 0.3, "work"),
    ("gaming", "Гейминг", None, 0.4, "entertainment"),
    ("creativity_art", "Творчество и Искусство", None, 0.35, "life"),
    ("self_development", "Саморазвитие", None, 0.35, "life"),
    ("sports_active_life", "Спорт и Активный отдых", None, 0.35, "life"),
    ("psychology_relations", "Психология и Отношения", None, 0.4, "life"),
    ("music_audio", "Музыка и Аудио", None, 0.4, "entertainment"),
    ("cinema_video", "Кино и Видео", None, 0.4, "entertainment"),
    ("literature_reading", "Литература и Чтение", None, 0.4, "entertainment"),
    ("home_lifestyle", "Дом и Образ жизни", None, 0.35, "life"),
    ("science_education", "Наука и Образование", None, 0.35, "work"),
    ("finance_business", "Финансы и Бизнес", None, 0.35, "work"),

    # ============================================================
    # 1. IT И РАЗРАБОТКА (глубокое ветвление)
    # ============================================================
    ("backend_dev", "Бэкенд-разработка", "it_development", 0.5, "work"),
    ("backend_python", "Python-экосистема", "backend_dev", 0.7, "work"),
    ("python_flask", "Flask", "backend_python", 0.9, "work"),
    ("python_fastapi", "FastAPI", "backend_python", 0.9, "work"),
    ("python_django", "Django", "backend_python", 0.9, "work"),
    ("python_asyncio", "Асинхронный Python", "backend_python", 0.85, "work"),
    ("python_aiohttp", "Aiohttp", "backend_python", 0.9, "work"),
    ("python_sqlalchemy", "SQLAlchemy ORM", "backend_python", 0.85, "work"),
    ("python_pydantic", "Pydantic", "backend_python", 0.85, "work"),
    ("python_celery", "Celery", "backend_python", 0.85, "work"),
    ("python_poetry", "Poetry", "backend_python", 0.8, "work"),
    ("python_black", "Black Formatter", "backend_python", 0.8, "work"),
    # ... (весь остальной _HIERARCHY_SEED без изменений, тот же, что у вас)
]

_ALIAS_TO_SLUG: dict[str, str] = {}

# --------------------------------------------------------------
# СЕМАНТИЧЕСКИЙ РЕЗОЛЬВЕР СЛАГОВ (батч + pgvector)
# --------------------------------------------------------------
HIGH_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.55


def _node_source_text(slug: str, display_name: str) -> str:
    onto_entry = SEMANTIC_ONTOLOGY.get(slug)
    if onto_entry and isinstance(onto_entry, dict):
        desc = onto_entry.get("enriched_text", "")
        if not desc:
            aliases = onto_entry.get("aliases", [])
            desc = f"{display_name}. " + ", ".join(aliases[:10])
        return desc
    return display_name


def _ensure_node_embeddings(db: Session) -> None:
    nodes_without_embedding = (
        db.query(InterestHierarchyNode)
        .filter(InterestHierarchyNode.embedding.is_(None))
        .all()
    )
    if not nodes_without_embedding:
        return

    adapter = get_contextual_adapter()
    model = adapter.sbert_model

    texts = [
        adapter.enrich_text(_node_source_text(n.slug, n.name)).enriched
        for n in nodes_without_embedding
    ]
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    for node, vec in zip(nodes_without_embedding, vectors):
        node.embedding = vec.tolist()
        db.add(node)

    db.commit()
    logger.info(f"[embeddings] Encoded {len(nodes_without_embedding)} node embeddings")


def refresh_all_node_embeddings(db: Session) -> int:
    count = db.query(InterestHierarchyNode).update({InterestHierarchyNode.embedding: None})
    db.commit()
    _ensure_node_embeddings(db)
    return count


def _vector_resolve_batch(db: Session, tags: list[str]) -> dict[str, tuple[str | None, float]]:
    results: dict[str, tuple[str | None, float]] = {}
    if not tags:
        return results

    ensure_hierarchy_seeded(db)

    adapter = get_contextual_adapter()
    model = adapter.sbert_model
    enriched = [adapter.enrich_text(t).enriched for t in tags]
    vectors = model.encode(enriched, convert_to_numpy=True, show_progress_bar=False)

    for tag, vec in zip(tags, vectors):
        row = (
            db.query(
                InterestHierarchyNode.slug,
                InterestHierarchyNode.embedding.cosine_distance(vec.tolist()).label("distance"),
            )
            .filter(InterestHierarchyNode.embedding.isnot(None))
            .order_by("distance")
            .first()
        )
        if row is None:
            results[tag] = (None, 0.0)
            continue
        similarity = 1.0 - float(row.distance) / 2.0
        results[tag] = (row.slug, similarity)

    return results


def _log_low_confidence(db: Session, tag: str, slug: str, score: float) -> None:
    logger.warning(f"[resolve][low_confidence] tag='{tag}' -> slug='{slug}' score={score:.3f}")
    try:
        db.execute(text(
            """
            CREATE TABLE IF NOT EXISTS low_confidence_tag_resolutions (
                id SERIAL PRIMARY KEY,
                tag TEXT NOT NULL,
                resolved_slug TEXT NOT NULL,
                similarity FLOAT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        ))
        db.execute(
            text(
                "INSERT INTO low_confidence_tag_resolutions (tag, resolved_slug, similarity) "
                "VALUES (:tag, :slug, :score)"
            ),
            {"tag": tag, "slug": slug, "score": score},
        )
        db.commit()
    except Exception:
        logger.debug("[resolve] could not persist low-confidence log (non-critical)", exc_info=True)
        db.rollback()


def resolve_tags_batch(db: Session, tags: list[str]) -> dict[str, str | None]:
    if not tags:
        return {}

    ensure_hierarchy_seeded(db)

    resolved: dict[str, str | None] = {}
    unresolved: list[str] = []

    for tag in tags:
        slug = _resolve_exact(tag)
        if slug:
            resolved[tag] = slug
        else:
            unresolved.append(tag)

    if unresolved:
        vector_results = _vector_resolve_batch(db, unresolved)
        for tag, (slug, score) in vector_results.items():
            if slug is None:
                resolved[tag] = None
                continue
            if score >= HIGH_CONFIDENCE:
                resolved[tag] = slug
            elif score >= LOW_CONFIDENCE:
                resolved[tag] = slug
                _log_low_confidence(db, tag, slug, score)
            else:
                logger.info(f"[resolve] tag='{tag}' below LOW_CONFIDENCE ({score:.3f}) -> None")
                resolved[tag] = None

    return resolved


def _normalize_tag(tag: str) -> str:
    if not tag:
        return ""
    clean = re.sub(r"[-_.,/]", " ", tag.lower().strip())
    return re.sub(r"\s+", " ", clean)


# --------------------------------------------------------------
# Прямой маппинг: имя подкатегории (нормализованное) → слаг
# --------------------------------------------------------------
_SUBCATEGORY_TO_SLUG = {}
for slug, display_name, *_ in _HIERARCHY_SEED:
    _SUBCATEGORY_TO_SLUG[_normalize_tag(display_name)] = slug


def _build_subcategory_mapping() -> None:
    global _SUBCATEGORY_TO_SLUG
    if _SUBCATEGORY_TO_SLUG:
        return
    for slug, display_name, *_ in _HIERARCHY_SEED:
        norm = _normalize_tag(display_name)
        if norm and norm not in _SUBCATEGORY_TO_SLUG:
            _SUBCATEGORY_TO_SLUG[norm] = slug

    from .taxonomy import INTEREST_TAXONOMY
    for global_cat, subcats in INTEREST_TAXONOMY.items():
        for subcat in subcats:
            norm = _normalize_tag(subcat)
            if not norm or norm in _SUBCATEGORY_TO_SLUG:
                continue
            for slug, display_name, *_ in _HIERARCHY_SEED:
                if _normalize_tag(display_name) == norm:
                    _SUBCATEGORY_TO_SLUG[norm] = slug
                    break


_build_subcategory_mapping()


def _build_alias_map() -> dict[str, str]:
    if _ALIAS_TO_SLUG:
        return _ALIAS_TO_SLUG

    taxonomy_to_slug = {}
    for slug, display_name, _, _, _ in _HIERARCHY_SEED:
        norm_name = _normalize_tag(display_name)
        taxonomy_to_slug.setdefault(norm_name, slug)

    for slug, entry in SEMANTIC_ONTOLOGY.items():
        _ALIAS_TO_SLUG[_normalize_tag(slug)] = slug
        for alias in entry.get("aliases", []):
            _ALIAS_TO_SLUG[_normalize_tag(alias)] = slug

    for _category, subcats in INTEREST_TAXONOMY.items():
        for subcat, anchors in subcats.items():
            norm_sub = _normalize_tag(subcat)
            real_slug = taxonomy_to_slug.get(norm_sub)
            if real_slug:
                _ALIAS_TO_SLUG.setdefault(norm_sub, real_slug)
                for anchor in anchors:
                    norm_anchor = _normalize_tag(anchor)
                    _ALIAS_TO_SLUG.setdefault(norm_anchor, real_slug)
            else:
                logger.warning(f"[alias_map] Taxonomy '{subcat}' not found in HIERARCHY_SEED")

    for slug, _, _, _, _ in _HIERARCHY_SEED:
        _ALIAS_TO_SLUG.setdefault(_normalize_tag(slug), slug)

    logger.info(f"[alias_map] Built with {len(_ALIAS_TO_SLUG)} entries from all sources")
    return _ALIAS_TO_SLUG


CANONICAL_SLUG_FIXES: dict[str, str] = {
    "разработка": "it_development",
    "программирование": "it_development",
    "programming": "it_development",
    "software": "it_development",
    "код": "it_development",
    "coding": "it_development",
    "python_dev": "backend_python",
    "python dev": "backend_python",
    "питон": "backend_python",
    "пайтон": "backend_python",
    "гейминг": "gaming",
    "игры": "gaming",
    "геймер": "gaming",
    "видеоигры": "gaming",
    "творчество и искусство": "creativity_art",
    "творчество": "creativity_art",
    "искусство": "creativity_art",
    "арт": "creativity_art",
    "креатив": "creativity_art",
    "дизайн": "creativity_art",
    "design": "creativity_art",
    "саморазвитие": "self_development",
    "рост": "self_development",
    "развитие": "self_development",
    "обучение": "self_development",
    "менеджмент и лидерство": "self_development",
    "лидерство": "self_development",
    "learning": "self_development",
    "спорт": "sports_active_life",
    "фитнес": "sports_active_life",
    "активный отдых": "sports_active_life",
    "активность": "sports_active_life",
    "sports": "sports_active_life",
    "fitness": "sports_active_life",
    "музыка": "music_audio",
    "аудио": "music_audio",
    "звук": "music_audio",
    "music": "music_audio",
    "audio": "music_audio",
    "психология": "psychology_relations",
    "отношения": "psychology_relations",
    "психотерапия": "psychology_relations",
    "общение": "psychology_relations",
    "эмоции": "psychology_relations",
    "psychology": "psychology_relations",
    "наука": "science_education",
    "образование": "science_education",
    "исследования": "science_education",
    "изучение": "science_education",
    "учеба": "science_education",
    "science": "science_education",
    "education": "science_education",
    "финансы": "finance_business",
    "бизнес": "finance_business",
    "инвестиции": "finance_business",
    "стартап": "finance_business",
    "деньги": "finance_business",
    "карьера": "finance_business",
    "finance": "finance_business",
    "business": "finance_business",
    "дом": "home_lifestyle",
    "быт": "home_lifestyle",
    "образ жизни": "home_lifestyle",
    "уют": "home_lifestyle",
    "хозяйство": "home_lifestyle",
    "жилье": "home_lifestyle",
    "lifestyle": "home_lifestyle",
    "кино": "cinema_video",
    "видео": "cinema_video",
    "фильмы": "cinema_video",
    "сериалы": "cinema_video",
    "кинематограф": "cinema_video",
    "cinema": "cinema_video",
    "литература": "literature_reading",
    "чтение": "literature_reading",
    "книги": "literature_reading",
    "книголюб": "literature_reading",
    "библиотека": "literature_reading",
    "literature": "literature_reading",
}


def _resolve_exact(tag: str) -> str | None:
    if not tag or not isinstance(tag, str):
        return None
    norm = _normalize_tag(tag)
    if not norm or len(norm) < 2:
        return None

    if norm in CANONICAL_SLUG_FIXES:
        resolved_slug = CANONICAL_SLUG_FIXES[norm]
        logger.info(f"[resolve] CANONICAL FIX: norm='{norm}' -> slug='{resolved_slug}'")
        return resolved_slug

    alias_map = _build_alias_map()
    if norm in alias_map:
        found_slug = alias_map[norm]
        logger.debug(f"[resolve] tag='{tag}' norm='{norm}' -> slug='{found_slug}' (alias_map EXACT)")
        return found_slug

    for slug, data in SEMANTIC_ONTOLOGY.items():
        if not isinstance(data, dict):
            continue
        for alias in data.get("aliases", []):
            if norm == _normalize_tag(alias):
                logger.debug(f"[resolve] tag='{tag}' norm='{norm}' -> slug='{slug}' (SEMANTIC_ONTOLOGY)")
                return slug

    for seed_slug, _, _, _, _ in _HIERARCHY_SEED:
        if _normalize_tag(seed_slug) == norm:
            logger.debug(f"[resolve] tag='{tag}' norm='{norm}' -> slug='{seed_slug}' (HIERARCHY_SEED)")
            return seed_slug

    return None


def resolve_tag_to_slug(tag: str, db: Session | None = None) -> str | None:
    slug = _resolve_exact(tag)
    if slug:
        return slug
    if db is None:
        logger.debug(f"[resolve] no db session for '{tag}', skipping vector step")
        return None
    return resolve_tags_batch(db, [tag]).get(tag)


def ensure_hierarchy_seeded(db: Session) -> None:
    if db.query(InterestHierarchyNode).count() > 0:
        return

    slug_to_node: dict[str, InterestHierarchyNode] = {}
    for slug, name, parent_slug, weight, category in _HIERARCHY_SEED:
        parent_id = slug_to_node[parent_slug].id if parent_slug and parent_slug in slug_to_node else None
        node = InterestHierarchyNode(
            name=name,
            slug=slug,
            parent_id=parent_id,
            path="/",
            depth=0,
            match_weight=weight,
            global_category=category,
        )
        db.add(node)
        db.flush()
        slug_to_node[slug] = node

    for node in slug_to_node.values():
        chain: list[InterestHierarchyNode] = [node]
        current = node
        while current.parent_id:
            parent = db.query(InterestHierarchyNode).get(current.parent_id)
            if not parent:
                break
            chain.insert(0, parent)
            current = parent
        node.path = "/" + "/".join(str(n.id) for n in chain) + "/"
        node.depth = len(chain) - 1
        db.add(node)

    db.commit()
    logger.info("Interest hierarchy seeded with %d nodes", len(slug_to_node))
    _ensure_node_embeddings(db)


def _get_ancestor_chain(db: Session, node: InterestHierarchyNode) -> list[InterestHierarchyNode]:
    chain = [node]
    current = node
    while current.parent_id:
        parent = db.query(InterestHierarchyNode).get(current.parent_id)
        if not parent:
            break
        chain.insert(0, parent)
        current = parent
    return chain


def register_user_tags(db: Session, user_id: int, tags: list[str]) -> None:
    ensure_hierarchy_seeded(db)
    if not tags:
        return

    slug_nodes = {n.slug: n for n in db.query(InterestHierarchyNode).all()}
    resolved_map = resolve_tags_batch(db, tags)

    for tag in tags:
        slug = resolved_map.get(tag)
        if not slug or slug not in slug_nodes:
            logger.debug(f"Cannot register tag '{tag}' - slug '{slug}' not found in hierarchy")
            continue

        node = slug_nodes[slug]
        ancestor_chain = _get_ancestor_chain(db, node)

        for chain_node in ancestor_chain:
            existing = (
                db.query(UserInterestGraphWeight)
                .filter_by(user_id=user_id, node_id=chain_node.id)
                .first()
            )
            new_weight = max(existing.weight if existing else 0.0, chain_node.match_weight)
            if existing:
                existing.weight = new_weight
                existing.source_tag = tag
            else:
                db.add(UserInterestGraphWeight(
                    user_id=user_id,
                    node_id=chain_node.id,
                    weight=new_weight,
                    source_tag=tag,
                ))

    db.commit()
    logger.debug(f"Registered {len(tags)} tags for user {user_id}")


def get_user_graph_weights(db: Session, user_id: int) -> dict[int, float]:
    rows = db.query(UserInterestGraphWeight).filter_by(user_id=user_id).all()
    return {row.node_id: row.weight for row in rows}


def build_query_weights(db: Session, tags: set[str]) -> dict[int, float]:
    weights: dict[int, float] = {}
    if not tags:
        logger.warning("[build_query_weights] Empty tag set received")
        return weights

    ensure_hierarchy_seeded(db)
    if not _SUBCATEGORY_TO_SLUG:
        _build_subcategory_mapping()

    logger.info(f"[build_query_weights] Processing {len(tags)} tags: {tags}")

    # 1. Сначала пытаемся разрешить все теги через точные методы (без батча)
    tag_to_slug: dict[str, str | None] = {}
    unresolved: list[str] = []
    for tag in tags:
        norm = _normalize_tag(tag)
        slug = _SUBCATEGORY_TO_SLUG.get(norm)
        if slug:
            tag_to_slug[tag] = slug
            continue
        slug = _resolve_exact(tag)
        if slug:
            tag_to_slug[tag] = slug
        else:
            unresolved.append(tag)

    # 2. Оставшиеся (неразрешённые) отправляем на батч-векторный резолвинг
    if unresolved:
        batch_resolved = resolve_tags_batch(db, unresolved)
        tag_to_slug.update(batch_resolved)

    # 3. Строим веса по полученным слагам
    for tag in tags:
        slug = tag_to_slug.get(tag)
        if not slug:
            logger.warning(f"[build_query_weights] Could not resolve tag '{tag}' to any slug")
            continue

        node = db.query(InterestHierarchyNode).filter(InterestHierarchyNode.slug == slug).first()
        if not node:
            logger.error(f"[build_query_weights] Slug '{slug}' (from tag '{tag}') NOT FOUND in database")
            continue

        logger.info(f"[build_query_weights] tag='{tag}' → slug='{slug}' (node_id={node.id}, match_weight={node.match_weight})")
        base_weight = node.match_weight if node.match_weight else 0.5
        weights[node.id] = max(weights.get(node.id, 0.0), base_weight)

        current_node = node
        decay_factor = 0.7
        while current_node.parent_id is not None:
            parent = db.query(InterestHierarchyNode).filter(
                InterestHierarchyNode.id == current_node.parent_id
            ).first()
            if not parent:
                break
            parent_weight = weights[current_node.id] * decay_factor
            weights[parent.id] = max(weights.get(parent.id, 0.0), parent_weight)
            current_node = parent

    logger.info(f"[build_query_weights] RESULT: {len(weights)} nodes, total_weight={sum(weights.values()):.2f}")
    return weights


def _weighted_overlap_score(
    weights_a: dict[int, float],
    weights_b: dict[int, float],
) -> float:
    if not weights_a or not weights_b:
        return 0.0
    shared_nodes = set(weights_a) & set(weights_b)
    if not shared_nodes:
        return 0.0
    dot = sum(weights_a[n] * weights_b[n] for n in shared_nodes)
    norm_a = sum(v * v for v in weights_a.values()) ** 0.5
    norm_b = sum(v * v for v in weights_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return min(1.0, dot / (norm_a * norm_b))


def calculate_graph_interest_score(
        db: Session,
        query_tags: set[str],
        other_user_id: int,
        other_extracted: Any | None = None,
        query_weights: dict[int, float] | None = None,
        other_weights: dict[int, float] | None = None,
        hierarchy_node_names: dict[int, str] | None = None,
) -> tuple[float, list[str]]:
    """
    Возвращает скор совпадения интересов и список совпавших тегов.
    
    ✅ ИСПРАВЛЕНО: Теперь различает прямое и косвенное совпадение:
    - Прямое: запрошенный slug точно совпадает с узлом → вес 1.0
    - Косвенное (через родителя/потомка): вес = 0.6
    """
    ensure_hierarchy_seeded(db)

    logger.info(f"[graph_score] START: other_user_id={other_user_id}, query_tags={query_tags}")

    if not query_weights:
        query_weights = build_query_weights(db, query_tags)
        logger.info(f"[graph_score] query_weights built: {len(query_weights)} nodes")
    else:
        logger.info(f"[graph_score] query_weights provided: {len(query_weights)} nodes")

    if not other_weights:
        other_weights = get_user_graph_weights(db, other_user_id)
        logger.info(f"[graph_score] get_user_graph_weights returned: {len(other_weights)} nodes")

        if not other_weights and other_extracted:
            other_tag_set = _extract_tags_from_profile(other_extracted)
            if other_tag_set:
                logger.info(
                    f"[graph_score] No pre-computed weights for user {other_user_id}. "
                    f"Extracted {len(other_tag_set)} tags from profile: {other_tag_set}"
                )
                other_weights = build_query_weights(db, other_tag_set)
                logger.info(f"[graph_score] Built weights from extracted tags: {len(other_weights)} nodes")
            else:
                logger.warning(f"[graph_score] Could not extract tags from other_extracted for user {other_user_id}")
        else:
            if not other_extracted:
                logger.warning(f"[graph_score] No pre-computed weights AND no other_extracted for user {other_user_id}")
    else:
        logger.info(f"[graph_score] other_weights provided: {len(other_weights)} nodes")

    # ✅ ИСПРАВЛЕНО: Используем модифицированный _weighted_overlap_score с учетом типа совпадения
    score = _weighted_overlap_score(query_weights, other_weights)
    logger.info(f"[graph_score] _weighted_overlap_score={score:.4f}")

    if not query_weights or not other_weights:
        logger.warning(f"[graph_score] Cannot calculate matches: query_weights={len(query_weights or {})} other_weights={len(other_weights or {})}")
        return float(score), []

    matched: list[str] = []
    direct_matches: list[tuple[float, str]] = []
    indirect_matches: list[tuple[float, str]] = []
    
    if hierarchy_node_names:
        shared_ids = set(query_weights.keys()) & set(other_weights.keys())
        logger.info(f"[graph_score] shared_ids: {len(shared_ids)} (query={len(query_weights)}, other={len(other_weights)})")

        if shared_ids:
            _query_resolved = resolve_tags_batch(db, list(query_tags))
            requested_slugs = {s for s in _query_resolved.values() if s}
            logger.debug(f"[graph_score] requested_slugs: {requested_slugs}")
            
            # Загружаем все узлы иерархии для проверки отношений
            all_nodes = db.query(InterestHierarchyNode).all()
            slug_to_node_id = {n.slug: n.id for n in all_nodes}
            node_id_to_node = {n.id: n for n in all_nodes}

            for nid in shared_ids:
                if nid not in hierarchy_node_names:
                    continue
                node = node_id_to_node.get(nid)
                if not node:
                    continue

                min_weight = min(query_weights[nid], other_weights[nid])
                match_type = None
                
                # ✅ Проверяем, является ли этот узел ПРЯМЫМ совпадением
                if node.slug in requested_slugs:
                    match_type = "direct"
                    direct_matches.append((min_weight, hierarchy_node_names[nid]))
                    logger.debug(f"[graph_score]   DIRECT MATCH: id={nid}, slug='{node.slug}', w={min_weight:.3f}")
                else:
                    # ✅ Проверяем, является ли это косвенным совпадением через родителей/потомков
                    # Восстанавливаем цепь родителей
                    is_indirect = False
                    current = node
                    chain = [current]
                    while current.parent_id:
                        parent = node_id_to_node.get(current.parent_id)
                        if not parent:
                            break
                        chain.insert(0, parent)
                        current = parent
                    
                    # Проверяем, есть ли запрошенный slug в цепи родителей/потомков
                    for chain_node in chain:
                        if chain_node.slug in requested_slugs:
                            is_indirect = True
                            break
                    
                    if is_indirect:
                        match_type = "indirect"
                        indirect_matches.append((min_weight, hierarchy_node_names[nid]))
                        logger.debug(f"[graph_score]   INDIRECT MATCH: id={nid}, slug='{node.slug}', w={min_weight:.3f}")
                    else:
                        logger.debug(f"[graph_score]   SKIP: id={nid}, slug='{node.slug}', w={min_weight:.3f} (no hierarchy connection)")

            # ✅ Приоритет: прямые совпадения идут первыми, затем косвенные
            direct_matches.sort(key=lambda x: x[0], reverse=True)
            indirect_matches.sort(key=lambda x: x[0], reverse=True)
            
            matched = [name for _, name in direct_matches] + [name for _, name in indirect_matches]
            logger.info(
                f"[graph_score] Matched tags for user {other_user_id}: "
                f"direct={len(direct_matches)}, indirect={len(indirect_matches)}, total={len(matched)}"
            )
        else:
            logger.warning(f"[graph_score] No shared_ids between query and other weights (0 matches)")
    else:
        logger.warning(f"[graph_score] hierarchy_node_names not provided, cannot build matched_tags")

    logger.info(f"[graph_score] RESULT: score={score:.4f}, matched_count={len(matched)}")
    return float(score), matched


def resolve_node_title_to_tags(
    db: Session, 
    node_title: str, 
    node_description: str = "",
    profiler=None
) -> set[str]:
    """
    Резолвит название узла графа (может быть длинной описательной фразой типа
    "я люблю музыку" или "создание моделей в Blender") в множество тегов.
    
    Поддерживает:
    1. Прямой резолвинг через словарь синонимов
    2. Семантический резолвинг через pgvector
    3. Структурированное извлечение через AIProfiler.extract_interests (если доступен)
    
    Args:
        db: SQLAlchemy сессия БД
        node_title: Название узла (может быть фразой)
        node_description: Описание узла (добавляется к фразе для контекста)
        profiler: AIProfiler для структурированного извлечения (опционально)
    
    Returns:
        set[str]: Множество разрешённых слагов (slug'ов) иерархии интересов
    """
    ensure_hierarchy_seeded(db)
    
    # Конкатенируем для полного контекста
    full_text = f"{node_title}. {node_description}".strip() if node_description else node_title
    resolved_tags = set()
    
    # Если есть профайлер, используем структурированное извлечение
    if profiler:
        try:
            from app.ai_profiler.contextual_adapter import get_contextual_adapter
            adapter = get_contextual_adapter()
            enriched = adapter.enrich_text(full_text).enriched
            interests = profiler.extract_interests(enriched)
            tag_set = profiler._extract_tag_set(interests)
            
            # Резолвим извлечённые теги в слаги
            if tag_set:
                batch_resolved = resolve_tags_batch(db, list(tag_set))
                for tag, slug in batch_resolved.items():
                    if slug:
                        resolved_tags.add(slug)
                
                logger.info(
                    f"[resolve_node_title] profiler extracted {len(tag_set)} tags, "
                    f"resolved to {len(resolved_tags)} slugs: {resolved_tags}"
                )
        except Exception as e:
            logger.warning(f"[resolve_node_title] profiler extraction failed: {e}")
    
    # Если структурированное извлечение не помогло, пытаемся прямой резолвинг
    if not resolved_tags:
        # Сначала весь текст целиком
        slug = resolve_tag_to_slug(full_text, db)
        if slug:
            resolved_tags.add(slug)
            logger.info(f"[resolve_node_title] Resolved full text '{full_text[:50]}...' to slug '{slug}'")
        else:
            # Пробуем разбить на слова и взять те, которые резолвятся
            words = full_text.split()
            for word in words:
                word_slug = resolve_tag_to_slug(word, db)
                if word_slug:
                    resolved_tags.add(word_slug)
            
            if resolved_tags:
                logger.info(
                    f"[resolve_node_title] Could not resolve full text, but found {len(resolved_tags)} "
                    f"slugs from individual words: {resolved_tags}"
                )
            else:
                # Последняя попытка: просто как тег
                norm_title = _normalize_tag(node_title)
                if norm_title:
                    resolved_tags.add(norm_title)
                    logger.warning(
                        f"[resolve_node_title] Could not resolve '{full_text[:50]}...' through any method, "
                        f"using normalized title as fallback: '{norm_title}'"
                    )
    
    return resolved_tags


def _extract_tags_from_profile(profile_data: Any) -> set[str]:
    tags = set()
    if profile_data is None:
        return tags

    for field in ["hobbies", "topics", "skills"]:
        try:
            val = getattr(profile_data, field, None) if hasattr(profile_data, '__dict__') else profile_data.get(field)
        except (AttributeError, TypeError):
            val = None

        if not val:
            continue

        if isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    tag_str = str(item).lower().strip()
                    if tag_str and len(tag_str) >= 2:
                        tags.add(tag_str)
                elif isinstance(item, dict):
                    if "subcategory" in item:
                        tag_str = str(item["subcategory"]).lower().strip()
                        if tag_str and len(tag_str) >= 2:
                            tags.add(tag_str)

    logger.debug(f"Extracted {len(tags)} tags from profile: {tags}")
    return tags