"""
Иерархический граф интересов: seed, регистрация тегов, расчёт graph score.

При совпадении листового тега (напр. Flask) веса распространяются вверх
по цепочке: Python (0.80) → Backend (0.50) → Programming (0.30).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy.orm import Session

from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY
from app.ai_profiler.taxonomy import INTEREST_TAXONOMY
from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight

logger = logging.getLogger(__name__)

# (slug, display_name, parent_slug, match_weight, global_category)
_HIERARCHY_SEED: list[tuple[str, str, str | None, float, str | None]] = [
    ("programming", "Программирование", None, 0.30, "work"),
    ("backend", "Backend", "programming", 0.50, "work"),
    ("python", "Python", "backend", 0.80, "work"),
    ("flask", "Flask", "python", 1.00, "work"),
    ("frontend", "Frontend", "programming", 0.75, "work"),
    ("devops", "DevOps", "programming", 0.70, "work"),
    ("ml_ai", "ML / AI", "programming", 0.85, "work"),
    ("design", "Дизайн", None, 0.60, "work"),
    ("startup", "Стартапы", None, 0.55, "work"),
    ("analytics", "Аналитика", None, 0.50, "work"),
    ("entertainment", "Развлечения", None, 0.30, "hobby"),
    ("gaming", "Гейминг", "entertainment", 0.70, "hobby"),
    ("cs2", "Counter-Strike", "gaming", 1.00, "hobby"),
    ("dota", "Dota 2", "gaming", 1.00, "hobby"),
    ("rpg", "RPG", "gaming", 0.90, "hobby"),
    ("fitness", "Спорт и фитнес", None, 0.80, "hobby"),
    ("music", "Музыка", None, 0.75, "hobby"),
    ("psychology", "Психология", None, 0.70, "psychology"),
    ("self_development", "Саморазвитие", None, 0.65, "psychology"),
]

_ALIAS_TO_SLUG: dict[str, str] = {}


def _normalize_tag(tag: str) -> str:
    """Очищает тег, превращая спецсимволы в пробелы (например, 'python_dev' -> 'python dev')."""
    if not tag:
        return ""
    # Заменяем подчеркивания, дефисы и слэши на пробелы
    clean = re.sub(r"[-_.,/]", " ", tag.lower().strip())
    return re.sub(r"\s+", " ", clean)


def _build_alias_map() -> dict[str, str]:
    if _ALIAS_TO_SLUG:
        return _ALIAS_TO_SLUG

    # 1. Добавляем из SEMANTIC_ONTOLOGY (приоритет 1 - самые частые теги)
    for slug, entry in SEMANTIC_ONTOLOGY.items():
        # Основной слаг
        _ALIAS_TO_SLUG[_normalize_tag(slug)] = slug
        
        # Разбираем все aliases (поле "aliases" в SEMANTIC_ONTOLOGY)
        for alias in entry.get("aliases", []):
            norm_alias = _normalize_tag(alias)
            _ALIAS_TO_SLUG[norm_alias] = slug

    # 2. Добавляем из INTEREST_TAXONOMY (менее специфичные)
    for _category, subcats in INTEREST_TAXONOMY.items():
        for subcat, anchors in subcats.items():
            norm_sub = _normalize_tag(subcat)
            slug_guess = norm_sub.replace(" ", "_").replace("/", "_")
            _ALIAS_TO_SLUG.setdefault(norm_sub, slug_guess)
            for anchor in anchors:
                norm_anchor = _normalize_tag(anchor)
                _ALIAS_TO_SLUG.setdefault(norm_anchor, _ALIAS_TO_SLUG[norm_sub])

    # 3. Добавляем из seed (как fallback)
    for slug, _, _, _, _ in _HIERARCHY_SEED:
        norm_slug = _normalize_tag(slug)
        _ALIAS_TO_SLUG.setdefault(norm_slug, slug)

    logger.debug(f"Alias map built with {len(_ALIAS_TO_SLUG)} entries")
    return _ALIAS_TO_SLUG


def resolve_tag_to_slug(tag: str) -> str | None:
    """
    Сопоставляет произвольный тег с slug узла графа.
    
    Стратегия:
    1. Нормализуем входящий тег (приводим к нижнему регистру, очищаем спецсимволы)
    2. Ищем ТОЧНОЕ совпадение в alias_map
    3. Пробуем поиск в SEMANTIC_ONTOLOGY по aliases
    4. Пробуем поиск в HIERARCHY_SEED по slug
    5. Если ничего не нашли - возвращаем None
    """
    if not tag or not isinstance(tag, str):
        return None

    norm = _normalize_tag(tag)
    if not norm or len(norm) < 2:
        return None

    alias_map = _build_alias_map()

    # Шаг 1: Точное совпадение с нормализованным алиасом
    if norm in alias_map:
        found_slug = alias_map[norm]
        logger.debug(f"Tag '{tag}' -> slug '{found_slug}' (alias_map match)")
        return found_slug

    # Шаг 2: Поиск в SEMANTIC_ONTOLOGY по aliases (более аккуратно)
    for slug, data in SEMANTIC_ONTOLOGY.items():
        if not isinstance(data, dict):
            continue
        
        aliases = data.get("aliases", [])
        for alias in aliases:
            norm_alias = _normalize_tag(alias)
            if norm == norm_alias:
                logger.debug(f"Tag '{tag}' -> slug '{slug}' (semantic_ontology aliases match)")
                return slug

    # Шаг 3: Поиск в seed
    for seed_slug, _, _, _, _ in _HIERARCHY_SEED:
        if _normalize_tag(seed_slug) == norm:
            logger.debug(f"Tag '{tag}' -> slug '{seed_slug}' (hierarchy_seed match)")
            return seed_slug

    # Шаг 4: Если ничего не нашли - пробуем частичный поиск (последняя попытка)
    # Ищем alias, у которого наш тег является подстрокой или vice versa
    norm_words = set(norm.split())
    for alias, slug in alias_map.items():
        alias_words = set(alias.split())
        # Если пересечение слов есть и оно не тривиально, вернем slug
        if norm_words & alias_words and len(norm_words & alias_words) >= 1:
            # Убедимся, что это не просто общее слово типа "и", "в" и т.д.
            if not all(len(w) <= 2 for w in (norm_words & alias_words)):
                logger.debug(f"Tag '{tag}' -> slug '{slug}' (partial word match)")
                return slug

    logger.debug(f"Tag '{tag}' -> None (no match found)")
    return None

def ensure_hierarchy_seeded(db: Session) -> None:
    """Создаёт узлы графа, если таблица пуста."""
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
    """
    Записывает теги пользователя в граф с пропагацией весов к предкам.
    
    Для каждого тега:
    1. Разрешаем его в slug
    2. Находим узел и его цепь предков
    3. Записываем веса для узла и всех его предков
    """
    ensure_hierarchy_seeded(db)
    if not tags:
        return

    slug_nodes = {n.slug: n for n in db.query(InterestHierarchyNode).all()}

    for tag in tags:
        slug = resolve_tag_to_slug(tag)
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
    """
    Строит веса узлов графа на основе переданных тегов.
    
    Для каждого тега:
    1. Разрешаем его в slug
    2. Находим узел в БД
    3. Назначаем вес узлу
    4. Рекурсивно идём к родителям, уменьшая вес (0.7 множитель)
    
    Это обеспечивает распространение релевантности вверх по иерархии.
    """
    weights: dict[int, float] = {}
    
    if not tags:
        return weights

    ensure_hierarchy_seeded(db)
    
    for tag in tags:
        slug = resolve_tag_to_slug(tag)
        if not slug:
            logger.debug(f"Could not resolve tag '{tag}' to slug")
            continue

        # Запрос узла по slug (строгое совпадение)
        node = db.query(InterestHierarchyNode).filter(
            InterestHierarchyNode.slug == slug
        ).first()
        
        if not node:
            logger.debug(f"Slug '{slug}' (from tag '{tag}') not found in hierarchy")
            continue

        logger.debug(f"Processing tag '{tag}' → slug '{slug}' (node_id={node.id})")

        # Вес самого узла
        base_weight = node.match_weight if node.match_weight else 0.5
        weights[node.id] = max(weights.get(node.id, 0.0), base_weight)

        # Рекурсивно поднимаемся к родителям
        current_node = node
        decay_factor = 0.7
        while current_node.parent_id is not None:
            parent = db.query(InterestHierarchyNode).filter(
                InterestHierarchyNode.id == current_node.parent_id
            ).first()
            
            if not parent:
                logger.debug(f"Parent node with id={current_node.parent_id} not found")
                break

            # Вес родителя = вес текущего × decay_factor
            parent_weight = weights[current_node.id] * decay_factor
            weights[parent.id] = max(weights.get(parent.id, 0.0), parent_weight)
            
            logger.debug(f"  Parent: node_id={parent.id}, slug='{parent.slug}', weight={parent_weight:.2f}")
            
            current_node = parent

    logger.debug(f"Query weights computed: {len(weights)} nodes with total weight {sum(weights.values()):.2f}")
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
    Рассчитывает graph score [0, 1] и точечный список совпавших тегов.
    
    Args:
        db: SQLAlchemy session
        query_tags: Набор тегов поиска
        other_user_id: ID другого пользователя
        other_extracted: AIExtractedInterests объект или словарь с ключами {hobbies, topics, skills, ...}
        query_weights: Предвычисленные веса для query_tags (опционально)
        other_weights: Предвычисленные веса для другого пользователя (опционально)
        hierarchy_node_names: Маппинг {node_id: node_name} для вывода (опционально)
    
    Returns:
        (score, matched_tags_list) где score ∈ [0, 1], matched_tags - список названий узлов
    """
    ensure_hierarchy_seeded(db)

    if not query_weights:
        query_weights = build_query_weights(db, query_tags)

    if not other_weights:
        other_weights = get_user_graph_weights(db, other_user_id)

        # Если в БД нет весов, пробуем вытащить теги из other_extracted
        if not other_weights and other_extracted:
            other_tag_set = _extract_tags_from_profile(other_extracted)
            
            if other_tag_set:
                logger.debug(
                    f"No pre-computed weights for user {other_user_id}. "
                    f"Extracting {len(other_tag_set)} tags from profile data: {other_tag_set}"
                )
                other_weights = build_query_weights(db, other_tag_set)

    # Рассчитываем математическую совместимость по полным графам
    score = _weighted_overlap_score(query_weights, other_weights)

    matched: list[str] = []

    # ФИЛЬТРАЦИЯ: Собираем matched_tags с умом (без мусора из соседних веток)
    if query_weights and other_weights and hierarchy_node_names:
        shared_ids = set(query_weights.keys()) & set(other_weights.keys())

        if shared_ids:
            # Получаем slug'и, которые пользователь запросил (для точной фильтрации)
            requested_slugs = {
                resolve_tag_to_slug(t) 
                for t in query_tags 
                if resolve_tag_to_slug(t)
            }

            valid_matches = []
            for nid in shared_ids:
                if nid not in hierarchy_node_names:
                    continue

                # Подтягиваем узел для проверки slug и weight
                node = db.query(InterestHierarchyNode).filter(
                    InterestHierarchyNode.id == nid
                ).first()
                if not node:
                    continue

                # Минимальный вес пересечения между query и other
                min_weight = min(query_weights[nid], other_weights[nid])

                # Условие для добавления в matched_tags:
                # 1. Либо это напрямую запрошенный тег (его slug в requested_slugs)
                # 2. Либо вес пересечения >= 0.6 (высокий threshold, чтобы отсечь эхо весов)
                if node.slug in requested_slugs or min_weight >= 0.6:
                    valid_matches.append((min_weight, hierarchy_node_names[nid]))

            # Сортируем по релевантности (весу) и готовим финальный список
            valid_matches.sort(key=lambda x: x[0], reverse=True)
            matched = [name for _, name in valid_matches]

            logger.debug(
                f"Matched tags for user {other_user_id}: "
                f"shared_ids={len(shared_ids)}, valid_matches={len(matched)}, "
                f"matched={matched}"
            )

    return float(score), matched


def _extract_tags_from_profile(profile_data: Any) -> set[str]:
    """
    Извлекает теги из профиля пользователя.
    
    Работает как с AIExtractedInterests объектом, так и с словарём.
    Ищет строки и подстроки в полях hobbies, topics, skills.
    """
    tags = set()

    if profile_data is None:
        return tags

    # Если это объект с атрибутами (AIExtractedInterests)
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
                    # Может быть структура типа {"subcategory": "...", "level": "..."} 
                    if "subcategory" in item:
                        tag_str = str(item["subcategory"]).lower().strip()
                        if tag_str and len(tag_str) >= 2:
                            tags.add(tag_str)

    logger.debug(f"Extracted {len(tags)} tags from profile: {tags}")
    return tags
