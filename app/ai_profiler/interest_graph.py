"""
Семантический граф интересов: иерархия, эмбеддинги, веса пользователей.

Архитектура:
- WRITE-фаза (Celery): теги предварительно разрешаются в слаг через dynamic_enrichment.py
- READ-фаза (match_by_node): один SQL-запрос, без векторного разрешения
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.ai_profiler.contextual_adapter import get_contextual_adapter
from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY
from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight

logger = logging.getLogger(__name__)

_HIERARCHY_SEED = [
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
    ("frontend_dev", "Фронтенд-разработка", "it_development", 0.5, "work"),
    ("javascript", "JavaScript", "frontend_dev", 0.85, "work"),
    ("react", "React", "javascript", 0.9, "work"),
    ("vue", "Vue.js", "javascript", 0.9, "work"),
    ("typescript", "TypeScript", "frontend_dev", 0.85, "work"),
    ("css", "CSS & Design Systems", "frontend_dev", 0.8, "work"),
    ("mobile_dev", "Мобильная разработка", "it_development", 0.5, "work"),
    ("react_native", "React Native", "mobile_dev", 0.85, "work"),
    ("ios_dev", "iOS разработка", "mobile_dev", 0.85, "work"),
    ("android_dev", "Android разработка", "mobile_dev", 0.85, "work"),
    ("devops", "DevOps & Infrastructure", "it_development", 0.5, "work"),
    ("docker", "Docker", "devops", 0.85, "work"),
    ("kubernetes", "Kubernetes", "devops", 0.85, "work"),
    ("ci_cd", "CI/CD", "devops", 0.85, "work"),
    ("data_science", "Data Science", "it_development", 0.5, "work"),
    ("machine_learning", "Machine Learning", "data_science", 0.85, "work"),
    ("deep_learning", "Deep Learning", "machine_learning", 0.9, "work"),
    ("nlp", "Natural Language Processing", "machine_learning", 0.85, "work"),
    ("indie_games", "Инди-игры", "gaming", 0.6, "entertainment"),
    ("competitive_gaming", "Киберспорт", "gaming", 0.6, "entertainment"),
    ("game_development", "Game Development", "gaming", 0.6, "work"),
    ("game_engines", "Game Engines", "game_development", 0.75, "work"),
    ("unity", "Unity", "game_engines", 0.85, "work"),
    ("unreal_engine", "Unreal Engine", "game_engines", 0.85, "work"),
    ("godot", "Godot", "game_engines", 0.85, "work"),
    ("music_production", "Производство музыки", "creativity_art", 0.7, "life"),
    ("digital_art", "Цифровое искусство", "creativity_art", 0.7, "life"),
    ("3d_modeling", "3D-моделирование", "creativity_art", 0.75, "life"),
    ("blender", "Blender", "3d_modeling", 0.85, "life"),
    ("character_design", "Character Design", "digital_art", 0.8, "life"),
    ("animation", "Анимация", "digital_art", 0.8, "life"),
    ("graphic_design", "Графический дизайн", "creativity_art", 0.7, "life"),
    ("ux_ui_design", "UX/UI Design", "graphic_design", 0.8, "work"),
    ("photography", "Фотография", "creativity_art", 0.7, "life"),
    ("videography", "Видеография", "cinema_video", 0.75, "life"),
    ("fitness_training", "Фитнес и тренировки", "sports_active_life", 0.7, "life"),
    ("yoga", "Йога", "sports_active_life", 0.7, "life"),
    ("martial_arts", "Боевые искусства", "sports_active_life", 0.7, "life"),
    ("hiking_outdoor", "Туризм и походы", "sports_active_life", 0.7, "life"),
    ("team_sports", "Командные виды спорта", "sports_active_life", 0.7, "life"),
    ("football", "Футбол", "team_sports", 0.8, "life"),
    ("basketball", "Баскетбол", "team_sports", 0.8, "life"),
    ("mental_health", "Психическое здоровье", "psychology_relations", 0.7, "life"),
    ("relationships", "Отношения", "psychology_relations", 0.7, "life"),
    ("personal_growth", "Личностный рост", "psychology_relations", 0.7, "life"),
    ("mindfulness", "Осознанность", "psychology_relations", 0.7, "life"),
    ("therapy", "Терапия", "mental_health", 0.8, "life"),
    ("music_genres", "Музыкальные жанры", "music_audio", 0.7, "entertainment"),
    ("classical_music", "Классическая музыка", "music_genres", 0.8, "entertainment"),
    ("jazz", "Джаз", "music_genres", 0.8, "entertainment"),
    ("rock_music", "Рок", "music_genres", 0.8, "entertainment"),
    ("electronic_music", "Электронная музыка", "music_genres", 0.8, "entertainment"),
    ("hip_hop", "Хип-хоп", "music_genres", 0.8, "entertainment"),
    ("cinema_genres", "Жанры кино", "cinema_video", 0.7, "entertainment"),
    ("horror", "Ужасы", "cinema_genres", 0.8, "entertainment"),
    ("scifi", "Научная фантастика", "cinema_genres", 0.8, "entertainment"),
    ("drama", "Драма", "cinema_genres", 0.8, "entertainment"),
    ("comedy", "Комедия", "cinema_genres", 0.8, "entertainment"),
    ("animation_films", "Мультфильмы", "cinema_video", 0.75, "entertainment"),
    ("reading_genres", "Жанры литературы", "literature_reading", 0.7, "entertainment"),
    ("fantasy", "Фантастика", "reading_genres", 0.8, "entertainment"),
    ("mystery", "Детективы", "reading_genres", 0.8, "entertainment"),
    ("romance", "Романтика", "reading_genres", 0.8, "entertainment"),
    ("science_fiction_lit", "Научная фантастика (книги)", "reading_genres", 0.8, "entertainment"),
    ("learning_skills", "Обучение новым навыкам", "self_development", 0.7, "life"),
    ("career_development", "Развитие карьеры", "self_development", 0.7, "work"),
    ("leadership", "Лидерство", "self_development", 0.7, "work"),
    ("public_speaking", "Публичные выступления", "self_development", 0.7, "life"),
    ("time_management", "Управление временем", "self_development", 0.7, "life"),
    ("physics", "Физика", "science_education", 0.7, "work"),
    ("chemistry", "Химия", "science_education", 0.7, "work"),
    ("biology", "Биология", "science_education", 0.7, "work"),
    ("astronomy", "Астрономия", "science_education", 0.7, "life"),
    ("mathematics", "Математика", "science_education", 0.7, "work"),
    ("linguistics", "Лингвистика", "science_education", 0.7, "work"),
    ("investing", "Инвестирование", "finance_business", 0.7, "work"),
    ("startups", "Стартапы", "finance_business", 0.7, "work"),
    ("trading", "Торговля", "finance_business", 0.7, "work"),
    ("cryptocurrency", "Криптовалюты", "finance_business", 0.7, "work"),
    ("entrepreneurship", "Предпринимательство", "finance_business", 0.7, "work"),
    ("cooking", "Кулинария", "home_lifestyle", 0.7, "life"),
    ("gardening", "Садоводство", "home_lifestyle", 0.7, "life"),
    ("interior_design", "Дизайн интерьера", "home_lifestyle", 0.7, "life"),
    ("sustainability", "Устойчивый образ жизни", "home_lifestyle", 0.7, "life"),
]


def _node_source_text(slug: str, display_name: str) -> str:
    """Формирует обогащённый текст узла для эмбеддинга: имя + алиасы из онтологии.

    Args:
        slug: Слаг узла иерархии.
        display_name: Отображаемое имя узла.

    Returns:
        Строка текста для векторного кодирования.
    """
    onto_entry = SEMANTIC_ONTOLOGY.get(slug)
    if onto_entry and isinstance(onto_entry, dict):
        desc = onto_entry.get("enriched_text", "")
        if not desc:
            aliases = onto_entry.get("aliases", [])
            desc = f"{display_name}. " + ", ".join(aliases[:10])
        return desc
    return display_name


def _ensure_node_embeddings(db: Session) -> None:
    """Генерирует pgvector-эмбеддинги для всех узлов, у которых их нет.

    Args:
        db: Сессия SQLAlchemy.
    """
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
    """Очищает и перегенерирует эмбеддинги всех узлов иерархии.

    Args:
        db: Сессия SQLAlchemy.

    Returns:
        Количество перегенерированных эмбеддингов.
    """
    count = db.query(InterestHierarchyNode).update({InterestHierarchyNode.embedding: None})
    db.commit()
    _ensure_node_embeddings(db)
    return count


def ensure_hierarchy_seeded(db: Session) -> None:
    """Инициализирует иерархию интересов из _HIERARCHY_SEED, если таблица пуста.

    Также вычисляет depth и materialized path для каждого узла,
    и генерирует эмбеддинги для новых узлов.

    Args:
        db: Сессия SQLAlchemy.
    """
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
    logger.info(f"[ensure_hierarchy_seeded] Seeded {len(slug_to_node)} nodes")
    _ensure_node_embeddings(db)


def _get_ancestor_chain(db: Session, node: InterestHierarchyNode) -> list[InterestHierarchyNode]:
    """Возвращает цепочку предков узла (от корня до самого узла).

    Args:
        db: Сессия SQLAlchemy.
        node: Узел иерархии.

    Returns:
        Список узлов от корневого до переданного.
    """
    chain = [node]
    current = node
    while current.parent_id:
        parent = db.query(InterestHierarchyNode).get(current.parent_id)
        if not parent:
            break
        chain.insert(0, parent)
        current = parent
    return chain


def register_user_tags(db: Session, user_id: int, resolved_slugs: list[str]) -> None:
    """Регистрирует предварительно разрешённые теги пользователя в графе весов.

    Ожидает, что все слаги уже разрешены и валидны (WRITE-фаза).
    Для каждого слага находит узел иерархии и propagates вес
    на всех предков с максимизацией.

    Args:
        db: Сессия SQLAlchemy.
        user_id: ID пользователя.
        resolved_slugs: Список валидных слагов иерархии.
    """
    ensure_hierarchy_seeded(db)
    
    if not resolved_slugs:
        return

    slug_nodes = {n.slug: n for n in db.query(InterestHierarchyNode).all()}

    for slug in resolved_slugs:
        if not slug or slug not in slug_nodes:
            logger.warning(f"[register_user_tags] Slug '{slug}' not found in hierarchy, skipping")
            continue

        node = slug_nodes[slug]
        ancestor_chain = _get_ancestor_chain(db, node)

        for chain_node in ancestor_chain:
            existing = (
                db.query(UserInterestGraphWeight)
                .filter_by(user_id=user_id, node_id=chain_node.id)
                .first()
            )
            
            new_weight = max(existing.weight if existing else 0.0, chain_node.match_weight or 1.0)
            
            if existing:
                existing.weight = new_weight
                existing.source_tag = slug
            else:
                db.add(UserInterestGraphWeight(
                    user_id=user_id,
                    node_id=chain_node.id,
                    weight=new_weight,
                    source_tag=slug,
                ))

    db.commit()
    logger.debug(f"[register_user_tags] Registered {len(resolved_slugs)} slugs for user {user_id}")


def get_user_graph_weights(db: Session, user_id: int) -> dict[int, float]:
    """Возвращает все веса графа интересов пользователя.

    Args:
        db: Сессия SQLAlchemy.
        user_id: ID пользователя.

    Returns:
        Словарь {node_id: weight}.
    """
    rows = db.query(UserInterestGraphWeight).filter_by(user_id=user_id).all()
    return {row.node_id: row.weight for row in rows}


def compute_hierarchical_overlap_score(
    db: Session, user_id_1: int, user_id_2: int,
) -> float:
    """Вычисляет иерархическое перекрытие графов интересов двух пользователей.

    Учитывает глубину узлов: чем ближе к корню, тем меньше вклад.

    Args:
        db: Сессия SQLAlchemy.
        user_id_1: ID первого пользователя.
        user_id_2: ID второго пользователя.

    Returns:
        Балл перекрытия от 0.0 до 1.0.
    """
    weights_1 = get_user_graph_weights(db, user_id_1)
    weights_2 = get_user_graph_weights(db, user_id_2)

    if not weights_1 or not weights_2:
        return 0.0

    common_ids = set(weights_1.keys()) & set(weights_2.keys())
    if not common_ids:
        return 0.0

    all_nodes = {n.id: n for n in db.query(InterestHierarchyNode).all()}
    total_weighted = 0.0
    max_possible = 0.0

    for node_id in common_ids:
        node = all_nodes.get(node_id)
        if not node:
            continue
        depth_factor = 1.0 / (1.0 + node.depth)
        combined = (weights_1[node_id] + weights_2[node_id]) / 2.0
        total_weighted += combined * depth_factor
        max_possible += 1.0 * depth_factor

    if max_possible == 0.0:
        return 0.0

    return min(total_weighted / max_possible, 1.0)


def compute_jaccard_interest_similarity(
    db: Session, user_id_1: int, user_id_2: int,
) -> float:
    """Вычисляет Jaccard-схожесть множеств узлов интересов двух пользователей.

    Args:
        db: Сессия SQLAlchemy.
        user_id_1: ID первого пользователя.
        user_id_2: ID второго пользователя.

    Returns:
        Балл схожести от 0.0 до 1.0.
    """
    weights_1 = get_user_graph_weights(db, user_id_1)
    weights_2 = get_user_graph_weights(db, user_id_2)

    set_1 = set(weights_1.keys())
    set_2 = set(weights_2.keys())

    if not set_1 and not set_2:
        return 0.0

    intersection = set_1 & set_2
    union = set_1 | set_2

    return len(intersection) / len(union) if union else 0.0


def resolve_tags_batch(
    db: Session, raw_tags: list[str], force: bool = False,
) -> dict[str, Optional[str]]:
    """Пакетно разрешает сырые теги в слаги с кэшированием через DynamicTagEnricher.

    Args:
        db: Сессия SQLAlchemy.
        raw_tags: Список исходных тегов.
        force: Переразрешить даже если тег ранее помечен как неразрешимый.

    Returns:
        Словарь {raw_tag: slug_or_None}.
    """
    from app.ai_profiler.dynamic_enrichment import get_tag_enricher

    enricher = get_tag_enricher()
    result: dict[str, Optional[str]] = {}
    for tag in raw_tags:
        try:
            slug = enricher.resolve_tag_to_slug(db, tag, fallback_to_enrichment=True, force=force)
            result[tag] = slug
        except Exception as e:
            logger.warning(f"[resolve_tags_batch] Failed to resolve '{tag}': {e}")
            result[tag] = None
    return result


def calculate_graph_interest_score(
    db: Session,
    query_tags: set[str],
    target_user_id: int,
) -> tuple[float, list[str]]:
    """Вычисляет балл пересечения интересов между набором тегов и пользователем.

    Прямое совпадение (точный слаг): weight * 1.0
    Косвенное совпадение (родитель/потомок): weight * 0.4 с понижением за глубину.

    Args:
        db: Сессия SQLAlchemy.
        query_tags: Набор исходных тегов для поиска.
        target_user_id: ID целевого пользователя.

    Returns:
        Кортеж (score, matched_tag_names): балл и список совпавших названий.
    """
    if not query_tags:
        return 0.0, []

    resolved = resolve_tags_batch(db, list(query_tags))
    requested_slugs = {s for s in resolved.values() if s}
    if not requested_slugs:
        return 0.0, []

    all_nodes = {n.slug: n for n in db.query(InterestHierarchyNode).all()}
    user_weights = get_user_graph_weights(db, target_user_id)
    node_map = {n.id: n for n in all_nodes.values()}

    score = 0.0
    matched_names: list[str] = []

    for node_id, weight in user_weights.items():
        node = node_map.get(node_id)
        if not node:
            continue

        if node.slug in requested_slugs:
            score += weight * 1.0
            matched_names.append(node.name)
        else:
            current = node
            is_indirect = False
            while current.parent_id:
                parent = node_map.get(current.parent_id)
                if not parent:
                    break
                if parent.slug in requested_slugs:
                    is_indirect = True
                    break
                current = parent

            if is_indirect:
                depth_diff = abs(node.depth - current.depth)
                coeff = max(0.4 - 0.05 * depth_diff, 0.1)
                score += weight * coeff
                matched_names.append(f"{node.name} (похоже)")

    final_score = min(score, 1.0)
    return final_score, matched_names
