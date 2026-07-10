"""
Clean hierarchical interest graph system.

ARCHITECTURE:
- WRITE phase (Celery): All tags are pre-resolved to slugs via dynamic_enrichment.py
- READ phase (match_by_node): Ultra-fast, single SQL query, no vector resolution

This module handles:
1. Hierarchy seeding and embedding generation
2. User tag registration (expects pre-resolved slugs)
3. User graph weight queries
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.ai_profiler.contextual_adapter import get_contextual_adapter
from app.ai_profiler.semantic_ontology import SEMANTIC_ONTOLOGY
from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight

logger = logging.getLogger(__name__)

# Hierarchy seed: (slug, display_name, parent_slug, match_weight, global_category)
_HIERARCHY_SEED = [
    # Root nodes
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

    # IT & Development subtree
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

    # Frontend
    ("frontend_dev", "Фронтенд-разработка", "it_development", 0.5, "work"),
    ("javascript", "JavaScript", "frontend_dev", 0.85, "work"),
    ("react", "React", "javascript", 0.9, "work"),
    ("vue", "Vue.js", "javascript", 0.9, "work"),
    ("typescript", "TypeScript", "frontend_dev", 0.85, "work"),
    ("css", "CSS & Design Systems", "frontend_dev", 0.8, "work"),

    # Mobile development
    ("mobile_dev", "Мобильная разработка", "it_development", 0.5, "work"),
    ("react_native", "React Native", "mobile_dev", 0.85, "work"),
    ("ios_dev", "iOS разработка", "mobile_dev", 0.85, "work"),
    ("android_dev", "Android разработка", "mobile_dev", 0.85, "work"),

    # DevOps & Infrastructure
    ("devops", "DevOps & Infrastructure", "it_development", 0.5, "work"),
    ("docker", "Docker", "devops", 0.85, "work"),
    ("kubernetes", "Kubernetes", "devops", 0.85, "work"),
    ("ci_cd", "CI/CD", "devops", 0.85, "work"),

    # Data Science
    ("data_science", "Data Science", "it_development", 0.5, "work"),
    ("machine_learning", "Machine Learning", "data_science", 0.85, "work"),
    ("deep_learning", "Deep Learning", "machine_learning", 0.9, "work"),
    ("nlp", "Natural Language Processing", "machine_learning", 0.85, "work"),

    # Gaming subtree
    ("indie_games", "Инди-игры", "gaming", 0.6, "entertainment"),
    ("competitive_gaming", "Киберспорт", "gaming", 0.6, "entertainment"),
    ("game_development", "Game Development", "gaming", 0.6, "work"),
    ("game_engines", "Game Engines", "game_development", 0.75, "work"),
    ("unity", "Unity", "game_engines", 0.85, "work"),
    ("unreal_engine", "Unreal Engine", "game_engines", 0.85, "work"),
    ("godot", "Godot", "game_engines", 0.85, "work"),

    # Creativity & Arts
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

    # Sports & Active life
    ("fitness_training", "Фитнес и тренировки", "sports_active_life", 0.7, "life"),
    ("yoga", "Йога", "sports_active_life", 0.7, "life"),
    ("martial_arts", "Боевые искусства", "sports_active_life", 0.7, "life"),
    ("hiking_outdoor", "Туризм и походы", "sports_active_life", 0.7, "life"),
    ("team_sports", "Командные виды спорта", "sports_active_life", 0.7, "life"),
    ("football", "Футбол", "team_sports", 0.8, "life"),
    ("basketball", "Баскетбол", "team_sports", 0.8, "life"),

    # Psychology & Relations
    ("mental_health", "Психическое здоровье", "psychology_relations", 0.7, "life"),
    ("relationships", "Отношения", "psychology_relations", 0.7, "life"),
    ("personal_growth", "Личностный рост", "psychology_relations", 0.7, "life"),
    ("mindfulness", "Осознанность", "psychology_relations", 0.7, "life"),
    ("therapy", "Терапия", "mental_health", 0.8, "life"),

    # Music & Audio
    ("music_genres", "Музыкальные жанры", "music_audio", 0.7, "entertainment"),
    ("classical_music", "Классическая музыка", "music_genres", 0.8, "entertainment"),
    ("jazz", "Джаз", "music_genres", 0.8, "entertainment"),
    ("rock_music", "Рок", "music_genres", 0.8, "entertainment"),
    ("electronic_music", "Электронная музыка", "music_genres", 0.8, "entertainment"),
    ("hip_hop", "Хип-хоп", "music_genres", 0.8, "entertainment"),

    # Cinema & Video
    ("cinema_genres", "Жанры кино", "cinema_video", 0.7, "entertainment"),
    ("horror", "Ужасы", "cinema_genres", 0.8, "entertainment"),
    ("scifi", "Научная фантастика", "cinema_genres", 0.8, "entertainment"),
    ("drama", "Драма", "cinema_genres", 0.8, "entertainment"),
    ("comedy", "Комедия", "cinema_genres", 0.8, "entertainment"),
    ("animation_films", "Мультфильмы", "cinema_video", 0.75, "entertainment"),

    # Literature
    ("reading_genres", "Жанры литературы", "literature_reading", 0.7, "entertainment"),
    ("fantasy", "Фантастика", "reading_genres", 0.8, "entertainment"),
    ("mystery", "Детективы", "reading_genres", 0.8, "entertainment"),
    ("romance", "Романтика", "reading_genres", 0.8, "entertainment"),
    ("science_fiction_lit", "Научная фантастика (книги)", "reading_genres", 0.8, "entertainment"),

    # Self-development
    ("learning_skills", "Обучение новым навыкам", "self_development", 0.7, "life"),
    ("career_development", "Развитие карьеры", "self_development", 0.7, "work"),
    ("leadership", "Лидерство", "self_development", 0.7, "work"),
    ("public_speaking", "Публичные выступления", "self_development", 0.7, "life"),
    ("time_management", "Управление временем", "self_development", 0.7, "life"),

    # Science & Education
    ("physics", "Физика", "science_education", 0.7, "work"),
    ("chemistry", "Химия", "science_education", 0.7, "work"),
    ("biology", "Биология", "science_education", 0.7, "work"),
    ("astronomy", "Астрономия", "science_education", 0.7, "life"),
    ("mathematics", "Математика", "science_education", 0.7, "work"),
    ("linguistics", "Лингвистика", "science_education", 0.7, "work"),

    # Finance & Business
    ("investing", "Инвестирование", "finance_business", 0.7, "work"),
    ("startups", "Стартапы", "finance_business", 0.7, "work"),
    ("trading", "Торговля", "finance_business", 0.7, "work"),
    ("cryptocurrency", "Криптовалюты", "finance_business", 0.7, "work"),
    ("entrepreneurship", "Предпринимательство", "finance_business", 0.7, "work"),

    # Home & Lifestyle
    ("cooking", "Кулинария", "home_lifestyle", 0.7, "life"),
    ("gardening", "Садоводство", "home_lifestyle", 0.7, "life"),
    ("interior_design", "Дизайн интерьера", "home_lifestyle", 0.7, "life"),
    ("sustainability", "Устойчивый образ жизни", "home_lifestyle", 0.7, "life"),
]


def _node_source_text(slug: str, display_name: str) -> str:
    """Get enriched text for node embedding: display name + aliases from ontology."""
    onto_entry = SEMANTIC_ONTOLOGY.get(slug)
    if onto_entry and isinstance(onto_entry, dict):
        desc = onto_entry.get("enriched_text", "")
        if not desc:
            aliases = onto_entry.get("aliases", [])
            desc = f"{display_name}. " + ", ".join(aliases[:10])
        return desc
    return display_name


def _ensure_node_embeddings(db: Session) -> None:
    """Generate pgvector embeddings for all nodes that don't have them."""
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
    """Clear and regenerate all node embeddings."""
    count = db.query(InterestHierarchyNode).update({InterestHierarchyNode.embedding: None})
    db.commit()
    _ensure_node_embeddings(db)
    return count


def ensure_hierarchy_seeded(db: Session) -> None:
    """Initialize hierarchy from _HIERARCHY_SEED if not already seeded."""
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

    # Compute depth and materialized path
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
    """Get full ancestor chain for a node (root -> node)."""
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
    """
    Register pre-resolved tags for a user.
    
    IMPORTANT: This function expects that all slugs are ALREADY resolved and valid.
    It does NOT do any resolution or enrichment - that happens in WRITE phase (Celery).
    
    For each slug:
    1. Find node in hierarchy
    2. Register all ancestor nodes (propagate weight up the hierarchy)
    3. Store in user_interest_graph_weights
    
    Args:
        db: SQLAlchemy session
        user_id: User ID
        resolved_slugs: List of valid hierarchy slugs (pre-resolved)
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
            
            # Use maximum weight from node or existing (to preserve higher matches)
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
    """Get all graph weights for a user: {node_id: weight}."""
    rows = db.query(UserInterestGraphWeight).filter_by(user_id=user_id).all()
    return {row.node_id: row.weight for row in rows}
