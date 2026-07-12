import os
import logging

from flask import Blueprint, render_template, request, redirect, abort, jsonify, current_app
from flask_login import login_required, current_user

from data.ai import AIExtractedInterests
from data.knowledge_graph import KnowledgeNode, KnowledgeConnection

from data.interest import Interest
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
from app.ai.match_report import generate_match_report
from app.ai_profiler.contextual_adapter import get_contextual_adapter
from config import config
import redis
from app.db import get_db_session
from data.user import User
from data.ai import UserPersonalityProfile, UserSchwartzProfile
from data.behavior import UserBehaviorProfile
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

REDIS_CACHE_URL = getattr(config, 'REDIS_CACHE_URL', 'redis://localhost:6379/1')
cache_redis = redis.from_url(REDIS_CACHE_URL, decode_responses=True)

profile_bp = Blueprint("profile", __name__)


@profile_bp.route("/viewProfile", methods=["GET"])
def view_profile():
    """Отображает страницу профиля указанного пользователя по его ID.

    Args:
        user_id (int): ID пользователя из query-параметра `user_id`.

    Returns:
        flask.Response: HTML-страница профиля пользователя.
    """
    user_id: int = request.args.get("user_id")
    with get_db_session() as db_sess:
        interest = db_sess.query(Interest).filter(Interest.user_id == user_id)
        user = db_sess.query(User).get(user_id)
        if not user:
            abort(404)

        profile_personality = user.personality_profile
        schwartz_profile = db_sess.query(UserSchwartzProfile).filter_by(user_id=user_id).first()
        extracted_interests = db_sess.query(AIExtractedInterests).filter_by(user_id=user_id).first()
        behavior_profile = db_sess.query(UserBehaviorProfile).filter_by(user_id=user_id).first()

    return render_template(
        "view_profile.html",
        interest=interest,
        user=user,
        profile_personality=profile_personality,
        schwartz_profile=schwartz_profile,
        extracted_interests=extracted_interests,
        behavior_profile=behavior_profile,
    )


@profile_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """Страница профиля текущего пользователя.

    Returns:
        flask.Response: HTML-страница профиля пользователя.
    """
    with get_db_session() as db_sess:
        user = db_sess.query(User).get(current_user.id)
        if not user:
            abort(404)
        from data.favorite_interest import FavoriteInterest
        from sqlalchemy.orm import joinedload

        created_interests = db_sess.query(Interest).filter(Interest.user == current_user).all()

        favorite_ids = [
            fav.interest_id for fav in db_sess.query(FavoriteInterest)
            .filter(FavoriteInterest.user_id == current_user.id)
            .all()
        ]
        favorite_interests = (
            db_sess.query(Interest)
            .options(joinedload(Interest.user))
            .filter(Interest.id.in_(favorite_ids)).all()
            if favorite_ids else []
        )

        schwartz_profile = db_sess.query(UserSchwartzProfile).filter_by(user_id=current_user.id).first()
        extracted_interests = db_sess.query(AIExtractedInterests).filter_by(user_id=current_user.id).first()

        return render_template(
            "profile.html",
            title="Профиль",
            interest=created_interests,
            favorite_interests=favorite_interests,
            current_user=current_user,
            schwartz_profile=schwartz_profile,
            extracted_interests=extracted_interests,
        )


@profile_bp.route("/upload_avatar", methods=["POST"])
@login_required
def upload_avatar():
    """Загружает, обрезает до квадрата и сжимает аватар пользователя.

    Принимает файл через поле `photo`, конвертирует в WEBP 400x400.
    Удаляет старый аватар, если он не стандартный.

    Returns:
        flask.Response: JSON с путём к новому аватару или ошибкой.
    """
    if "photo" not in request.files:
        return jsonify({"success": False, "message": "Файл не найден"}), 400

    photo = request.files["photo"]
    if photo.filename == "":
        return jsonify({"success": False, "message": "Файл не выбран"}), 400

    filename = secure_filename(photo.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"png", "jpg", "jpeg", "webp"}:
        return jsonify({"success": False, "message": "Недопустимый формат"}), 400

    base_upload_folder = current_app.config["UPLOAD_FOLDER"]
    avatar_folder = os.path.join(base_upload_folder, "avatars")
    os.makedirs(avatar_folder, exist_ok=True)

    unique_filename = f"{uuid.uuid4().hex}.webp"
    photo_path = os.path.join(avatar_folder, unique_filename)

    try:
        with Image.open(photo) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            min_side = min(width, height)
            left = (width - min_side) / 2
            top = (height - min_side) / 2
            right = (width + min_side) / 2
            bottom = (height + min_side) / 2
            img = img.crop((left, top, right, bottom))

            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            img.save(photo_path, 'WEBP', quality=85, optimize=True)

    except Exception as e:
        return jsonify({"success": False, "message": f"Ошибка обработки: {str(e)}"}), 500

    relative_path = f"uploads/avatars/{unique_filename}"

    with get_db_session() as db_sess:
        user = db_sess.get(User, current_user.id)

        if user.image_path and 'default' not in user.image_path:
            old_full_path = os.path.join('static', user.image_path)
            if os.path.exists(old_full_path):
                try:
                    os.remove(old_full_path)
                except Exception:
                    pass

        user.image_path = relative_path
        db_sess.commit()

    return jsonify({"success": True, "image_path": f"/static/{relative_path}"})


@profile_bp.route("/process_profile", methods=["POST"])
@login_required
def process_profile():
    """Обновляет имя, информацию и цели связи пользователя из POST-формы.

    Returns:
        flask.Response: Перенаправление на страницу профиля.
    """
    with get_db_session() as db_sess:
        user = db_sess.get(User, current_user.id)
        if not user:
            abort(404)
        user.name = request.form.get("name", user.name)
        user.information = request.form.get("information", user.information)
        user.connection = request.form.get("connection", user.connection)
        db_sess.commit()
    return redirect("/profile")


@profile_bp.route("/settings", methods=["GET"])
@login_required
def settings():
    """Страница настроек пользователя.

    Returns:
        flask.Response: HTML-страница настроек.
    """
    return render_template("settings.html", current_user=current_user)


@profile_bp.route("/update_profile", methods=["POST"])
@login_required
def update_profile():
    """Сохраняет изменения профиля (имя, информация, пароль) через JSON-запрос.

    Returns:
        flask.Response: JSON с результатом сохранения.
    """
    data = request.json
    with get_db_session() as db_sess:
        user = db_sess.get(User, current_user.id)
        if not user:
            return jsonify({"success": False, "message": "Пользователь не найден"}), 404

        if "name" in data:
            user.name = data["name"]

        if "information" in data:
            user.information = data["information"]

        if "password" in data and data["password"]:
            user.set_password(data["password"])

        db_sess.commit()

    return jsonify({"success": True, "message": "Настройки сохранены"})


@profile_bp.route("/knowledge_graph", methods=["GET"])
@login_required
def knowledge_graph():
    """Отображает страницу графа знаний текущего пользователя.

    Загружает узлы и связи из БД и передаёт их в шаблон.

    Returns:
        flask.Response: HTML-страница с графом знаний.
    """
    from sqlalchemy.orm import joinedload

    with get_db_session() as db_sess:
        nodes_db = db_sess.query(KnowledgeNode).filter(
            KnowledgeNode.user_id == current_user.id
        ).all()

        connections_db = db_sess.query(KnowledgeConnection).join(
            KnowledgeNode, KnowledgeConnection.from_node_id == KnowledgeNode.id
        ).filter(KnowledgeNode.user_id == current_user.id).all()

        nodes = [
            {
                "id": n.id,
                "title": n.title,
                "description": n.description,
                "category": n.category,
                "x": n.x,
                "y": n.y,
            }
            for n in nodes_db
        ]

        connections = [
            {
                "id": c.id,
                "from": c.from_node_id,
                "to": c.to_node_id,
                "label": c.label,
            }
            for c in connections_db
        ]

    return render_template(
        "index.html",
        nodes=nodes,
        connections=connections,
        current_user=current_user
    )


@profile_bp.route("/knowledge_graph/node", methods=["POST"])
@login_required
def create_node():
    """Создаёт новый узел графа знаний для текущего пользователя.

    Returns:
        flask.Response: JSON с данными созданного узла.
    """
    from data.knowledge_graph import KnowledgeNode

    data = request.json
    with get_db_session() as db_sess:
        node = KnowledgeNode(
            user_id=current_user.id,
            title=data.get("title", "Новый узел"),
            description=data.get("description", ""),
            category=data.get("category", ""),
            x=data.get("x", 0),
            y=data.get("y", 0)
        )
        db_sess.add(node)
        db_sess.commit()

        return jsonify({"success": True, "node": {
            "id": node.id,
            "title": node.title,
            "description": node.description,
            "category": node.category,
            "x": node.x,
            "y": node.y
        }})


@profile_bp.route("/knowledge_graph/node/<int:node_id>", methods=["PUT", "DELETE"])
@login_required
def update_node(node_id):
    """Обновляет поля или удаляет узел графа знаний.

    При DELETE также удаляет все связи узла.

    Args:
        node_id (int): ID узла графа.

    Returns:
        flask.Response: JSON с результатом операции.
    """
    from data.knowledge_graph import KnowledgeNode, KnowledgeConnection

    with get_db_session() as db_sess:
        node = db_sess.query(KnowledgeNode).filter(
            KnowledgeNode.id == node_id,
            KnowledgeNode.user_id == current_user.id
        ).first()

        if not node:
            return jsonify({"success": False, "message": "Узел не найден"}), 404

        if request.method == "DELETE":
            db_sess.query(KnowledgeConnection).filter(
                (KnowledgeConnection.from_node_id == node_id) |
                (KnowledgeConnection.to_node_id == node_id)
            ).delete()
            db_sess.delete(node)
            db_sess.commit()
            return jsonify({"success": True})
        else:
            data = request.json
            if "title" in data:
                node.title = data["title"]
            if "description" in data:
                node.description = data["description"]
            if "category" in data:
                node.category = data["category"]
            if "x" in data:
                node.x = data["x"]
            if "y" in data:
                node.y = data["y"]

            db_sess.commit()
            return jsonify({"success": True, "node": {
                "id": node.id,
                "title": node.title,
                "description": node.description,
                "category": node.category,
                "x": node.x,
                "y": node.y
            }})


@profile_bp.route("/knowledge_graph/connection", methods=["POST", "DELETE"])
@login_required
def manage_connection():
    """Создаёт или удаляет связь между узлами графа знаний.

    POST — создаёт новую связь, DELETE — удаляет существующую по connection_id.

    Returns:
        flask.Response: JSON с результатом операции.
    """
    from data.knowledge_graph import KnowledgeNode, KnowledgeConnection

    with get_db_session() as db_sess:
        if request.method == "POST":
            data = request.json
            from_id = data.get("from_node_id")
            to_id = data.get("to_node_id")

            from_node = db_sess.query(KnowledgeNode).filter(
                KnowledgeNode.id == from_id,
                KnowledgeNode.user_id == current_user.id
            ).first()
            to_node = db_sess.query(KnowledgeNode).filter(
                KnowledgeNode.id == to_id,
                KnowledgeNode.user_id == current_user.id
            ).first()

            if not from_node or not to_node:
                return jsonify({"success": False, "message": "Узлы не найдены"}), 404

            existing = db_sess.query(KnowledgeConnection).filter(
                KnowledgeConnection.from_node_id == from_id,
                KnowledgeConnection.to_node_id == to_id
            ).first()

            if existing:
                return jsonify({"success": False, "message": "Связь уже существует"}), 400

            connection = KnowledgeConnection(
                from_node_id=from_id,
                to_node_id=to_id,
                label=data.get("label", "")
            )
            db_sess.add(connection)
            db_sess.commit()

            return jsonify({"success": True, "connection": {
                "id": connection.id,
                "from_node_id": connection.from_node_id,
                "to_node_id": connection.to_node_id,
                "label": connection.label
            }})
        else:
            connection_id = request.json.get("connection_id")
            connection = db_sess.query(KnowledgeConnection).join(
                KnowledgeNode, KnowledgeConnection.from_node_id == KnowledgeNode.id
            ).filter(
                KnowledgeConnection.id == connection_id,
                KnowledgeNode.user_id == current_user.id
            ).first()

            if not connection:
                return jsonify({"success": False, "message": "Связь не найдена"}), 404

            db_sess.delete(connection)
            db_sess.commit()
            return jsonify({"success": True})


@profile_bp.route("/knowledge_graph_data")
@login_required
def get_graph_data():
    """Возвращает JSON-структуру графа знаний текущего пользователя.

    Returns:
        flask.Response: JSON с массивами узлов и связей.
    """
    with get_db_session() as db_sess:
        nodes_db = db_sess.query(KnowledgeNode).filter(
            KnowledgeNode.user_id == current_user.id
        ).all()

        connections_db = db_sess.query(KnowledgeConnection).join(
            KnowledgeNode, KnowledgeConnection.from_node_id == KnowledgeNode.id
        ).filter(KnowledgeNode.user_id == current_user.id).all()

        nodes = [
            {
                "id": n.id,
                "title": n.title,
                "description": n.description,
                "category": n.category,
                "x": n.x,
                "y": n.y,
            }
            for n in nodes_db
        ]

        connections = [
            {
                "id": c.id,
                "from": c.from_node_id,
                "to": c.to_node_id,
                "label": c.label,
            }
            for c in connections_db
        ]
    return jsonify({"nodes": nodes, "connections": connections})


CATEGORY_CONFIG = {
    'work': {
        'weights': [0.3, 1.0, 0.5, 0.2, 0.8],
        'complementary': [2]
    },
    'hobby': {
        'weights': [1.0, 0.2, 0.8, 0.5, 0.1],
        'complementary': []
    },
    'psychology': {
        'weights': [0.7, 0.4, 0.4, 1.0, 0.6],
        'complementary': []
    }
}


def _serialize_extracted_interests(row: AIExtractedInterests) -> dict:
    """Преобразует ORM-строку AIExtractedInterests в плоский словарь для скоринга.

    Args:
        row: ORM-объект с извлечёнными интересами пользователя.

    Returns:
        Словарь с полями hobbies, topics, skills, occupation, semantic_categories.
    """
    return {
        "hobbies": row.hobbies or [],
        "topics": row.topics or [],
        "skills": row.skills or [],
        "occupation": row.occupation,
        "semantic_categories": getattr(row, "semantic_categories", None) or {},
        "extraction_method": getattr(row, "extraction_method", None),
    }


def _ocean_vector(profile: UserPersonalityProfile | None) -> list[float] | None:
    """Извлекает вектор OCEAN из профиля личности.

    Args:
        profile: Профиль личности пользователя.

    Returns:
        Список из 5 значений OCEAN или None, если профиль отсутствует.
    """
    if profile is None:
        return None
    return [
        profile.openness,
        profile.conscientiousness,
        profile.extraversion,
        profile.agreeableness,
        profile.neuroticism,
    ]


def _get_related_node_ids(db_sess, target_node_id: int) -> list[int]:
    """Собирает ID целевого узла иерархии и всех его родительских и дочерних узлов.

    Args:
        db_sess: Сессия SQLAlchemy.
        target_node_id: ID узла InterestHierarchyNode.

    Returns:
        Список ID узлов (целевой + родители + потомки).
    """
    from data.interest_hierarchy import InterestHierarchyNode
    
    related_ids = {target_node_id}
    
    target = db_sess.query(InterestHierarchyNode).filter_by(id=target_node_id).first()
    if not target:
        return list(related_ids)
    
    current = target
    while current.parent_id:
        related_ids.add(current.parent_id)
        current = db_sess.query(InterestHierarchyNode).filter_by(id=current.parent_id).first()
        if not current:
            break
    
    def _add_children(node_id: int):
        children = db_sess.query(InterestHierarchyNode).filter_by(parent_id=node_id).all()
        for child in children:
            related_ids.add(child.id)
            _add_children(child.id)
    
    _add_children(target_node_id)
    
    return list(related_ids)


def _build_hierarchy_cache(db_sess) -> dict:
    """Загружает всю иерархию интересов в память для O(1)-доступа при скоринге.

    Args:
        db_sess: Сессия SQLAlchemy.

    Returns:
        Словарь {node_id: {slug, depth, parent_id, match_weight, name}}.
    """
    from data.interest_hierarchy import InterestHierarchyNode
    
    cache = {}
    all_nodes = db_sess.query(InterestHierarchyNode).all()
    
    for node in all_nodes:
        cache[node.id] = {
            'slug': node.slug,
            'depth': node.depth,
            'parent_id': node.parent_id,
            'match_weight': node.match_weight or 1.0,
            'name': node.name,
        }
    
    return cache


def _calculate_graph_interest_score(
    target_node,
    matched_weights_dict: dict,
    hierarchy_cache: dict,
) -> tuple[float, list[str]]:
    """Вычисляет скоринговый балл пересечения по графу интересов.

    Прямое совпадение (точный slug): coeff = 1.0
    Косвенное совпадение (родитель/потомок): coeff = 0.4 с понижением за глубину.

    Args:
        target_node: Целевой узел InterestHierarchyNode.
        matched_weights_dict: {node_id: weight} совпавшие узлы пользователя.
        hierarchy_cache: Предзагруженный кэш иерархии.

    Returns:
        Кортеж (score, matched_tags_list): балл совместимости и список тегов.
    """
    if not matched_weights_dict:
        return 0.0, []
    
    score = 0.0
    max_possible = 0.0
    matched_tags = []
    
    for matched_node_id, weight in matched_weights_dict.items():
        if matched_node_id not in hierarchy_cache:
            continue
        
        matched_node_data = hierarchy_cache[matched_node_id]
        matched_slug = matched_node_data['slug']
        
        if matched_node_id == target_node.id:
            coeff = 1.0
            score += weight * coeff
            matched_tags.append(f"{matched_slug} (точное)")
        else:
            depth_diff = abs(matched_node_data['depth'] - target_node.depth)
            coeff = max(0.4 - (0.05 * depth_diff), 0.1)
            
            score += weight * coeff
            matched_tags.append(f"{matched_slug} (похоже)")
        max_possible += coeff
    
    final_score = score / max_possible if max_possible > 0 else 0.0
    final_score = min(final_score, 1.0)
    
    return final_score, matched_tags


@profile_bp.route("/api/graph/match/<int:node_id>")
@login_required
def match_by_node(node_id: int):
    """Ищет наиболее совместимых пользователей по узлу графа интересов.

    READ-фаза: один SQL-запрос + быстрый цикл скоринга.
    WRITE-фаза (Celery): теги предварительно разрешены в slugs.

    Алгоритм:
    1. Получить целевой узел из иерархии (или через KnowledgeNode).
    2. Один SQL-запрос для всех кандидатов с весами.
    3. Предзагрузить кэш иерархии (slug, depth, parent_id).
    4. Рассчитать graph_score с прямыми/косвенными совпадениями.
    5. Смешать с personality_score через root-категорию.
    6. Вернуть топ-10 результатов.

    Args:
        node_id: ID узла InterestHierarchyNode (или KnowledgeNode).

    Returns:
        flask.Response: JSON со списком лучших совпадений.
    """
    from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight
    from data.user import User as DBUser
    from app.ai_profiler.dynamic_enrichment import get_tag_enricher
    from app.ai_profiler.root_personalities import (
        find_root_category, compute_root_personality_score, ROOT_ARCHETYPES,
        KNOWLEDGE_CATEGORY_TO_ROOT,
    )

    with get_db_session() as db_sess:
        target_node = db_sess.query(InterestHierarchyNode).filter_by(id=node_id).first()
        kn_category = None

        if not target_node:
            knowledge_node = db_sess.query(KnowledgeNode).filter_by(id=node_id).first()
            if knowledge_node:
                kn_category = (knowledge_node.category or "").lower()
                title = knowledge_node.title.lower()
                try:
                    enricher = get_tag_enricher()
                    slug = enricher.resolve_tag_to_slug(db_sess, title, fallback_to_enrichment=False)
                    if slug:
                        target_node = db_sess.query(InterestHierarchyNode).filter_by(slug=slug).first()
                except Exception:
                    logger.warning(f"[match_by_node] SBERT resolve failed, trying keyword match for '{title}'")

                if not target_node:
                    from sqlalchemy import or_
                    fuzzy = db_sess.query(InterestHierarchyNode).filter(
                        or_(
                            InterestHierarchyNode.slug.ilike(f"%{title}%"),
                            InterestHierarchyNode.name.ilike(f"%{title}%"),
                        )
                    ).first()
                    if fuzzy:
                        target_node = fuzzy
                        logger.info(f"[match_by_node] Keyword fallback matched node '{title}' -> {target_node.slug}")

        if not target_node:
            logger.warning(f"[match_by_node] Node {node_id} not found in hierarchy")
            return jsonify({"error": "Node not found"}), 404
        
        logger.debug(f"[match_by_node] Target node: {target_node.slug} (id={node_id}, depth={target_node.depth})")
        
        hierarchy_cache = _build_hierarchy_cache(db_sess)

        if kn_category and kn_category in KNOWLEDGE_CATEGORY_TO_ROOT:
            root_category = KNOWLEDGE_CATEGORY_TO_ROOT[kn_category]
        else:
            root_category = find_root_category(target_node, hierarchy_cache)
        root_archetype = ROOT_ARCHETYPES.get(root_category, {})
        personality_blend_weight = config.ROOT_PERSONALITY_BLEND_WEIGHT
        chain = []
        cid = target_node.id
        while cid in hierarchy_cache:
            chain.append(f"{hierarchy_cache[cid]['slug']}(id={cid})")
            pid = hierarchy_cache[cid].get("parent_id")
            if pid is None:
                break
            cid = pid
        logger.info(
            "[match_by_node] Node %s kn_category=%s → root_category=%s, chain: %s",
            target_node.slug, kn_category, root_category, " ← ".join(reversed(chain)),
        )
        
        from sqlalchemy import or_, text
        
        target_and_related_ids = _get_related_node_ids(db_sess, target_node.id)
        
        candidates_query = db_sess.query(
            UserInterestGraphWeight.user_id,
            DBUser.id,
            DBUser.name,
            UserInterestGraphWeight.node_id,
            UserInterestGraphWeight.weight,
        ).join(
            DBUser, UserInterestGraphWeight.user_id == DBUser.id
        ).filter(
            UserInterestGraphWeight.user_id != current_user.id,
            UserInterestGraphWeight.node_id.in_(target_and_related_ids),
            UserInterestGraphWeight.weight > 0.0,
        ).all()
        
        if not candidates_query:
            logger.debug("[match_by_node] No candidates with graph weights found")
            return jsonify([])
        
        candidates_by_user = {}
        for row in candidates_query:
            user_id, db_user_id, user_name, node_id_match, weight = row
            if user_id not in candidates_by_user:
                candidates_by_user[user_id] = {
                    "name": user_name,
                    "matched_weights": {},
                }
            candidates_by_user[user_id]["matched_weights"][node_id_match] = weight
        
        logger.debug(f"[match_by_node] Found {len(candidates_by_user)} unique candidates")
        
        candidate_user_ids = list(candidates_by_user.keys())
        personality_profiles = {
            p.user_id: p
            for p in db_sess.query(UserPersonalityProfile)
            .filter(UserPersonalityProfile.user_id.in_(candidate_user_ids))
            .all()
        }
        schwartz_profiles = {
            s.user_id: s
            for s in db_sess.query(UserSchwartzProfile)
            .filter(UserSchwartzProfile.user_id.in_(candidate_user_ids))
            .all()
        }
        logger.info(
            "[match_by_node] Profiles loaded: %d candidates, %d personality, %d schwartz",
            len(candidate_user_ids), len(personality_profiles), len(schwartz_profiles),
        )
        
        matches = []
        for candidate_user_id, candidate_data in candidates_by_user.items():
            candidate_name = candidate_data["name"]
            matched_weights_dict = candidate_data["matched_weights"]
            
            graph_score, matched_tags = _calculate_graph_interest_score(
                target_node=target_node,
                matched_weights_dict=matched_weights_dict,
                hierarchy_cache=hierarchy_cache,
            )
            
            if not matched_tags or graph_score < 0.1:
                continue
            
            personality_score = compute_root_personality_score(
                personality_profile=personality_profiles.get(candidate_user_id),
                schwartz_profile=schwartz_profiles.get(candidate_user_id),
                root_category=root_category,
            )
            
            final_score = (
                (1.0 - personality_blend_weight) * graph_score
                + personality_blend_weight * personality_score
            )
            
            match_reason = root_archetype.get("match_reason", "Общие интересы")
            
            matches.append({
                "user_id": candidate_user_id,
                "user_name": candidate_name,
                "compatibility": round(final_score * 100, 1),
                "match_reason": match_reason,
                "matched_tags": matched_tags,
                "root_category": root_category,
                "score_breakdown": {
                    "graph_interest": round(graph_score, 3),
                    "personality_fit": round(personality_score, 3),
                    "blend_weight": personality_blend_weight,
                },
            })
        
        matches.sort(key=lambda x: x["compatibility"], reverse=True)
        top_matches = matches[:10]
        
        if top_matches:
            logger.info(
                "[match_by_node] Top match: user=%s graph=%.3f personality=%.3f final=%.1f%%",
                top_matches[0]["user_name"],
                top_matches[0]["score_breakdown"]["graph_interest"],
                top_matches[0]["score_breakdown"]["personality_fit"],
                top_matches[0]["compatibility"],
            )
        logger.info("[match_by_node] Returning %d matches for node %d", len(top_matches), node_id)
        return jsonify(top_matches)



@profile_bp.route("/api/graph/report/<int:target_user_id>")
@login_required
def get_match_report(target_user_id):
    """Эндпоинт ленивой загрузки AI-отчёта о совместимости с кэшированием в Redis.

    Args:
        target_user_id: ID целевого пользователя для сравнения.

    Returns:
        flask.Response: JSON с текстом отчёта и флагом кэширования.
    """
    node_id = request.args.get('node_id', type=int)

    cache_key = f"nexus:match_report:{current_user.id}:{target_user_id}:{node_id or 'none'}"

    cached_report = cache_redis.get(cache_key)
    if cached_report:
        return jsonify({"report": cached_report, "cached": True})

    with get_db_session() as db_sess:
        target_user = db_sess.query(User).get(target_user_id)
        if not target_user:
            return jsonify({"error": "User not found"}), 404

        matched_tags = []
        if node_id:
            node = db_sess.query(KnowledgeNode).get(node_id)
            if node:
                matched_tags.append(node.title)

        try:
            report = generate_match_report(
                current_user.id,
                target_user_id,
                db_sess,
                matched_tags=matched_tags,
                use_llm=None
            )

            if report:
                cache_redis.setex(cache_key, 86400, report)

            return jsonify({"report": report, "cached": False})

        except Exception as e:
            current_app.logger.error(f"Error in match report generation: {e}")
            fallback = generate_match_report(
                current_user.id,
                target_user_id,
                db_sess,
                matched_tags=matched_tags,
                use_llm=False,
            )
            return jsonify({"report": fallback, "cached": False, "fallback": True})