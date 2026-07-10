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
from app.ai.matching_engine import calculate_multidimensional_compatibility
from app.ai_profiler.contextual_adapter import get_contextual_adapter
from config import config
import redis
import config  # Единый модуль конфигурации из корня проекта
from app.db import get_db_session  # Контекстный менеджер БД
from data.user import User
from data.ai import UserPersonalityProfile, UserSchwartzProfile
from data.behavior import UserBehaviorProfile
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

# Инициализация Redis для кэша (используем БД 1, чтобы не мешать Celery в БД 0)
# Если в config нет REDIS_CACHE_URL, фоллбек на локальный
REDIS_CACHE_URL = getattr(config, 'REDIS_CACHE_URL', 'redis://localhost:6379/1')
cache_redis = redis.from_url(REDIS_CACHE_URL, decode_responses=True)

profile_bp = Blueprint("profile", __name__)


@profile_bp.route("/viewProfile", methods=["GET"])
def view_profile():
    """Возвращает страницу профиля пользователя.

    Returns:
        flask.Response: HTML-страница с профилем пользователя.
    """
    user_id: int = request.args.get("user_id")
    with get_db_session() as db_sess:
        interest = db_sess.query(Interest).filter(Interest.user_id == user_id)
        user = db_sess.query(User).get(user_id)
        if not user:
            abort(404)
    return render_template("view_profile.html", interest=interest, user=user)


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
        return render_template(
            "profile.html",
            title="Профиль",
            interest=created_interests,
            favorite_interests=favorite_interests,
            current_user=current_user
        )


@profile_bp.route("/upload_avatar", methods=["POST"])
@login_required
def upload_avatar():
    """Загрузка и обработка аватара пользователя.

    Returns:
        flask.Response: JSON с результатом выполнения.
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
    """Обновление данных профиля пользователя из формы.

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
    """Обработка и сохранение изменений профиля через JSON-запрос.

    Returns:
        flask.Response: JSON-ответ о результатах сохранения.
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
    """Страница отображения графа знаний пользователя.

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
    """Создание нового узла графа знаний.

    Returns:
        flask.Response: JSON-ответ с информацией о созданном узле.
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
    """Обновление или удаление узла графа знаний.

    Args:
        node_id (int): ID узла графа.

    Returns:
        flask.Response: JSON-ответ об успешном завершении действия.
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
    """Создание или удаление связи между узлами графа знаний.

    Returns:
        flask.Response: JSON-ответ с результатом.
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
    """Возвращает JSON-структуру графа знаний пользователя.

    Returns:
        flask.Response: JSON с массивом узлов и связей графа.
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
    """
    ORM-строка AIExtractedInterests -> dict в формате, который понимает
    AIProfiler.calculate_hybrid_matching_score / _extract_tag_set.

    "semantic_categories" и "extraction_method" — новые JSON-колонки, добавленные
    вместе с ZeroShotInterestExtractor/CustomInterestClassifier. Для старых строк
    (созданных до внедрения) они могут быть NULL — _extract_tag_set в core.py
    сам делает fallback на hobbies/skills/topics в этом случае, так что здесь
    достаточно отдать None/пустой dict, ничего дополнительно обрабатывать не нужно.
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
    """
    Get target node ID plus all its parent and child node IDs.
    
    This determines the scope of the search: direct match + related hierarchy.
    
    Returns: list of InterestHierarchyNode IDs
    """
    from data.interest_hierarchy import InterestHierarchyNode
    
    related_ids = {target_node_id}
    
    # Get target node
    target = db_sess.query(InterestHierarchyNode).filter_by(id=target_node_id).first()
    if not target:
        return list(related_ids)
    
    # Add all parent nodes
    current = target
    while current.parent_id:
        related_ids.add(current.parent_id)
        current = db_sess.query(InterestHierarchyNode).filter_by(id=current.parent_id).first()
        if not current:
            break
    
    # Add all child nodes (recursive)
    def _add_children(node_id: int):
        children = db_sess.query(InterestHierarchyNode).filter_by(parent_id=node_id).all()
        for child in children:
            related_ids.add(child.id)
            _add_children(child.id)
    
    _add_children(target_node_id)
    
    return list(related_ids)


def _build_hierarchy_cache(db_sess) -> dict:
    """
    Pre-load entire interest hierarchy into memory for O(1) lookups during scoring.
    
    Returns: {node_id: {
        'slug': str,
        'depth': int,
        'parent_id': int | None,
        'match_weight': float,
        'name': str
    }, ...}
    
    This cache eliminates all DB lookups in the scoring loop.
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
    
    logger.debug(f"[_build_hierarchy_cache] Cached {len(cache)} nodes")
    return cache


def _calculate_graph_interest_score(
    target_node,
    matched_weights_dict: dict,
    hierarchy_cache: dict,
) -> tuple[float, list[str]]:
    """
    Calculate graph interest score with direct/indirect distinction.
    
    Direct Match: User's slug exactly matches target_node.slug → coeff = 1.0
    Indirect Match: User's slug is parent/child of target_node → coeff = 0.4 (with depth penalty)
    
    Args:
        target_node: InterestHierarchyNode instance
        matched_weights_dict: {node_id: weight, ...} - matched nodes for this user
        hierarchy_cache: pre-loaded hierarchy dict
    
    Returns: (score, matched_tags_list)
        - score: float (0..1)
        - matched_tags_list: list of matched tag slugs to display
    """
    if not matched_weights_dict:
        return 0.0, []
    
    score = 0.0
    matched_tags = []
    
    for matched_node_id, weight in matched_weights_dict.items():
        if matched_node_id not in hierarchy_cache:
            continue
        
        matched_node_data = hierarchy_cache[matched_node_id]
        matched_slug = matched_node_data['slug']
        
        # Determine match type (direct vs indirect)
        if matched_node_id == target_node.id:
            # Direct match: exact node ID
            coeff = 1.0
            score += weight * coeff
            matched_tags.append(f"{matched_slug} (точное)")
        else:
            # Indirect match: parent or child relationship
            # Depth penalty: the further apart in hierarchy, the less relevant
            depth_diff = abs(matched_node_data['depth'] - target_node.depth)
            indirect_coeff = max(0.4 - (0.05 * depth_diff), 0.1)  # Min 0.1, decay with depth
            
            score += weight * indirect_coeff
            matched_tags.append(f"{matched_slug} (похоже)")
    
    # Normalize score to 0..1 range
    final_score = min(score, 1.0)
    
    return final_score, matched_tags


@profile_bp.route("/api/graph/match/<int:node_id>")
@login_required
def match_by_node(node_id: int):
    """
    Search for the best matching users by graph node (clean architecture).
    
    NEW ARCHITECTURE (Write/Read Separation):
    - WRITE phase (Celery): All user tags are pre-resolved to slugs and stored in user_interest_graph_weights
    - READ phase (this function): Single SQL query + ultra-fast scoring loop
    
    Flow:
    1. Get target_node from hierarchy (node_id maps to InterestHierarchyNode.id)
    2. Execute ONE SQL query to fetch all candidates with matching weights
    3. Pre-load hierarchy cache (slug, depth, parent_id) for scoring
    4. Loop through candidates, calculate graph_score with direct/indirect distinction
    5. Return top-10 with non-empty matched_tags
    
    Performance: Single SQL query, no N+1 issues, no SBERT calls in read phase.
    """
    from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight
    from data.user import User as DBUser
    
    with get_db_session() as db_sess:
        # Step 1: Get target hierarchy node
        target_node = db_sess.query(InterestHierarchyNode).filter_by(id=node_id).first()
        if not target_node:
            logger.warning(f"[match_by_node] Node {node_id} not found in hierarchy")
            return jsonify({"error": "Node not found"}), 404
        
        logger.debug(f"[match_by_node] Target node: {target_node.slug} (id={node_id}, depth={target_node.depth})")
        
        # Step 2: SINGLE SQL query - fetch all candidates with graph weights for target node or its parents/children
        #
        # Query finds users who have:
        # - Direct match: target_node.id
        # - Parent match: any parent of target_node
        # - Child match: any child of target_node
        
        from sqlalchemy import or_, text
        
        # Get target node's ancestor and descendant IDs
        target_and_related_ids = _get_related_node_ids(db_sess, node_id)
        
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
        
        # Step 3: Pre-load entire hierarchy as cache for O(1) lookups
        hierarchy_cache = _build_hierarchy_cache(db_sess)
        
        # Group candidates by user_id for aggregate scoring
        candidates_by_user = {}
        for row in candidates_query:
            user_id, db_user_id, user_name, node_id_match, weight = row
            if user_id not in candidates_by_user:
                candidates_by_user[user_id] = {
                    "name": user_name,
                    "matched_weights": {},  # node_id -> weight
                }
            candidates_by_user[user_id]["matched_weights"][node_id_match] = weight
        
        logger.debug(f"[match_by_node] Found {len(candidates_by_user)} unique candidates")
        
        # Step 4: Calculate scores for each candidate (no DB hits)
        matches = []
        for candidate_user_id, candidate_data in candidates_by_user.items():
            candidate_name = candidate_data["name"]
            matched_weights_dict = candidate_data["matched_weights"]
            
            # Calculate graph score with direct/indirect distinction
            graph_score, matched_tags = _calculate_graph_interest_score(
                target_node=target_node,
                matched_weights_dict=matched_weights_dict,
                hierarchy_cache=hierarchy_cache,
            )
            
            # Skip if no relevant tags matched
            if not matched_tags or graph_score < 0.1:
                continue
            
            matches.append({
                "user_id": candidate_user_id,
                "user_name": candidate_name,
                "compatibility": round(graph_score * 100, 1),
                "match_reason": "Общие интересы",
                "matched_tags": matched_tags,
                "score_breakdown": {
                    "graph_interest": round(graph_score, 3),
                },
            })
        
        # Step 5: Sort and return top-10
        matches.sort(key=lambda x: x["compatibility"], reverse=True)
        top_matches = matches[:10]
        
        logger.info(f"[match_by_node] Returning {len(top_matches)} matches for node {node_id}")
        return jsonify(top_matches)



@profile_bp.route("/api/graph/report/<int:target_user_id>")
@login_required
def get_match_report(target_user_id):
    """
    Эндпоинт ленивой загрузки AI-отчета.
    Кэширует результат в Redis.
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

        # ИСПРАВЛЕНО: Вытаскиваем название узла, на который кликнул пользователь,
        # чтобы передать его как matched_tags вместо пустого списка.
        matched_tags = []
        if node_id:
            node = db_sess.query(KnowledgeNode).get(node_id)
            if node:
                matched_tags.append(node.title)

        try:
            # ИСПРАВЛЕНО: use_llm=None, чтобы функция match_report сама взяла
            # актуальное значение config.OLLAMA_ENABLED, не ломаясь об getattr
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