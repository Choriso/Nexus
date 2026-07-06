import os

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
from app.db import get_db_session  # Контекстный менеджер БД
from data.user import User
from data.ai import UserPersonalityProfile, UserSchwartzProfile
from data.behavior import UserBehaviorProfile
from sqlalchemy.orm import joinedload

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


@profile_bp.route("/api/graph/match/<int:node_id>")
@login_required
def match_by_node(node_id):
    """
    Поиск наиболее подходящих пользователей по выбранному узлу графа.
    ОПТИМИЗИРОВАНО: Убраны тяжелые вызовы нейросети на лету, убраны мутирующие запросы (register_user_tags),
    исправлен N+1 и двойные сессии БД.
    """

    # Оставляем ОДИН контекстный менеджер для всего эндпоинта
    with get_db_session() as db_sess:
        raw_cat = request.args.get('category', 'psychology').lower().strip()
        cat_mapping = {
            'работа': 'work', 'work': 'work',
            'хобби': 'hobby', 'hobby': 'hobby',
            'психология': 'psychology', 'psychology': 'psychology'
        }
        cat_type = cat_mapping.get(raw_cat, 'psychology')
        config_data = CATEGORY_CONFIG.get(cat_type, CATEGORY_CONFIG['psychology'])

        from app.ai_profiler import get_profiler
        from data.ai import AIExtractedInterests, UserCompatibility, UserPersonalityProfile, UserSchwartzProfile
        from data.behavior import UserBehaviorProfile
        from data.knowledge_graph import KnowledgeNode
        from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight
        from app.ai_profiler.interest_graph import build_query_weights

        profiler = get_profiler()
        contextual_adapter = get_contextual_adapter(
            enabled=getattr(config, "CONTEXTUAL_ADAPTER_ENABLED", True)
        )

        target_node = db_sess.query(KnowledgeNode).get(node_id)
        if not target_node:
            return jsonify({"error": "Node not found"}), 404

        # Обогащаем поисковый запрос через быстрый ContextualAdapter
        raw_search_query = f"{target_node.title}. {target_node.description or ''}".strip()
        enrichment = contextual_adapter.enrich_text(raw_search_query)
        search_query = enrichment.enriched

        # ОПТИМИЗАЦИЯ: Вместо вызова тяжелого ИИ экстрактора в рантайме,
        # собираем теги, которые ContextualAdapter уже вытащил по словарям и таксономии!
        raw_tags = enrichment.matched_concepts + enrichment.subcategories

        # Принудительно очищаем, убираем пробелы и переводим в lowercase
        query_tags = set()
        for t in raw_tags:
            clean_t = str(t).lower().strip()
            # Защита: не пускаем системные категории в теги
            if clean_t not in ['work', 'hobby', 'psychology', 'работа', 'хобби', 'психология']:
                query_tags.add(clean_t)

        # Если адаптер ничего не вытащил, берем название самого узла
        if not query_tags and target_node.title:
            node_title_clean = target_node.title.lower().strip()
            if node_title_clean not in ['work', 'hobby', 'psychology']:
                query_tags.add(node_title_clean)

        # Получаем узлы других пользователей с ЛИМИТОМ (например, топ-100 кандидатов для стабильности)
        candidates = db_sess.query(KnowledgeNode).options(
            joinedload(KnowledgeNode.user)
        ).filter(
            KnowledgeNode.user_id != current_user.id
        ).limit(100).all()

        if not candidates:
            return jsonify([])

        user_ids = {node.user_id for node in candidates}

        # БАТЧ-ЗАПРОСЫ (Защита от N+1)
        profiles = db_sess.query(UserPersonalityProfile).filter(
            UserPersonalityProfile.user_id.in_(user_ids)
        ).all()
        profile_map = {p.user_id: p for p in profiles}

        extracted_rows = db_sess.query(AIExtractedInterests).filter(
            AIExtractedInterests.user_id.in_(user_ids)
        ).all()

        extracted_map = {row.user_id: row for row in extracted_rows}

        compat_rows = db_sess.query(UserCompatibility).filter(
            ((UserCompatibility.user_id_1 == current_user.id) & (UserCompatibility.user_id_2.in_(user_ids))) |
            ((UserCompatibility.user_id_2 == current_user.id) & (UserCompatibility.user_id_1.in_(user_ids)))
        ).all()

        compat_map = {}
        for row in compat_rows:
            other_id = row.user_id_2 if row.user_id_1 == current_user.id else row.user_id_1
            compat_map[other_id] = row.overall_score

        my_profile = db_sess.query(UserPersonalityProfile).filter_by(user_id=current_user.id).first()
        my_vec = [0.0] * 5
        if my_profile:
            my_vec = [
                my_profile.openness, my_profile.conscientiousness, my_profile.extraversion,
                my_profile.agreeableness, my_profile.neuroticism
            ]

        schwartz_rows = db_sess.query(UserSchwartzProfile).filter(
            UserSchwartzProfile.user_id.in_(user_ids | {current_user.id})
        ).all()
        schwartz_map = {row.user_id: row for row in schwartz_rows}
        my_schwartz = schwartz_map.get(current_user.id)

        behavior_rows = db_sess.query(UserBehaviorProfile).filter(
            UserBehaviorProfile.user_id.in_(user_ids | {current_user.id})
        ).all()
        behavior_map = {row.user_id: row for row in behavior_rows}
        my_behavior = behavior_map.get(current_user.id)

        # Preload hierarchy node names
        hierarchy_nodes = db_sess.query(InterestHierarchyNode).all()
        hierarchy_node_names = {n.id: n.name for n in hierarchy_nodes}

        # Preload graph weights из базы (БЕЗ динамической регистрации register_user_tags на лету)
        graph_weights_rows = db_sess.query(UserInterestGraphWeight).filter(
            UserInterestGraphWeight.user_id.in_(user_ids | {current_user.id})
        ).all()
        graph_weights_map = {}
        for row in graph_weights_rows:
            graph_weights_map.setdefault(row.user_id, {})[row.node_id] = row.weight

        # Pre-build query weights на основе вытащенных тегов
        query_graph_weights = build_query_weights(db_sess, query_tags)

        # ДОБАВЛЕНО: Регистрируем теги текущего пользователя один раз, если их еще нет в базе
        if query_tags and current_user.id not in graph_weights_map:
            from app.ai_profiler.interest_graph import register_user_tags
            # Фоновая асинхронная задача подошла бы лучше, но пока просто выносим из цикла
            register_user_tags(db_sess, current_user.id, list(query_tags))
            # Обновляем мапу весов для текущего юзера на всякий случай
            graph_weights_map[current_user.id] = build_query_weights(db_sess, query_tags)

        matches = []
        for node in candidates:
            candidate_user_id = node.user_id
            if not candidate_user_id:
                continue

            other_profile = profile_map.get(candidate_user_id)
            other_extracted = extracted_map.get(candidate_user_id)

            # Big Five / OCEAN (базовый компонент 50%)
            ocean_score_val = compat_map.get(candidate_user_id)
            if ocean_score_val is not None:
                ocean_score_normalized = (
                    float(ocean_score_val)
                    if float(ocean_score_val) <= 1.0
                    else float(ocean_score_val) / 100.0
                )
            elif my_profile and other_profile:
                other_vec = [
                    other_profile.openness, other_profile.conscientiousness, other_profile.extraversion,
                    other_profile.agreeableness, other_profile.neuroticism
                ]
                processed_other_vec = other_vec[:]
                for c_idx in config_data['complementary']:
                    processed_other_vec[c_idx] = 1.0 - other_vec[c_idx]

                ocean_score_normalized = profiler.calculate_compatibility(
                    my_vec, processed_other_vec, weights=config_data['weights']
                ) / 100.0
            else:
                ocean_score_normalized = 0.5

            # Вызываем расчет движка (он теперь мгновенный, т.к. SBERT закэширован как синглтон)
            score_dict = calculate_multidimensional_compatibility(
                db_sess,
                ocean_score_normalized=ocean_score_normalized,
                query_tags=query_tags,
                current_user_id=current_user.id,
                other_user_id=candidate_user_id,
                other_extracted=other_extracted,
                my_schwartz=my_schwartz,
                other_schwartz=schwartz_map.get(candidate_user_id),
                my_behavior=my_behavior,
                other_behavior=behavior_map.get(candidate_user_id),
                query_graph_weights=query_graph_weights,
                other_graph_weights=graph_weights_map.get(candidate_user_id, None),
                hierarchy_node_names=hierarchy_node_names,
            )
            final_score = score_dict.get("final_score", 0.0)
            matched_tags = score_dict.get("matched_tags", [])

            other_user_name = node.user.name if node.user else f"Пользователь #{candidate_user_id}"

            reason = "Похожие интересы"
            if cat_type == 'work' and my_profile and other_profile:
                if abs(my_vec[2] - other_profile.extraversion) > 0.4:
                    reason = "Дополняет вашу команду"

            matches.append({
                "user_id": candidate_user_id,
                "user_name": other_user_name,
                "node_title": node.title,
                "compatibility": round(final_score * 100, 1),
                "category": node.category,
                "match_reason": reason,
                "matched_tags": matched_tags,
                "score_breakdown": {
                    "big_five": score_dict.get("big_five_score"),
                    "graph_interest": score_dict.get("graph_interest_score"),
                    "schwartz": score_dict.get("schwartz_score"),
                    "behavioral": score_dict.get("behavioral_score"),
                    "weights": score_dict.get("weights_applied"),
                },
                "_score_for_report": final_score,
            })

        # Исправлено: Сортируем ОДИН раз
        matches.sort(key=lambda x: x['compatibility'], reverse=True)
        response_data = jsonify(matches)
    return response_data


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