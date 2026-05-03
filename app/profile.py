import os

from flask import Blueprint, render_template, request, redirect, abort, jsonify, current_app
from flask_login import login_required, current_user
from data.knowledge_graph import KnowledgeNode, KnowledgeConnection

from data.user import User
from data.interest import Interest
from app.db import get_db_session
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
from app.ai_profiler.core import AIProfiler
from app.ai.models import UserPersonalityProfile

profile_bp = Blueprint("profile", __name__)


@profile_bp.route("/viewProfile", methods=["GET"])
def view_profile():
    user_id = request.args.get("user_id")

    with get_db_session() as db_sess:
        interest = db_sess.query(Interest).filter(Interest.user_id == user_id)
        user = db_sess.query(User).get(user_id)
        if not user:
            abort(404)
    return render_template("view_profile.html", interest=interest, user=user)


@profile_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    with get_db_session() as db_sess:
        user = db_sess.query(User).get(current_user.id)
        if not user:
            abort(404)
        # Созданные интересы
        created_interests = db_sess.query(Interest).filter(Interest.user == current_user).all()
        # Избранные интересы
        from data.favorite_interest import FavoriteInterest
        from sqlalchemy.orm import joinedload
        favorite_ids = [fav.interest_id for fav in db_sess.query(FavoriteInterest).filter(
            FavoriteInterest.user_id == current_user.id
        ).all()]
        favorite_interests = db_sess.query(Interest).options(joinedload(Interest.user)).filter(
            Interest.id.in_(favorite_ids)
        ).all() if favorite_ids else []
        return render_template("profile.html", title="Профиль",
                               interest=created_interests,
                               favorite_interests=favorite_interests,
                               current_user=current_user)


@profile_bp.route("/upload_avatar", methods=["POST"])
@login_required
def upload_avatar():
    if "photo" not in request.files:
        return jsonify({"success": False, "message": "Файл не найден"}), 400

    photo = request.files["photo"]
    if photo.filename == "":
        return jsonify({"success": False, "message": "Файл не выбран"}), 400

    # Проверка расширения
    filename = secure_filename(photo.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"png", "jpg", "jpeg", "webp"}:
        return jsonify({"success": False, "message": "Недопустимый формат"}), 400

    # ИЗМЕНЕНИЕ: Указываем путь именно в avatars
    # Обычно в config['UPLOAD_FOLDER'] лежит 'static/uploads'
    base_upload_folder = current_app.config["UPLOAD_FOLDER"]
    avatar_folder = os.path.join(base_upload_folder, "avatars")
    os.makedirs(avatar_folder, exist_ok=True)

    # Всегда сохраняем в .webp для экономии места
    unique_filename = f"{uuid.uuid4().hex}.webp"
    photo_path = os.path.join(avatar_folder, unique_filename)

    try:
        # ОБРАБОТКА (как обсуждали: квадрат + сжатие)
        with Image.open(photo) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Делаем квадрат
            width, height = img.size
            min_side = min(width, height)
            left = (width - min_side) / 2
            top = (height - min_side) / 2
            right = (width + min_side) / 2
            bottom = (height + min_side) / 2
            img = img.crop((left, top, right, bottom))

            # Ресайз до 400px (оптимально для профиля)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            img.save(photo_path, 'WEBP', quality=85, optimize=True)

    except Exception as e:
        return jsonify({"success": False, "message": f"Ошибка обработки: {str(e)}"}), 500

    # ИЗМЕНЕНИЕ: Относительный путь теперь включает avatars
    relative_path = f"uploads/avatars/{unique_filename}"

    with get_db_session() as db_sess:
        user = db_sess.get(User, current_user.id)

        # Удаление старой аватарки (с защитой от удаления дефолтной)
        if user.image_path and 'default' not in user.image_path:
            # Важно: удаляем через полный путь от корня проекта
            old_full_path = os.path.join('static', user.image_path)
            if os.path.exists(old_full_path):
                try:
                    os.remove(old_full_path)
                except:
                    pass

        user.image_path = relative_path
        db_sess.commit()

    # Возвращаем путь, который поймет <img> src
    return jsonify({"success": True, "image_path": f"/static/{relative_path}"})


@profile_bp.route("/process_profile", methods=["POST"])
@login_required
def process_profile():
    with get_db_session() as db_sess:
        user = db_sess.get(User, current_user.id)
        if not user:
            abort(404)
        # ожидаем, что фронт отправляет данные формой — используем form в качестве базового варианта
        user.name = request.form.get("name", user.name)
        user.information = request.form.get("information", user.information)
        user.connection = request.form.get("connection", user.connection)
        db_sess.commit()
    return redirect("/profile")


@profile_bp.route("/settings", methods=["GET"])
@login_required
def settings():
    return render_template("settings.html", current_user=current_user)


@profile_bp.route("/update_profile", methods=["POST"])
@login_required
def update_profile():
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
            # Здесь можно добавить проверку кода подтверждения через email
            # Пока просто обновляем пароль
            user.set_password(data["password"])

        db_sess.commit()

    return jsonify({"success": True, "message": "Настройки сохранены"})


@profile_bp.route("/knowledge_graph", methods=["GET"])
@login_required
def knowledge_graph():
    """Страница графа знаний"""
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
    """Создание нового узла графа"""
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
    """Обновление или удаление узла"""
    from data.knowledge_graph import KnowledgeNode, KnowledgeConnection

    with get_db_session() as db_sess:
        node = db_sess.query(KnowledgeNode).filter(
            KnowledgeNode.id == node_id,
            KnowledgeNode.user_id == current_user.id
        ).first()

        if not node:
            return jsonify({"success": False, "message": "Узел не найден"}), 404

        if request.method == "DELETE":
            # Удаляем все связи
            db_sess.query(KnowledgeConnection).filter(
                (KnowledgeConnection.from_node_id == node_id) |
                (KnowledgeConnection.to_node_id == node_id)
            ).delete()
            db_sess.delete(node)
            db_sess.commit()
            return jsonify({"success": True})
        else:
            # PUT - обновление
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
    """Создание или удаление связи"""
    from data.knowledge_graph import KnowledgeNode, KnowledgeConnection

    with get_db_session() as db_sess:
        if request.method == "POST":
            data = request.json
            from_id = data.get("from_node_id")
            to_id = data.get("to_node_id")

            # Проверяем, что узлы принадлежат пользователю
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

            # Проверяем, нет ли уже такой связи
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
            # DELETE
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


# Настраиваем веса под категории
CATEGORY_CONFIG = {
    'work': {
        'weights': [0.3, 1.0, 0.5, 0.2, 0.8],  # Акцент на C (дисциплина) и N (стабильность)
        'complementary': [2]  # Индекс Экстраверсии (E) — ищем дополнение
    },
    'hobby': {
        'weights': [1.0, 0.2, 0.8, 0.5, 0.1],  # Акцент на O (открытость) и E (драйв)
        'complementary': []
    },
    'psychology': {
        'weights': [0.7, 0.4, 0.4, 1.0, 0.6],  # Акцент на A (дружелюбие)
        'complementary': []
    }
}


@profile_bp.route("/api/graph/match/<int:node_id>")
@login_required
def match_by_node(node_id):
    # Получаем категорию из запроса (по умолчанию 'psychology')
    raw_cat = request.args.get('category', 'psychology').lower().strip()

    # Маппинг для поддержки русского и английского
    cat_mapping = {
        'работа': 'work',
        'work': 'work',
        'хобби': 'hobby',
        'hobby': 'hobby',
        'психология': 'psychology',
        'psychology': 'psychology'
    }

    cat_type = cat_mapping.get(raw_cat, 'psychology')

    # ДЕБАГ: Проверяем в консоли Flask, что пришло
    print(f"--- MATCHING LOG ---")
    print(f"Raw category from request: '{raw_cat}'")
    print(f"Mapped category: '{cat_type}'")
    print(f"Weights used: {CATEGORY_CONFIG[cat_type]['weights']}")
    config = CATEGORY_CONFIG.get(cat_type, CATEGORY_CONFIG['psychology'])

    with get_db_session() as db_sess:
        target_node = db_sess.query(KnowledgeNode).get(node_id)
        if not target_node:
            return jsonify({"error": "Node not found"}), 404

        my_profile = db_sess.query(UserPersonalityProfile).filter_by(user_id=current_user.id).first()

        from app.ai_profiler.core import AIProfiler
        profiler = AIProfiler()

        candidates = db_sess.query(KnowledgeNode).filter(
            # Либо ищем по категории узла (например, все узлы из 'Работа')
            # Либо берем вообще всех, если база небольшая (до 1000-5000 записей это быстро)
            KnowledgeNode.user_id != current_user.id
        ).all()
        matches = []
        for node in candidates:
            other_user = node.user
            other_profile = db_sess.query(UserPersonalityProfile).filter_by(user_id=other_user.id).first()

            interest_sim = profiler.calculate_text_similarity(target_node.title, node.title)

            # Инициализируем векторы нулями на случай, если профиля нет
            my_vec = [0, 0, 0, 0, 0]
            other_vec = [0, 0, 0, 0, 0]
            psy_score = 0.5

            if my_profile and other_profile:
                my_vec = [my_profile.openness, my_profile.conscientiousness, my_profile.extraversion,
                          my_profile.agreeableness, my_profile.neuroticism]
                other_vec = [other_profile.openness, other_profile.conscientiousness, other_profile.extraversion,
                             other_profile.agreeableness, other_profile.neuroticism]

                processed_other_vec = other_vec[:]
                for idx in config['complementary']:
                    processed_other_vec[idx] = 1.0 - other_vec[idx]

                psy_score = profiler.calculate_compatibility(
                    my_vec,
                    processed_other_vec,
                    weights=config['weights']
                ) / 100

            total_score = (interest_sim * 0.7) + (psy_score * 0.3)

            # Теперь my_vec точно существует, даже если профили пустые
            reason = "Похожие интересы"
            if cat_type == 'work' and my_vec[2] != 0 and other_vec[2] != 0:
                if abs(my_vec[2] - other_vec[2]) > 0.4:
                    reason = "Дополняет вашу команду"

            matches.append({
                "user_id": other_user.id,
                "user_name": other_user.name,
                "node_title": node.title,
                "compatibility": round(total_score * 100, 1),
                "category": node.category,
                "match_reason": reason
            })

        matches = sorted(matches, key=lambda x: x['compatibility'], reverse=True)
        return jsonify(matches)