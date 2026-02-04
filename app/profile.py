import os

from flask import Blueprint, render_template, request, redirect, abort, jsonify, current_app
from flask_login import login_required, current_user

from data.user import User
from data.interest import Interest
from app.db import get_db_session
from werkzeug.utils import secure_filename

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

    filename = secure_filename(photo.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"png", "jpg", "jpeg", "webp"}:
        return jsonify({"success": False, "message": "Недопустимый формат файла. Разрешены только изображения (JPG, PNG, WEBP)"}), 400

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)
    
    # Генерируем уникальное имя файла, чтобы избежать конфликтов
    import uuid
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    photo_path = os.path.join(upload_folder, unique_filename)
    photo.save(photo_path)

    # Сохраняем относительный путь для использования в шаблонах
    relative_path = f"uploads/{unique_filename}"

    with get_db_session() as db_sess:
        user = db_sess.get(User, current_user.id)
        if not user:
            return jsonify({"success": False, "message": "Пользователь не найден"}), 404
        
        # Удаляем старое изображение, если оно есть
        if user.image_path:
            old_path = user.image_path
            # Проверяем разные варианты пути
            if not old_path.startswith('http'):
                if not old_path.startswith('static/'):
                    old_path = os.path.join('static', old_path)
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except:
                        pass
        
        user.image_path = relative_path
        db_sess.commit()
    
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
    from data.knowledge_graph import KnowledgeNode, KnowledgeConnection
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
        "knowledge_graph.html",
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


