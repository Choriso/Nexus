import os

from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory, url_for
from flask_login import login_required, current_user
from flask_socketio import emit
import sqlalchemy as sa
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
import uuid
from PIL import Image as PILImage

from app.ai.personality_analyzer import analyze_user_profile
from app.ai_profiler.behavior_analyzer import compute_message_metadata, refresh_user_behavior_profile

from data.chat import Chat
from data.message import Message
from data.user import User
from data.chat_settings import ChatSettings
from app.db import get_db_session
from app.extensions import socketio

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["GET"])
@login_required
def all_chats():
    """
    Отображает страницу со списком всех чатов пользователя.

    Returns:
        flask.Response: HTML-страница со всеми чатами текущего пользователя.
    """
    chat_data = get_user_chats(current_user.id)
    requested_chat_id = request.args.get("chat_id")
    if requested_chat_id:
        try:
            requested_chat_id = int(requested_chat_id)
            if any(chat["chat_id"] == requested_chat_id for chat in chat_data):
                default_chat_id = requested_chat_id
            else:
                default_chat_id = chat_data[0]["chat_id"] if chat_data else None
        except (ValueError, TypeError):
            default_chat_id = chat_data[0]["chat_id"] if chat_data else None
    else:
        default_chat_id = chat_data[0]["chat_id"] if chat_data else None

    return render_template(
        "chat.html",
        chat_data=chat_data,
        default_chat_id=default_chat_id,
    )


def get_chat_name(chat_id: int, current_user_id: int) -> str | None:
    """
    Получает имя второго участника чата.

    Args:
        chat_id (int): ID чата.
        current_user_id (int): ID текущего пользователя.

    Returns:
        str | None: Имя второго участника чата или None, если чат не найден.
    """
    with get_db_session() as db_sess:
        chat = db_sess.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            return None
        user1 = db_sess.query(User).filter(User.id == chat.user1_id).first()
        user2 = db_sess.query(User).filter(User.id == chat.user2_id).first()
        if current_user_id == user1.id:
            return user2.name
        else:
            return user1.name


def get_last_message(chat_id: int) -> Message | None:
    """
    Получает последнее сообщение в чате.

    Args:
        chat_id (int): ID чата.

    Returns:
        Message | None: Последнее сообщение в чате или None, если сообщений нет.
    """
    with get_db_session() as db_sess:
        last_message = db_sess.query(Message).filter(Message.chat_id == chat_id) \
            .order_by(Message.timestamp.desc()).first()
        return last_message


def get_user_chats(user_id: int) -> list[dict]:
    """
    Получает все чаты для указанного пользователя с краткой информацией по каждому чату.

    Args:
        user_id (int): ID пользователя.

    Returns:
        list[dict]: Список чатов с основными данными (ID чата, имя собеседника, последнее сообщение и пр.).
    """
    with get_db_session() as db_sess:
        participant_chats = db_sess.query(Chat).filter(
            (Chat.user1_id == user_id) | (Chat.user2_id == user_id)
        ).all()

        chats = []
        for chat in participant_chats:
            other_user_id = chat.user2_id if chat.user1_id == user_id else chat.user1_id
            other_user = db_sess.query(User).filter(User.id == other_user_id).first()
            chat_name = other_user.name if other_user else "Неизвестный пользователь"
            last_message = db_sess.query(Message).filter(Message.chat_id == chat.id).order_by(
                Message.timestamp.desc()
            ).first()
            chats.append({
                "chat_id": chat.id,
                "chat_name": chat_name,
                "other_user_id": other_user_id,
                "chat_avatar": other_user.image_path if other_user and other_user.image_path else None,
                "last_message": last_message.content if last_message else None,
                "timestamp": last_message.timestamp if last_message else None,
                "message_type": last_message.message_type if last_message else None,
            })
        return chats


@chat_bp.route("/chat/messages/<int:chat_id>", methods=["GET"])
@login_required
def chat_messages(chat_id: int):
    """
    Получить все сообщения заданного чата, включая сведения о сообщениях-ответах.

    Args:
        chat_id (int): ID чата.

    Returns:
        flask.Response: JSON-объект с сообщениями чата и именем собеседника.
    """
    with get_db_session() as db_sess:
        chat = db_sess.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            return jsonify({"messages": [], "chat_name": "Чат не найден"}), 404

        messages = db_sess.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp.asc()).all()
        messages_data = []
        for message in messages:
            msg_item = {
                "id": message.id,
                "content": message.content,
                "message_type": message.message_type,
                "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                "sent_by_user": message.author_id == current_user.id,
                "reply_to_id": message.reply_to_id,
            }
            if message.reply_to_id:
                reply_msg = db_sess.query(Message).filter(Message.id == message.reply_to_id).first()
                if reply_msg:
                    if reply_msg.message_type == 'image':
                        msg_item["reply_to_content"] = "🖼 Фотография"
                    else:
                        msg_item["reply_to_content"] = reply_msg.content
                else:
                    msg_item["reply_to_content"] = "Сообщение удалено"
            messages_data.append(msg_item)
        other_user_id = chat.user2_id if chat.user1_id == current_user.id else chat.user1_id
        other_user = db_sess.query(User).filter(User.id == other_user_id).first()
        chat_name = other_user.name if other_user else "Неизвестный пользователь"

        return jsonify({"messages": messages_data, "chat_name": chat_name})


@chat_bp.route("/messages", methods=["GET"])
def get_messages():
    """
    Получает все сообщения из базы данных (служебный маршрут).

    Returns:
        flask.Response: JSON со всеми сообщениями.
    """
    with get_db_session() as db_sess:
        messages = db_sess.execute(
            sa.select(Message).order_by(Message.timestamp.asc())
        ).scalars().all()
        return jsonify([{
            "id": msg.id,
            "content": msg.content,
            "message_type": msg.message_type,
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
        } for msg in messages])


@socketio.on("send_message")
def handle_message(data: dict):
    """
    Обрабатывает отправку сообщения через WebSocket и сохраняет сообщение в базу данных.

    Args:
        data (dict): Данные сообщения. Ожидаемые ключи: text, chat_id, file_url.

    Returns:
        None
    """
    text = data.get("text", "")
    file_url = data.get("file_url", None)
    message_type = "text"
    if file_url:
        message_type = "image" if any(
            file_url.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) else "file"

    if not current_user.is_authenticated:
        return

    with get_db_session() as db_sess:
        from datetime import datetime

        chat_id = data.get("chat_id")
        prev_msg = None
        if chat_id:
            prev_msg = (
                db_sess.query(Message)
                .filter(Message.chat_id == chat_id)
                .order_by(Message.timestamp.desc())
                .first()
            )
        now = datetime.utcnow()
        char_count, reply_time, emoji_count = compute_message_metadata(
            text,
            prev_timestamp=prev_msg.timestamp if prev_msg else None,
            now=now,
        )
        message = Message(
            chat_id=chat_id,
            author_id=current_user.id,
            content=text if not file_url else file_url,
            message_type=message_type,
            char_count=char_count,
            reply_time=reply_time,
            emoji_count=emoji_count,
            timestamp=now,
        )
        db_sess.add(message)
        db_sess.commit()
        refresh_user_behavior_profile(db_sess, current_user.id)

        emit("addMessageToChat", {
            "author": current_user.id,
            "id": message.id,
            "content": message.content,
            "message_type": message.message_type,
        }, broadcast=True)


@chat_bp.route("/messages", methods=["POST"])
@login_required
def send_message():
    """
    Отправка нового сообщения в чат.

    Returns:
        flask.Response: JSON-статус. В случае успеха – сведения о новом сообщении.
    """
    data = request.json
    author_id = current_user.id
    recipient_id = data.get("recipient_id")
    chat_id = data.get("chat_id")
    content = data.get("content", "")
    message_type = data.get("type", "text")
    reply_to_id = data.get("reply_to_id")

    if not chat_id:
        if not recipient_id:
            return jsonify({"status": "error", "message": "recipient_id или chat_id обязателен"}), 400
        chat_id = get_or_create_chat(author_id, recipient_id)
    else:
        with get_db_session() as db_sess:
            chat = db_sess.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return jsonify({"status": "error", "message": "Чат не найден"}), 404
            if author_id not in {chat.user1_id, chat.user2_id}:
                return jsonify({"status": "error", "message": "Доступ запрещён"}), 403

    with get_db_session() as db_sess:
        from datetime import datetime

        if reply_to_id:
            reply_message = db_sess.query(Message).filter(Message.id == reply_to_id).first()
            if not reply_message or reply_message.chat_id != chat_id:
                return jsonify({"status": "error", "message": "Неверное сообщение для ответа"}), 400

        prev_msg = (
            db_sess.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.timestamp.desc())
            .first()
        )
        now = datetime.utcnow()
        char_count, reply_time, emoji_count = compute_message_metadata(
            content,
            prev_timestamp=prev_msg.timestamp if prev_msg else None,
            now=now,
        )

        new_message = Message(
            chat_id=chat_id,
            author_id=author_id,
            content=content,
            message_type=message_type,
            reply_to_id=reply_to_id,
            char_count=char_count,
            reply_time=reply_time,
            emoji_count=emoji_count,
            timestamp=now,
        )
        db_sess.add(new_message)
        db_sess.commit()
        refresh_user_behavior_profile(db_sess, author_id)
        try:
            analyze_user_profile.delay(author_id)
        except Exception as e:
            print(f"Ошибка вызова Celery: {e}")

        return jsonify({"status": "ok", "message": {
            "id": new_message.id,
            "chat_id": chat_id,
            "author_id": new_message.author_id,
            "content": new_message.content,
            "message_type": new_message.message_type,
            "reply_to_id": new_message.reply_to_id,
        }})


@chat_bp.route("/messages/<int:message_id>", methods=["PUT"])
@login_required
def edit_message(message_id: int):
    """
    Обновляет текстовое содержимое сообщения.

    Args:
        message_id (int): ID сообщения для редактирования.

    Returns:
        flask.Response: JSON-статус и обновлённый контент сообщения или ошибку.
    """
    data = request.json
    new_content = data.get("content")

    if not new_content:
        return jsonify({"status": "error", "message": "Content cannot be empty"}), 400

    with get_db_session() as db_sess:
        message = db_sess.execute(
            sa.select(Message).where(Message.id == message_id)
        ).scalar()
        if not message:
            return jsonify({"status": "error", "message": "Message not found"}), 404

        if message.author_id != current_user.id:
            return jsonify({"status": "error", "message": "Вы можете редактировать только свои сообщения"}), 403

        message.content = new_content
        db_sess.commit()

        return jsonify({"status": "ok", "message": {"id": message_id, "content": new_content}})


@chat_bp.route("/messages/<int:message_id>", methods=["DELETE"])
@login_required
def delete_message(message_id: int):
    """
    Удаляет сообщение пользователя, включая удаление файлов при необходимости.

    Args:
        message_id (int): ID сообщения.

    Returns:
        flask.Response: JSON-статус, код ошибки при необходимости.
    """
    with get_db_session() as db_sess:
        message = db_sess.execute(
            sa.select(Message).where(Message.id == message_id)
        ).scalar()
        if not message:
            return jsonify({"status": "error", "message": "Message not found"}), 404

        if message.author_id != current_user.id:
            return jsonify({"status": "error", "message": "Вы можете удалять только свои сообщения"}), 403
        if message.message_type in ["file", "image"]:
            upload_folder = current_app.config["UPLOAD_FOLDER"]
            file_path = os.path.join(upload_folder, os.path.basename(message.content))
            if os.path.exists(file_path):
                os.remove(file_path)

        db_sess.delete(message)
        db_sess.commit()

        return jsonify({"status": "ok"})


@chat_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Загружает файл в чат (поддержка изображений и документов), осуществляет обработку изображений.

    Returns:
        flask.Response: JSON-статус и URL загруженного файла, либо сообщение об ошибке.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    allowed_extensions = {
        "png", "jpg", "jpeg", "webp", "pdf", "doc", "docx", "xls", "xlsx",
        "ppt", "pptx", "txt", "md", "json", "xml", "csv", "zip", "rar", "7z",
        "mp3", "wav", "mp4", "avi", "mov"
    }

    if ext not in allowed_extensions:
        return jsonify({"error": "Unsupported file type"}), 400

    chat_folder = os.path.join(current_app.config["UPLOAD_FOLDER"], "chat_files")
    os.makedirs(chat_folder, exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(chat_folder, unique_filename)

    try:
        if ext in {"png", "jpg", "jpeg", "webp"}:
            unique_filename = f"{uuid.uuid4().hex}.webp"
            file_path = os.path.join(chat_folder, unique_filename)
            with PILImage.open(file) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.thumbnail((1920, 1080), PILImage.Resampling.LANCZOS)
                img.save(file_path, "WEBP", quality=75, optimize=True)
        else:
            file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Save error: {str(e)}"}), 500

    file_url = url_for('static', filename=f'uploads/chat_files/{unique_filename}')
    return jsonify({"status": "ok", "file_url": file_url})


@chat_bp.route("/uploads/<filename>")
def get_uploaded_file(filename: str):
    """
    Отдаёт загруженный файл из папки чата по указанному имени.

    Args:
        filename (str): Имя файла в папке загрузок.

    Returns:
        flask.Response: Файл для скачивания.
    """
    upload_folder = os.path.join(current_app.config["UPLOAD_FOLDER"], "chat_files")
    full_path = os.path.join(current_app.root_path, upload_folder)
    return send_from_directory(full_path, filename)


def get_or_create_chat(user1_id: int, user2_id: int) -> int:
    """
    Получает существующий чат между двумя пользователями или создаёт новый, если такого нет.

    Args:
        user1_id (int): ID первого пользователя.
        user2_id (int): ID второго пользователя.

    Returns:
        int: ID чата.
    """
    with get_db_session() as db_sess:
        chat = db_sess.execute(
            sa.select(Chat).where(
                ((Chat.user1_id == user1_id) & (Chat.user2_id == user2_id)) |
                ((Chat.user1_id == user2_id) & (Chat.user2_id == user1_id))
            )
        ).scalar()
        if not chat:
            chat = Chat(user1_id=user1_id, user2_id=user2_id)
            db_sess.add(chat)
            db_sess.commit()
        return chat.id


@chat_bp.route("/messages/<int:chat_id>", methods=["GET"])
@login_required
def get_chat_messages(chat_id: int):
    """
    Получает все сообщения указанного чата по ID, с учётом прав пользователя.

    Args:
        chat_id (int): ID чата.

    Returns:
        flask.Response: JSON-объект с сообщениями и названием чата, либо ошибку доступа/нахождения.
    """
    with get_db_session() as db_sess:
        chat = db_sess.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            return jsonify({"status": "error", "message": "Чат не найден"}), 404

        if current_user.id not in {chat.user1_id, chat.user2_id}:
            return jsonify({"status": "error", "message": "Доступ запрещён"}), 403

        messages = db_sess.execute(
            sa.select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.timestamp.asc())
        ).scalars().all()

        result = []
        for msg in messages:
            msg_data = {
                "id": msg.id,
                "author_id": msg.author_id,
                "content": msg.content,
                "message_type": msg.message_type,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "reply_to_id": msg.reply_to_id,
                "sent_by_user": msg.author_id == current_user.id,
            }
            if msg.reply_to_id:
                reply_msg = db_sess.query(Message).filter(Message.id == msg.reply_to_id).first()
                if reply_msg:
                    msg_data["reply_to_content"] = reply_msg.content or ""
            result.append(msg_data)

        return jsonify({"messages": result, "chat_name": get_chat_name(chat_id, current_user.id)})


@chat_bp.route("/chat/settings", methods=["GET", "POST"])
@login_required
def chat_settings():
    """
    Получение и изменение настроек чата текущего пользователя. 
    GET — получить настройки; POST — обновить настройки.

    Returns:
        flask.Response: JSON-объект с текущими/новыми настройками пользователя.
    """
    with get_db_session() as db_sess:
        if request.method == "GET":
            settings = db_sess.query(ChatSettings).filter(ChatSettings.user_id == current_user.id).first()
            if not settings:
                settings = ChatSettings(user_id=current_user.id)
                db_sess.add(settings)
                db_sess.commit()
            return jsonify({
                "sound_enabled": settings.sound_enabled,
                "notifications_enabled": settings.notifications_enabled,
                "theme": settings.theme,
                "font_size": settings.font_size,
            })
        else:
            data = request.json
            settings = db_sess.query(ChatSettings).filter(ChatSettings.user_id == current_user.id).first()
            if not settings:
                settings = ChatSettings(user_id=current_user.id)
                db_sess.add(settings)
            if "sound_enabled" in data:
                settings.sound_enabled = data["sound_enabled"]
            if "notifications_enabled" in data:
                settings.notifications_enabled = data["notifications_enabled"]
            if "theme" in data:
                settings.theme = data["theme"]
            if "font_size" in data:
                settings.font_size = int(data["font_size"])
            db_sess.commit()
            return jsonify({"success": True, "message": "Настройки сохранены"})


@chat_bp.route("/chat/<int:chat_id>/delete", methods=["DELETE"])
@login_required
def delete_chat(chat_id: int):
    """Удаляет чат и все его сообщения."""
    from data.message import Message
    with get_db_session() as db_sess:
        chat = db_sess.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            return jsonify({"success": False, "message": "Чат не найден"}), 404
        if chat.user1_id != current_user.id and chat.user2_id != current_user.id:
            return jsonify({"success": False, "message": "Нет доступа"}), 403
        db_sess.query(Message).filter(Message.chat_id == chat.id).delete()
        db_sess.delete(chat)
        db_sess.commit()
        return jsonify({"success": True})


@chat_bp.route("/chat/<int:user_id>/block", methods=["POST"])
@login_required
def block_user(user_id: int):
    """Блокирует пользователя."""
    if user_id == current_user.id:
        return jsonify({"success": False, "message": "Нельзя заблокировать себя"}), 400
    from data.blocked_user import BlockedUser
    with get_db_session() as db_sess:
        existing = db_sess.query(BlockedUser).filter_by(
            user_id=current_user.id, blocked_user_id=user_id
        ).first()
        if existing:
            return jsonify({"success": False, "message": "Пользователь уже заблокирован"})
        db_sess.add(BlockedUser(user_id=current_user.id, blocked_user_id=user_id))
        db_sess.commit()
        return jsonify({"success": True})


@chat_bp.route("/chat/<int:user_id>/unblock", methods=["POST"])
@login_required
def unblock_user(user_id: int):
    """Разблокирует пользователя."""
    from data.blocked_user import BlockedUser
    with get_db_session() as db_sess:
        entry = db_sess.query(BlockedUser).filter_by(
            user_id=current_user.id, blocked_user_id=user_id
        ).first()
        if entry:
            db_sess.delete(entry)
            db_sess.commit()
        return jsonify({"success": True})
