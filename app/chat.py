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
    # Получаем все чаты для текущего пользователя
    chat_data = get_user_chats(current_user.id)

    # Проверяем, передан ли chat_id в query параметрах
    requested_chat_id = request.args.get("chat_id")
    if requested_chat_id:
        try:
            requested_chat_id = int(requested_chat_id)
            # Проверяем, что запрошенный чат принадлежит текущему пользователю
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


def get_chat_name(chat_id, current_user_id):
    with get_db_session() as db_sess:
        # Получаем пользователей, участвующих в чате
        chat = db_sess.query(Chat).filter(Chat.id == chat_id).first()

        if not chat:
            return None

        # Получаем пользователей, связанные с этим чатом
        user1 = db_sess.query(User).filter(User.id == chat.user1_id).first()
        user2 = db_sess.query(User).filter(User.id == chat.user2_id).first()

        # Проверяем, чей чат и на основе этого возвращаем имя
        if current_user_id == user1.id:
            return user2.name
        else:
            return user1.name


def get_last_message(chat_id):
    with get_db_session() as db_sess:
        # Получаем последнее сообщение в чате
        last_message = db_sess.query(Message).filter(Message.chat_id == chat_id) \
            .order_by(Message.timestamp.desc()).first()

        return last_message


def get_user_chats(user_id):
    with get_db_session() as db_sess:
        # Получаем все чаты, в которых участвует данный пользователь (user_id)
        participant_chats = db_sess.query(Chat).filter(
            (Chat.user1_id == user_id) | (Chat.user2_id == user_id)
        ).all()

        chats = []

        for chat in participant_chats:
            # Получаем ID другого участника чата
            other_user_id = chat.user2_id if chat.user1_id == user_id else chat.user1_id

            # Получаем данные другого участника
            other_user = db_sess.query(User).filter(User.id == other_user_id).first()
            chat_name = other_user.name if other_user else "Неизвестный пользователь"

            # Получаем последнее сообщение в чате
            last_message = db_sess.query(Message).filter(Message.chat_id == chat.id).order_by(
                Message.timestamp.desc()
            ).first()

            # Собираем данные о чате
            chats.append({
                "chat_id": chat.id,
                "chat_name": chat_name,  # Имя второго участника чата или дефолт
                "last_message": last_message.content if last_message else None,
                "timestamp": last_message.timestamp if last_message else None,
                "message_type": last_message.message_type if last_message else None,
            })

        return chats


@chat_bp.route("/chat/messages/<int:chat_id>", methods=["GET"])
@login_required
def chat_messages(chat_id):
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

            # ВАЖНО: Добавляем текст сообщения, на которое отвечаем
            if message.reply_to_id:
                reply_msg = db_sess.query(Message).filter(Message.id == message.reply_to_id).first()
                if reply_msg:
                    # Если это картинка, можно написать "Фото", если текст — само содержание
                    if reply_msg.message_type == 'image':
                        msg_item["reply_to_content"] = "🖼 Фотография"
                    else:
                        msg_item["reply_to_content"] = reply_msg.content
                else:
                    msg_item["reply_to_content"] = "Сообщение удалено"

            messages_data.append(msg_item)

        # Получаем имя второго участника
        other_user_id = chat.user2_id if chat.user1_id == current_user.id else chat.user1_id
        other_user = db_sess.query(User).filter(User.id == other_user_id).first()
        chat_name = other_user.name if other_user else "Неизвестный пользователь"

        return jsonify({"messages": messages_data, "chat_name": chat_name})


@chat_bp.route("/messages", methods=["GET"])
def get_messages():
    """Получение всех сообщений из БД"""
    with get_db_session() as db_sess:
        # Получаем все сообщения, сортируя по timestamp (от новых к старым)
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
def handle_message(data):
    """Обработка входящего сообщения и его сохранение в БД"""
    text = data.get("text", "")
    file_url = data.get("file_url", None)
    message_type = "text"
    if file_url:
        # простой эвристический тип
        message_type = "image" if any(
            file_url.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) else "file"

    if not current_user.is_authenticated:
        # Игнорируем сообщения от неавторизованных пользователей
        return

    with get_db_session() as db_sess:
        message = Message(
            chat_id=data.get("chat_id"),
            author_id=current_user.id,
            content=text if not file_url else file_url,
            message_type=message_type,
        )

        db_sess.add(message)
        db_sess.commit()

        emit("addMessageToChat", {
            "author": current_user.id,
            "id": message.id,
            "content": message.content,
            "message_type": message.message_type,
        }, broadcast=True)


@chat_bp.route("/messages", methods=["POST"])
@login_required
def send_message():
    data = request.json
    author_id = current_user.id
    recipient_id = data.get("recipient_id")  # альтернативный способ определить чат
    chat_id = data.get("chat_id")
    content = data.get("content", "")
    message_type = data.get("type", "text")
    reply_to_id = data.get("reply_to_id")  # ID сообщения, на которое отвечаем

    # Определяем chat_id
    if not chat_id:
        if not recipient_id:
            return jsonify({"status": "error", "message": "recipient_id или chat_id обязателен"}), 400
        chat_id = get_or_create_chat(author_id, recipient_id)
    else:
        # Проверим принадлежность пользователя к чату
        with get_db_session() as db_sess:
            chat = db_sess.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return jsonify({"status": "error", "message": "Чат не найден"}), 404
            if author_id not in {chat.user1_id, chat.user2_id}:
                return jsonify({"status": "error", "message": "Доступ запрещён"}), 403

    with get_db_session() as db_sess:
        # Проверяем, что reply_to_id принадлежит тому же чату
        if reply_to_id:
            reply_message = db_sess.query(Message).filter(Message.id == reply_to_id).first()
            if not reply_message or reply_message.chat_id != chat_id:
                return jsonify({"status": "error", "message": "Неверное сообщение для ответа"}), 400

        new_message = Message(
            chat_id=chat_id,
            author_id=author_id,
            content=content,
            message_type=message_type,
            reply_to_id=reply_to_id,
        )
        db_sess.add(new_message)
        db_sess.commit()
        try:
            # Запускаем фоновый анализ пользователя
            analyze_user_profile.delay(author_id)
        except Exception as e:
            # Логируем ошибку, чтобы она не ломала чат
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
def edit_message(message_id):
    """Обновление содержимого сообщения в БД"""
    data = request.json
    new_content = data.get("content")

    if not new_content:
        return jsonify({"status": "error", "message": "Content cannot be empty"}), 400

    with get_db_session() as db_sess:
        # Проверяем, существует ли сообщение
        message = db_sess.execute(
            sa.select(Message).where(Message.id == message_id)
        ).scalar()

        if not message:
            return jsonify({"status": "error", "message": "Message not found"}), 404

        if message.author_id != current_user.id:
            return jsonify({"status": "error", "message": "Вы можете редактировать только свои сообщения"}), 403

        # Обновляем содержимое сообщения
        message.content = new_content
        db_sess.commit()

        return jsonify({"status": "ok", "message": {"id": message_id, "content": new_content}})


@chat_bp.route("/messages/<int:message_id>", methods=["DELETE"])
@login_required
def delete_message(message_id):
    with get_db_session() as db_sess:
        # Ищем сообщение в БД
        message = db_sess.execute(
            sa.select(Message).where(Message.id == message_id)
        ).scalar()
        if not message:
            return jsonify({"status": "error", "message": "Message not found"}), 404

        if message.author_id != current_user.id:
            return jsonify({"status": "error", "message": "Вы можете удалять только свои сообщения"}), 403
        # Если это файл или изображение — удаляем его из папки загрузок
        if message.message_type in ["file", "image"]:
            upload_folder = current_app.config["UPLOAD_FOLDER"]
            file_path = os.path.join(upload_folder, os.path.basename(message.content))
            if os.path.exists(file_path):
                os.remove(file_path)  # Удаляем файл

        # Удаляем сообщение из БД
        db_sess.delete(message)
        db_sess.commit()

        return jsonify({"status": "ok"})


@chat_bp.route("/upload", methods=["POST"])
def upload_file():
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

    # Определяем путь: static/uploads/chat_files
    # Убедись, что UPLOAD_FOLDER в конфиге указывает на 'static/uploads'
    chat_folder = os.path.join(current_app.config["UPLOAD_FOLDER"], "chat_files")
    os.makedirs(chat_folder, exist_ok=True)

    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(chat_folder, unique_filename)

    try:
        # Если это изображение — сжимаем, но НЕ обрезаем
        if ext in {"png", "jpg", "jpeg", "webp"}:
            unique_filename = f"{uuid.uuid4().hex}.webp"  # Переводим в webp
            file_path = os.path.join(chat_folder, unique_filename)

            with PILImage.open(file) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Ограничиваем только макс. размер (например, Full HD), сохраняя пропорции
                img.thumbnail((1920, 1080), PILImage.Resampling.LANCZOS)
                img.save(file_path, "WEBP", quality=75, optimize=True)
        else:
            # Все остальные файлы (PDF, ZIP и т.д.) просто сохраняем как есть
            file.save(file_path)

    except Exception as e:
        return jsonify({"error": f"Save error: {str(e)}"}), 500

    # Теперь используем стандартный url_for для статики
    # Это работает быстрее, так как Flask/Nginx отдают файлы напрямую
    file_url = url_for('static', filename=f'uploads/chat_files/{unique_filename}')

    return jsonify({"status": "ok", "file_url": file_url})


@chat_bp.route("/uploads/<filename>")
def get_uploaded_file(filename):
    # Указываем путь до конкретной папки чата
    upload_folder = os.path.join(current_app.config["UPLOAD_FOLDER"], "chat_files")
    full_path = os.path.join(current_app.root_path, upload_folder)

    return send_from_directory(full_path, filename)


def get_or_create_chat(user1_id, user2_id):
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
def get_chat_messages(chat_id):
    with get_db_session() as db_sess:
        chat = db_sess.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            return jsonify({"status": "error", "message": "Чат не найден"}), 404

        # Проверяем, что текущий пользователь участник чата
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
            # Загружаем информацию о сообщении, на которое отвечают
            if msg.reply_to_id:
                reply_msg = db_sess.query(Message).filter(Message.id == msg.reply_to_id).first()
                if reply_msg:
                    msg_data["reply_to_content"] = reply_msg.content or ""
            result.append(msg_data)

        return jsonify({"messages": result, "chat_name": get_chat_name(chat_id, current_user.id)})


@chat_bp.route("/chat/settings", methods=["GET", "POST"])
@login_required
def chat_settings():
    """Настройки чата"""
    with get_db_session() as db_sess:
        if request.method == "GET":
            settings = db_sess.query(ChatSettings).filter(ChatSettings.user_id == current_user.id).first()
            if not settings:
                # Создаём настройки по умолчанию
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
            # POST - обновление настроек
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
