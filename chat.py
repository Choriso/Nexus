import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
import sqlite3
from app.ai.personality_analyzer import analyze_user_profile
from PIL import Image
from FindLinks import contains_contact_info

conn = sqlite3.connect("chat.db")
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        author TEXT NOT NULL,
        content TEXT,
        file_url TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        message_type TEXT CHECK(message_type IN ('text', 'image', 'file')) NOT NULL
    )
""")

conn.commit()
conn.close()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chat.db"
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["SECRET_KEY"] = "your_secret_key"

socketio = SocketIO(app, cors_allowed_origins="*")
db = SQLAlchemy(app)
migrate = Migrate(app, db)

CORS(app)  # Разрешаем CORS для фронта
messages = []
message_id_counter = 1


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=True)
    file_path = db.Column(db.String(500), nullable=True)


with app.app_context():
    db.create_all()


def get_db_connection():
    conn = sqlite3.connect("chat.db")
    conn.row_factory = sqlite3.Row  # Для работы с результатами как с dict
    return conn


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("send_message")
def handle_message(data):
    text = data.get("text", "")
    file_path = data.get("file_path", None)
    print(file_path)
    message = Message(text=text, file_path=file_path)
    db.session.add(message)
    db.session.commit()
    db.session.close()
    emit("addMessageToChat", {"author": "0000", "id": message.id, "text": text}, broadcast=True)


# Добавь этот импорт в начало файла (там, где у тебя импорты)
from app.ai.personality_analyzer import analyze_user_profile


@app.route("/messages", methods=["POST"])
def send_message():
    print("--- ЗАПРОС ПРИШЕЛ В SEND_MESSAGE ---")
    data = request.json
    print(f"DEBUG: Полученные данные: {data}")
    author = data.get("author", "Аноним")
    content = data.get("content", "")
    file_url = data.get("file_url", None)
    message_type = data.get("type", "text")

    conn = get_db_connection()
    cur = conn.cursor()
    # Получаем ID автора (или его имя/ник, если ID нет)
    cur.execute(
        "INSERT INTO messages (author, content, file_url, message_type) VALUES (?, ?, ?, ?)",
        (author, content, file_url, message_type),
    )
    # Получаем ID только что созданного сообщения, чтобы понять, какой юзер пишет
    # (если у тебя есть таблица users, лучше использовать user_id)
    conn.commit()
    conn.close()
    print("dfdfdf")
    # --- ВОТ ЗДЕСЬ МАГИЯ ИИ ---
    # Мы не ждем окончания анализа, а просто ставим задачу в очередь
    # Предполагая, что author - это ID пользователя
    try:
        analyze_user_profile.delay(author)
    except Exception as e:
        print(f"Ошибка запуска задачи ИИ: {e}")
    # ---------------------------

    return jsonify({"status": "ok"})

@app.route("/messages", methods=["GET"])
def get_messages():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM messages ORDER BY timestamp DESC")
    messages = cur.fetchall()
    conn.close()

    return jsonify([dict(msg) for msg in messages])


# Изменение сообщения
@app.route("/messages/<int:message_id>", methods=["PUT"])
def edit_message(message_id):
    """Обновление содержимого сообщения в БД"""
    data = request.json
    new_content = data.get("content")

    if not new_content:
        return jsonify({"status": "error", "message": "Content cannot be empty"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    # Проверяем, существует ли сообщение
    cur.execute("SELECT id FROM messages WHERE id = ?", (message_id,))
    message = cur.fetchone()

    if not message:
        return jsonify({"status": "error", "message": "Message not found"}), 404

    # Обновляем содержимое сообщения
    cur.execute("UPDATE messages SET content = ? WHERE id = ?", (new_content, message_id))
    conn.commit()

    return jsonify({"status": "ok", "message": {"id": message_id, "content": new_content}})


# Удаление сообщения
@app.route("/messages/<int:message_id>", methods=["DELETE"])
def delete_message(message_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # Найти сообщение перед удалением (чтобы узнать, файл это или текст)
    cur.execute("SELECT content, message_type FROM messages WHERE id = ?", (message_id,))
    message = cur.fetchone()

    if not message:
        return jsonify({"status": "error", "message": "Message not found"}), 404

    content, msg_type = message

    # Если это файл — удаляем его из папки загрузок
    if msg_type == "file" or msg_type == "image":
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(content))
        if os.path.exists(file_path):
            os.remove(file_path)  # Удаляем файл

    # Удаляем запись из БД
    cur.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if filename.lower().endswith(("png", "jpg", "jpeg", "webp")):
        img = Image.open(file.stream)
        img = img.convert("RGB")  # Конвертируем в RGB (если PNG, уберем альфа-канал)
        img.save(file_path, "JPEG", quality=70)  # Сохраняем с качеством 70% (можно менять)
    else:
        file.save(file_path)  # Просто сохраняем файл, если это не картинка

    # Отдаем URL файла для отображения
    file_url = f"http://127.0.0.1:5000/static/uploads/{filename}"
    return jsonify({"status": "ok", "file_url": file_url})


@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    socketio.run(app, debug=True)
