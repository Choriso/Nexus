import os

from flask import make_response, session, url_for, render_template, request
from flask_login import current_user
from data import session as db_session

from app import create_app
from app.extensions import socketio

app = create_app()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "pdf"}


# запуск приложения
def main():
    # Путь к БД для ORM-слоя (users/interests/chats/messages)
    blogs_db_file = os.environ.get("BLOG_DB_FILE", "db/blogs.db")
    db_session.global_init(blogs_db_file)
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    
    # Хост и порт для запуска сервера
    # 0.0.0.0 позволяет подключаться с других устройств в локальной сети
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    
    print(f"🚀 Сервер запускается на http://{host}:{port}")
    print(f"📱 Для доступа с других устройств используй: http://[ТВОЙ_IP]:{port}")
    
    # Включаем allow_unsafe_werkzeug для локального dev-сервера
    socketio.run(app, host=host, port=port, debug=debug_mode, allow_unsafe_werkzeug=True)


@app.route("/", methods=['GET', 'POST'])
def start():
    return render_template("startScreen.html", current_user=current_user)


@app.route('/geolocation')
def geolocation():
    return render_template('geolocation_ip.html')


@app.route('/upload_render')
def upload_render():
    return render_template('upload_file.html')


@app.route("/cookie_test")
def cookie_test():
    visits_count = int(request.cookies.get("visits_count", 0))
    if visits_count:
        res = make_response(
            f"Вы пришли на эту страницу {visits_count + 1} раз")
        res.set_cookie("visits_count", str(visits_count + 1),
                       max_age=60 * 60 * 24 * 365 * 2)
    else:
        res = make_response(
            "Вы пришли на эту страницу в первый раз за последние 2 года")
        res.set_cookie("visits_count", '1',
                       max_age=60 * 60 * 24 * 365 * 2)
    return res


@app.route("/session_test")
def session_test():
    visits_count = session.get('visits_count', 0)
    session['visits_count'] = visits_count + 1
    return make_response(
        f"Вы пришли на эту страницу {visits_count + 1} раз")


if __name__ == "__main__":
    main()
