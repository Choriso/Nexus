import os

from flask import make_response, session, url_for, render_template, request
from flask_login import current_user
from data import session as db_session

from app import create_app
from app.extensions import socketio
from dotenv import load_dotenv
load_dotenv()

app = create_app()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "pdf"}


# запуск приложения
def main():
    # Используем DATABASE_URL из .env
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("⚠️ DATABASE_URL не найден в .env, использую SQLite")
        db_url = "sqlite:///db/blogs.db"

    print(f"📦 Подключение к БД: {db_url.split('@')[0].split('://')[0]}://...")
    db_session.global_init(db_url)

    # Настройки сервера из .env или значения по умолчанию
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "3000"))

    print(f"Сайт: http://192.168.3.41:{port}")
    socketio.run(app, host=host, port=port, debug=debug_mode, allow_unsafe_werkzeug=True)
    # app.run(host=host, port=port, debug=True)


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
