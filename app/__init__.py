import os

from flask import Flask

from .extensions import login_manager, socketio, cors
from data import session as db_session
from data.user import User


def create_app() -> Flask:
    """Фабрика приложения Flask. Настраивает конфигурацию и расширения."""
    # Явно указываем папки с шаблонами и статикой, которые находятся на уровень выше пакета app
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///chat.db")
    app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "static/uploads/")

    _secret = os.environ.get("SECRET_KEY")
    _default_dev_secret = "dev_secret_key_change_me"
    if os.environ.get("FLASK_ENV") == "production" and not _secret:
        raise ValueError(
            "SECRET_KEY must be set in production (environment variable)."
        )
    app.config["SECRET_KEY"] = _secret or _default_dev_secret

    db_session.global_init(app.config["SQLALCHEMY_DATABASE_URI"])

    max_content_length_env = os.environ.get("MAX_CONTENT_LENGTH")
    if max_content_length_env is not None:
        try:
            app.config["MAX_CONTENT_LENGTH"] = int(max_content_length_env)
        except ValueError:
            app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    else:
        app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # максимум 16 МБ

    # Инициализация расширений (однократно)
    login_manager.login_view = ""
    login_manager.init_app(app)

    # CORS настройки для доступа из локальной сети
    # Можно настроить через переменную окружения ALLOWED_ORIGINS (через запятую)
    # По умолчанию разрешаем все для удобства разработки в локальной сети
    allowed_origins_env = os.environ.get("ALLOWED_ORIGINS")
    if allowed_origins_env:
        # Если указаны конкретные origins через переменную окружения
        allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]
    else:
        # По умолчанию разрешаем все для локальной сети (dev режим)
        allowed_origins = "*"
    
    socketio.init_app(app, cors_allowed_origins=allowed_origins)
    cors.init_app(
        app,
        resources={
            r"/*": {
                "origins": allowed_origins,
            }
        },
    )

    # Загрузка пользователя
    @login_manager.user_loader
    def load_user(user_id):
        db_sess = db_session.create_session()
        return db_sess.get(User, user_id)

    from .routes import main_bp  # Создадим его сейчас
    app.register_blueprint(main_bp)
    from .auth import auth_bp
    from .profile import profile_bp
    from .interests import interests_bp
    from .chat import chat_bp
    from .moderation import moderation_bp
    from .analytics import analytics_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(interests_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(moderation_bp)
    app.register_blueprint(analytics_bp)

    return app


