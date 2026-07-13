import logging

from flask import Flask

from config import get_config
from .extensions import login_manager, socketio, cors, migrate
from data import session as db_session
from data.user import User

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")


def create_app() -> Flask:
    """
    Создаёт и конфигурирует экземпляр Flask-приложения.

    Настраивает базу данных, расширения, лимиты, CORS и загружает blueprints.

    Returns:
        Flask: Экземпляр сконфигурированного Flask-приложения.
    """
    cfg = get_config()

    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    if cfg.is_production() and not cfg.SECRET_KEY:
        raise ValueError(
            "SECRET_KEY must be set in production (environment variable)."
        )

    app.config["SQLALCHEMY_DATABASE_URI"] = cfg.SQLALCHEMY_DATABASE_URI
    app.config["UPLOAD_FOLDER"] = cfg.UPLOAD_FOLDER
    app.config["SECRET_KEY"] = cfg.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_CONTENT_LENGTH

    db_session.global_init(app.config["SQLALCHEMY_DATABASE_URI"])

    from app.ai_profiler.interest_graph import ensure_hierarchy_seeded
    with db_session.create_session() as sess:
        ensure_hierarchy_seeded(sess)

    login_manager.login_view = ""
    login_manager.init_app(app)

    allowed_origins = cfg.allowed_origins_list()

    socketio.init_app(app, cors_allowed_origins=allowed_origins)
    migrate.init_app(app)
    cors.init_app(
        app,
        resources={
            r"/*": {
                "origins": allowed_origins,
            }
        },
    )

    @login_manager.user_loader
    def load_user(user_id: int) -> User | None:
        """
        Загружает пользователя по ID для Flask-Login.

        Args:
            user_id (int): Идентификатор пользователя.

        Returns:
            User | None: Объект пользователя или None, если не найден.
        """
        db_sess = db_session.create_session()
        return db_sess.get(User, user_id)

    from .routes import main_bp
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
