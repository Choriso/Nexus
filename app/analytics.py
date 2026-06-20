from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user

from data.user import User
from data.interest import Interest
from data.message import Message
from data.chat import Chat
from data.favorite_interest import FavoriteInterest
from data.report import Report
from app.db import get_db_session
import sqlalchemy as sa
from datetime import datetime, timedelta

analytics_bp = Blueprint("analytics", __name__)


def is_moderator() -> bool:
    """
    Проверяет, является ли текущий пользователь модератором.

    Returns:
        bool: True, если пользователь является модератором и аутентифицирован, иначе False.
    """
    if not current_user.is_authenticated:
        return False
    with get_db_session() as db_sess:
        user = db_sess.get(User, current_user.id)
        return user and user.is_moderator


@analytics_bp.route("/analytics")
@login_required
def analytics_page():
    """
    Выводит HTML-страницу аналитики для модератора.

    Returns:
        flask.Response: HTML-страница с основными метриками платформы и статистическими таблицами.
        Если пользователь не модератор — JSON с ошибкой 403.
    """
    if not is_moderator():
        return jsonify({"error": "Доступ запрещён"}), 403

    with get_db_session() as db_sess:
        total_users = db_sess.query(User).count()
        total_interests = db_sess.query(Interest).count()
        total_messages = db_sess.query(Message).count()
        total_chats = db_sess.query(Chat).count()
        total_favorites = db_sess.query(FavoriteInterest).count()
        total_reports = db_sess.query(Report).count()
        pending_reports = db_sess.query(Report).filter(Report.status == "pending").count()

        # (NB: Для корректной аналитики желательно использовать поле created_at.)
        new_users_by_day = db_sess.query(
            sa.func.date(User.id).label('date'),
            sa.func.count(User.id).label('count')
        ).group_by(sa.func.date(User.id)).all()

        new_interests_by_day = db_sess.query(
            sa.func.date(Interest.id).label('date'),
            sa.func.count(Interest.id).label('count')
        ).group_by(sa.func.date(Interest.id)).all()

        top_authors = db_sess.query(
            User.name,
            sa.func.count(Interest.id).label('count')
        ).join(Interest).group_by(User.id, User.name).order_by(
            sa.func.count(Interest.id).desc()
        ).limit(10).all()

        top_interests = db_sess.query(
            Interest.title,
            sa.func.count(FavoriteInterest.id).label('favorites_count')
        ).join(FavoriteInterest).group_by(Interest.id, Interest.title).order_by(
            sa.func.count(FavoriteInterest.id).desc()
        ).limit(10).all()

        reports_by_reason = db_sess.query(
            Report.reason,
            sa.func.count(Report.id).label('count')
        ).group_by(Report.reason).all()

    return render_template(
        "analytics.html",
        total_users=total_users,
        total_interests=total_interests,
        total_messages=total_messages,
        total_chats=total_chats,
        total_favorites=total_favorites,
        total_reports=total_reports,
        pending_reports=pending_reports,
        top_authors=top_authors,
        top_interests=top_interests,
        reports_by_reason=reports_by_reason,
        current_user=current_user
    )


@analytics_bp.route("/analytics/api/data", methods=["GET"])
@login_required
def analytics_api():
    """
    API-эндпоинт для получения агрегированных аналитических данных для графиков.

    Returns:
        flask.Response: JSON с массивами дней и соответствующими данными по новым пользователям и интересам.
        Если пользователь не модератор — JSON с ошибкой 403.
    """
    if not is_moderator():
        return jsonify({"error": "Доступ запрещён"}), 403

    with get_db_session() as db_sess:
        days = []
        users_data = []
        interests_data = []

        for i in range(30):
            day = datetime.utcnow() - timedelta(days=30 - i)
            days.append(day.strftime("%Y-%m-%d"))
            # Здесь должны быть реальные данные по дате создания пользователей и интересов.
            users_data.append(0)
            interests_data.append(0)

        return jsonify({
            "days": days,
            "users": users_data,
            "interests": interests_data
        })
