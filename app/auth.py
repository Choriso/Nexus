from flask import Blueprint, request, jsonify, redirect
from flask_login import login_user, logout_user, login_required

from data.user import User
from app.db import get_db_session
import re

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Обрабатывает регистрацию нового пользователя.

    Args:
        Нет. Данные берутся из тела POST-запроса (JSON):
            - email (str): Email пользователя.
            - password (str): Пароль.
            - fullName (str): Имя пользователя.
            - allowLocation (bool | int, optional): Разрешение на доступ к геолокации.

    Returns:
        flask.Response: JSON-ответ с признаком успеха и/или сообщением об ошибке.

    Возможные ошибки:
        - Некорректный email.
        - Пользователь уже зарегистрирован.
    """
    data = request.json
    email = data.get("email")
    password = data.get("password")
    full_name = data.get("fullName")
    allow_location = data.get("allowLocation", 0)

    with get_db_session() as db_sess:
        if db_sess.query(User).filter(User.email == email).first():
            return jsonify({"success": False, "message": "Такой пользователь уже зарегистрирован."})

        email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        if not re.match(email_regex, email or ""):
            return jsonify({"success": False, "message": "Некорректный формат email."})

        user = User(
            name=full_name,
            email=email,
            allow_location=bool(allow_location),
        )
        user.set_password(password)

        db_sess.add(user)
        db_sess.commit()

        login_user(user)
        return jsonify({"success": True})


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Обрабатывает вход пользователя (логин).

    Args:
        Нет. Данные берутся из тела POST-запроса (JSON):
            - email (str): Email пользователя.
            - password (str): Пароль.

    Returns:
        flask.Response: JSON-ответ с признаком успеха и/или сообщением об ошибке.

    Возможные ошибки:
        - Пользователь не найден или неверный пароль.
    """
    data = request.json
    email = data.get("email")
    password = data.get("password")

    with get_db_session() as db_sess:
        user = db_sess.query(User).filter(User.email == email).first()

        if not user or not user.check_password(password):
            return jsonify({"success": False, "message": "Неверный email или пароль."})

        login_user(user)
        return jsonify({"success": True, "message": "Вход выполнен успешно."})


@auth_bp.route("/logout")
@login_required
def logout():
    """
    Выходит из аккаунта текущего пользователя.

    Args:
        Нет.

    Returns:
        flask.Response: Редирект на главную страницу сайта ("/").
    """
    logout_user()
    return redirect("/")