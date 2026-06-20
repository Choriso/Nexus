from flask import Blueprint, render_template, request, redirect, abort, url_for, jsonify
from flask_login import login_required, current_user
from sqlalchemy.orm import joinedload

from data.interest import Interest
from data.chat import Chat
from data.favorite_interest import FavoriteInterest
from app.db import get_db_session
import sqlalchemy as sa

interests_bp = Blueprint("interests", __name__)


@interests_bp.route("/intereses")
def index():
    """
    Главная страница интересов. Показывает список интересов пользователей.

    Returns:
        flask.Response: Страница с интересами с учетом авторизации пользователя и избранного.
    """
    with get_db_session() as db_sess:
        query = db_sess.query(Interest).options(joinedload(Interest.user))
        if current_user.is_authenticated:
            interest = query.filter(Interest.user_id != current_user.id).all()
            favorite_ids = {fav.interest_id for fav in db_sess.query(FavoriteInterest)
                                                        .filter(FavoriteInterest.user_id == current_user.id)
                                                        .all()}
        else:
            interest = query.all()
            favorite_ids = set()
        interest = list(reversed(interest))
    return render_template("index.html", interest=interest, current_user=current_user, favorite_ids=favorite_ids)


@interests_bp.route("/viewInteres", methods=["GET"])
def view_interest():
    """
    Страница просмотра определенного интереса.

    Returns:
        flask.Response: Страница с информацией об интересе и пользователе-авторе.

    Raises:
        404: Если интерес или его автор не найдены.
    """
    user_id = request.args.get("user_id")
    interest_id = request.args.get("interest_id")
    with get_db_session() as db_sess:
        interest = (
            db_sess.query(Interest)
            .options(joinedload(Interest.user))
            .filter(Interest.id == interest_id)
            .first()
        )
        if not interest or not interest.user:
            abort(404)
        user = interest.user
    return render_template("view_interes.html", title="", interests=interest, user=user)


@interests_bp.route("/interest", methods=["GET", "POST"])
@login_required
def add_interest():
    """
    Страница добавления нового интереса для авторизованного пользователя.

    Returns:
        flask.Response: HTML-форма для ввода интереса.
    """
    return render_template("interest.html", title="Добавление интереса")


@interests_bp.route("/process_interest", methods=["POST"], endpoint="process_interest")
@login_required
def process_interest():
    """
    Обработка формы добавления интереса. Добавляет интерес текущему пользователю.

    Returns:
        flask.Response: Перенаправление на список интересов.
    """
    title = request.form["title"]
    description = request.form["description"]
    with get_db_session() as db_sess:
        user = db_sess.merge(current_user)
        interest = Interest()
        interest.title = title
        interest.description = description
        user.interests.append(interest)
        db_sess.commit()
    return redirect("/intereses")


@interests_bp.route("/interest/<int:id>", methods=["GET", "POST"])
@login_required
def edit_interest(id: int):
    """
    Редактирование интереса пользователя.

    Args:
        id (int): ID интереса для редактирования.

    Returns:
        flask.Response: Форма редактирования или результат обновления и перенаправление.

    Raises:
        404: Если интерес не найден или не принадлежит пользователю.
    """
    from forms.interest import InterestForm
    form = InterestForm()
    if request.method == "GET":
        with get_db_session() as db_sess:
            interest = db_sess.query(Interest).filter(
                Interest.id == id, Interest.user == current_user
            ).first()
            if interest:
                form.title.data = interest.title
                form.description.data = interest.description
                form.tags.data = getattr(interest, "tags", "")
            else:
                abort(404)
    if form.validate_on_submit():
        with get_db_session() as db_sess:
            interest = db_sess.query(Interest).filter(
                Interest.id == id, Interest.user == current_user
            ).first()
            if interest:
                interest.title = form.title.data
                interest.description = form.description.data
                interest.tags = form.tags.data
                db_sess.commit()
                return redirect("/intereses")
            else:
                abort(404)
    return render_template("interest.html", title="Редактирование интереса", form=form)


@interests_bp.route("/interest_delete/<int:id>", methods=["GET", "POST"])
@login_required
def delete_interest(id: int):
    """
    Удаляет интерес пользователя по ID.

    Args:
        id (int): ID интереса для удаления.

    Returns:
        flask.Response: Перенаправление на предыдущую страницу или список интересов.

    Raises:
        404: Если интерес не найден или не принадлежит текущему пользователю.
    """
    with get_db_session() as db_sess:
        interest = db_sess.query(Interest).filter(
            Interest.id == id,
            Interest.user == current_user,
        ).first()
        if interest:
            db_sess.delete(interest)
            db_sess.commit()
        else:
            abort(404)
    previous_page = request.referrer
    if previous_page:
        return redirect(previous_page)
    else:
        return redirect(url_for("interests.index"))


@interests_bp.route("/search", methods=["GET"])
def search():
    """
    Поиск по интересам по заголовку или описанию.

    Returns:
        flask.Response: HTML-страница с результатами поиска.
    """
    query_text = request.args.get("q", "").strip().lower()
    if not query_text or query_text == "все":
        return redirect(url_for("interests.index"))
    with get_db_session() as db_sess:
        results = (
            db_sess.query(Interest)
            .join(Interest.user)
            .options(joinedload(Interest.user))
            .filter(
                (Interest.title.ilike(f"%{query_text}%")) |
                (Interest.description.ilike(f"%{query_text}%"))
            )
            .all()
        )
        favorite_ids = set()
        if current_user.is_authenticated:
            favorite_ids = {fav.interest_id for fav in db_sess.query(FavoriteInterest)
                                                        .filter(FavoriteInterest.user_id == current_user.id)
                                                        .all()}
        return render_template("index.html",
                               interest=results,
                               current_user=current_user,
                               favorite_ids=favorite_ids,
                               query=query_text)


@interests_bp.route("/create_chat/<int:user_id>", methods=["POST"])
@login_required
def create_chat_with_user(user_id: int):
    """
    Создает или получает существующий чат с указанным пользователем.

    Args:
        user_id (int): ID пользователя, с которым создается чат.

    Returns:
        flask.Response: JSON с успехом и chat_id. Ошибка, если пытаются создать чат с собой.
    """
    if user_id == current_user.id:
        return jsonify({"success": False, "message": "Нельзя создать чат с самим собой"}), 400
    with get_db_session() as db_sess:
        chat = db_sess.execute(
            sa.select(Chat).where(
                ((Chat.user1_id == current_user.id) & (Chat.user2_id == user_id)) |
                ((Chat.user1_id == user_id) & (Chat.user2_id == current_user.id))
            )
        ).scalar()
        if not chat:
            chat = Chat(user1_id=current_user.id, user2_id=user_id)
            db_sess.add(chat)
            db_sess.commit()
        return jsonify({"success": True, "chat_id": chat.id})


@interests_bp.route("/favorite/<int:interest_id>", methods=["POST"])
@login_required
def toggle_favorite(interest_id: int):
    """
    Добавляет или удаляет интерес из избранного для текущего пользователя.

    Args:
        interest_id (int): ID интереса.

    Returns:
        flask.Response: JSON-индикатор успешности и состояние избранного.
    """
    with get_db_session() as db_sess:
        interest = db_sess.query(Interest).filter(Interest.id == interest_id).first()
        if not interest:
            return jsonify({"success": False, "message": "Интерес не найден"}), 404
        favorite = db_sess.query(FavoriteInterest).filter(
            FavoriteInterest.user_id == current_user.id,
            FavoriteInterest.interest_id == interest_id
        ).first()
        if favorite:
            db_sess.delete(favorite)
            db_sess.commit()
            return jsonify({"success": True, "is_favorite": False, "message": "Удалено из избранного"})
        else:
            favorite = FavoriteInterest(user_id=current_user.id, interest_id=interest_id)
            db_sess.add(favorite)
            db_sess.commit()
            return jsonify({"success": True, "is_favorite": True, "message": "Добавлено в избранное"})


@interests_bp.route("/favorites")
@login_required
def favorites():
    """
    Страница с избранными интересами пользователя.

    Returns:
        flask.Response: HTML-страница только с избранными интересами.
    """
    with get_db_session() as db_sess:
        favorites_query = db_sess.query(FavoriteInterest).filter(
            FavoriteInterest.user_id == current_user.id
        )
        favorite_ids = [fav.interest_id for fav in favorites_query.all()]
        interests = db_sess.query(Interest).options(joinedload(Interest.user)).filter(
            Interest.id.in_(favorite_ids)
        ).all()
        authors = list(set([interest.user for interest in interests]))
    return render_template("index.html", interest=interests, current_user=current_user, show_favorites=True)


@interests_bp.route("/api/graph/match/<int:node_id>")
@login_required
def match_by_node(node_id: int):
    """
    Возвращает пользователей с похожими знаниями и рейтинг совместимости по психологии.

    Args:
        node_id (int): ID выбранного узла знаний.

    Returns:
        flask.Response: JSON с массивом пользователей и показателем совместимости.
        404: Если узел не найден.
    """
    with get_db_session() as db_sess:
        target_node = db_sess.query(KnowledgeNode).get(node_id)
        if not target_node:
            return jsonify({"error": "Node not found"}), 404
        similar_nodes = db_sess.query(KnowledgeNode).filter(
            KnowledgeNode.title.ilike(f"%{target_node.title}%"),
            KnowledgeNode.user_id != current_user.id
        ).all()
        my_profile = db_sess.query(UserPersonalityProfile).filter_by(user_id=current_user.id).first()
        matches = []
        for node in similar_nodes:
            other_user = node.user
            other_profile = db_sess.query(UserPersonalityProfile).filter_by(user_id=other_user.id).first()
            base_score = 0.8
            psy_score = 0.5
            if my_profile and other_profile:
                from app.ai_profiler.core import AIProfiler
                profiler = AIProfiler()
                my_vec = [my_profile.openness, my_profile.conscientiousness, my_profile.extraversion,
                          my_profile.agreeableness, my_profile.neuroticism]
                other_vec = [other_profile.openness, other_profile.conscientiousness, other_profile.extraversion,
                             other_profile.agreeableness, other_profile.neuroticism]
                psy_score = profiler.calculate_compatibility(my_vec, other_vec) / 100
            total_score = (base_score * 0.6) + (psy_score * 0.4)
            matches.append({
                "user_id": other_user.id,
                "user_name": other_user.name,
                "node_title": node.title,
                "compatibility": round(total_score * 100, 1),
                "category": node.category
            })
        matches = sorted(matches, key=lambda x: x['compatibility'], reverse=True)
        return jsonify(matches)