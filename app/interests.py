from flask import Blueprint, render_template, request, redirect, abort, url_for, jsonify
from flask_login import login_required, current_user
from sqlalchemy.orm import joinedload

from data.interest import Interest
from data.chat import Chat
from data.favorite_interest import FavoriteInterest
from app.db import get_db_session
from get_similar import line_vector, cosdis
import sqlalchemy as sa

interests_bp = Blueprint("interests", __name__)


@interests_bp.route("/intereses")
def index():
    with get_db_session() as db_sess:
        # Используем joinedload для eager loading связанных пользователей
        query = db_sess.query(Interest).options(joinedload(Interest.user))
        
        if current_user.is_authenticated:
            interest = query.filter(Interest.user_id != current_user.id).all()
            # Получаем ID избранных интересов для текущего пользователя
            favorite_ids = {fav.interest_id for fav in db_sess.query(FavoriteInterest).filter(
                FavoriteInterest.user_id == current_user.id
            ).all()}
        else:
            interest = query.all()
            favorite_ids = set()
        
        # Переворачиваем список, чтобы новые были первыми
        interest = list(reversed(interest))
    
    return render_template("index.html", interest=interest, current_user=current_user, favorite_ids=favorite_ids)


@interests_bp.route("/viewInteres", methods=["GET"])
def view_interest():
    user_id = request.args.get("user_id")
    interest_id = request.args.get("interest_id")

    with get_db_session() as db_sess:
        # Используем joinedload для загрузки пользователя вместе с интересом
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
    return render_template("interest.html", title="Добавление интереса")


@interests_bp.route("/process_interest", methods=["POST"], endpoint="process_interest")
@login_required
def process_interest():
    title = request.form["title"]
    description = request.form["description"]

    with get_db_session() as db_sess:
        user = db_sess.merge(current_user)  # привязываем current_user к сессии
        interest = Interest()
        interest.title = title
        interest.description = description
        user.interests.append(interest)
        db_sess.commit()
    return redirect("/intereses")


@interests_bp.route("/interest/<int:id>", methods=["GET", "POST"])
@login_required
def edit_interest(id):
    from forms.interest import InterestForm

    form = InterestForm()
    if request.method == "GET":
        with get_db_session() as db_sess:
            interest = db_sess.query(Interest).filter(Interest.id == id, Interest.user == current_user).first()
            if interest:
                form.title.data = interest.title
                form.description.data = interest.description
                form.tags.data = getattr(interest, "tags", "")
            else:
                abort(404)
    if form.validate_on_submit():
        with get_db_session() as db_sess:
            interest = db_sess.query(Interest).filter(Interest.id == id, Interest.user == current_user).first()
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
def delete_interest(id):
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
    SIMILAR_RATIO = 0.5
    query = request.args.get("q")
    if query == "все":
        return redirect("/intereses")

    with get_db_session() as db_sess:
        if query:
            vector_query = line_vector(query)
            # Используем joinedload для загрузки пользователей вместе с интересами
            all_interests = db_sess.query(Interest).options(joinedload(Interest.user)).all()
            interest_searched = {}
            for i in all_interests:
                tittle_cos = cosdis(vector_query, line_vector(i.title))
                disc_cos = cosdis(vector_query, line_vector(i.description))
                if tittle_cos > SIMILAR_RATIO:
                    interest_searched[i] = tittle_cos
                elif disc_cos > SIMILAR_RATIO:
                    interest_searched[i] = disc_cos
            sorted_interests = [i[0] for i in sorted(interest_searched.items(), key=lambda item: item[1])][::-1]
            return render_template("chat.html", interest=sorted_interests)


@interests_bp.route("/create_chat/<int:user_id>", methods=["POST"])
@login_required
def create_chat_with_user(user_id):
    """Создаёт или получает существующий чат с пользователем"""
    if user_id == current_user.id:
        return jsonify({"success": False, "message": "Нельзя создать чат с самим собой"}), 400
    
    with get_db_session() as db_sess:
        # Проверяем, существует ли уже чат
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
def toggle_favorite(interest_id):
    """Добавляет или удаляет интерес из избранного"""
    with get_db_session() as db_sess:
        # Проверяем, существует ли интерес
        interest = db_sess.query(Interest).filter(Interest.id == interest_id).first()
        if not interest:
            return jsonify({"success": False, "message": "Интерес не найден"}), 404
        
        # Проверяем, есть ли уже в избранном
        favorite = db_sess.query(FavoriteInterest).filter(
            FavoriteInterest.user_id == current_user.id,
            FavoriteInterest.interest_id == interest_id
        ).first()
        
        if favorite:
            # Удаляем из избранного
            db_sess.delete(favorite)
            db_sess.commit()
            return jsonify({"success": True, "is_favorite": False, "message": "Удалено из избранного"})
        else:
            # Добавляем в избранное
            favorite = FavoriteInterest(user_id=current_user.id, interest_id=interest_id)
            db_sess.add(favorite)
            db_sess.commit()
            return jsonify({"success": True, "is_favorite": True, "message": "Добавлено в избранное"})


@interests_bp.route("/favorites")
@login_required
def favorites():
    """Страница с избранными интересами"""
    with get_db_session() as db_sess:
        # Получаем избранные интересы текущего пользователя
        favorites_query = db_sess.query(FavoriteInterest).filter(
            FavoriteInterest.user_id == current_user.id
        )
        
        # Получаем ID избранных интересов
        favorite_ids = [fav.interest_id for fav in favorites_query.all()]
        
        # Загружаем интересы с авторами
        interests = db_sess.query(Interest).options(joinedload(Interest.user)).filter(
            Interest.id.in_(favorite_ids)
        ).all()
        
        # Получаем уникальных авторов
        authors = list(set([interest.user for interest in interests]))
    
    return render_template("index.html", interest=interests, current_user=current_user, show_favorites=True)


