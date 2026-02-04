from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from flask_login import login_required, current_user

from data.interest import Interest
from data.report import Report
from data.user import User
from app.db import get_db_session
import sqlalchemy as sa
from sqlalchemy import orm

moderation_bp = Blueprint("moderation", __name__)


def is_moderator():
    """Проверка, является ли пользователь модератором"""
    if not current_user.is_authenticated:
        return False
    with get_db_session() as db_sess:
        user = db_sess.get(User, current_user.id)
        return user and user.is_moderator


@moderation_bp.route("/moderation")
@login_required
def moderation_page():
    """Страница модерации"""
    if not is_moderator():
        return jsonify({"error": "Доступ запрещён"}), 403
    
    with get_db_session() as db_sess:
        # Получаем все жалобы со статусом pending
        reports = db_sess.query(Report).options(
            sa.orm.joinedload(Report.interest).joinedload(Interest.user),
            sa.orm.joinedload(Report.reporter)
        ).filter(Report.status == "pending").order_by(Report.created_at.desc()).all()
        
        # Получаем все жалобы для истории
        all_reports = db_sess.query(Report).options(
            sa.orm.joinedload(Report.interest).joinedload(Interest.user),
            sa.orm.joinedload(Report.reporter)
        ).order_by(Report.created_at.desc()).limit(50).all()
    
    return render_template("moderation.html", reports=reports, all_reports=all_reports, current_user=current_user)


@moderation_bp.route("/moderation/report/<int:report_id>/review", methods=["POST"])
@login_required
def review_report(report_id):
    """Обработка жалобы модератором"""
    if not is_moderator():
        return jsonify({"success": False, "message": "Доступ запрещён"}), 403
    
    data = request.json
    action = data.get("action")  # dismiss, delete, warn
    moderator_comment = data.get("moderator_comment", "")
    
    with get_db_session() as db_sess:
        report = db_sess.query(Report).filter(Report.id == report_id).first()
        if not report:
            return jsonify({"success": False, "message": "Жалоба не найдена"}), 404
        
        report.status = "reviewed"
        report.moderator_comment = moderator_comment
        report.reviewed_by = current_user.id
        from datetime import datetime
        report.reviewed_at = datetime.utcnow()
        
        if action == "delete":
            # Удаляем интерес
            interest = db_sess.query(Interest).filter(Interest.id == report.interest_id).first()
            if interest:
                db_sess.delete(interest)
        
        db_sess.commit()
    
    return jsonify({"success": True, "message": "Жалоба обработана"})


@moderation_bp.route("/report/<int:interest_id>", methods=["POST"])
@login_required
def create_report(interest_id):
    """Создание жалобы на интерес"""
    data = request.json
    reason = data.get("reason", "")
    description = data.get("description", "")
    
    with get_db_session() as db_sess:
        # Проверяем, существует ли интерес
        interest = db_sess.query(Interest).filter(Interest.id == interest_id).first()
        if not interest:
            return jsonify({"success": False, "message": "Интерес не найден"}), 404
        
        # Проверяем, не жаловался ли уже пользователь на этот интерес
        existing_report = db_sess.query(Report).filter(
            Report.interest_id == interest_id,
            Report.reporter_id == current_user.id,
            Report.status == "pending"
        ).first()
        
        if existing_report:
            return jsonify({"success": False, "message": "Вы уже отправили жалобу на этот интерес"}), 400
        
        # Создаём жалобу
        report = Report(
            interest_id=interest_id,
            reporter_id=current_user.id,
            reason=reason,
            description=description
        )
        db_sess.add(report)
        db_sess.commit()
    
    return jsonify({"success": True, "message": "Жалоба отправлена"})


@moderation_bp.route("/moderation/stats", methods=["GET"])
@login_required
def moderation_stats():
    """Статистика для модерации"""
    if not is_moderator():
        return jsonify({"error": "Доступ запрещён"}), 403
    
    with get_db_session() as db_sess:
        pending_count = db_sess.query(Report).filter(Report.status == "pending").count()
        total_reports = db_sess.query(Report).count()
        resolved_count = db_sess.query(Report).filter(Report.status == "resolved").count()
    
    return jsonify({
        "pending": pending_count,
        "total": total_reports,
        "resolved": resolved_count
    })

