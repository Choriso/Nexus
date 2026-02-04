import sqlalchemy
from .session import SqlAlchemyBase
from sqlalchemy import orm
from datetime import datetime


class Report(SqlAlchemyBase):
    """Жалобы на интересы"""
    
    __tablename__ = 'reports'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    interest_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("interests.id"), nullable=False)
    reporter_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=False)
    reason = sqlalchemy.Column(sqlalchemy.String, nullable=True)  # Причина жалобы
    description = sqlalchemy.Column(sqlalchemy.String, nullable=True)  # Описание проблемы
    status = sqlalchemy.Column(sqlalchemy.String, default="pending")  # pending, reviewed, resolved, dismissed
    moderator_comment = sqlalchemy.Column(sqlalchemy.String, nullable=True)  # Комментарий модератора автору
    created_at = sqlalchemy.Column(sqlalchemy.DateTime, default=datetime.utcnow)
    reviewed_at = sqlalchemy.Column(sqlalchemy.DateTime, nullable=True)
    reviewed_by = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=True)
    
    interest = orm.relationship("Interest", backref="reports")
    reporter = orm.relationship("User", foreign_keys=[reporter_id], backref="reports_made")
    moderator = orm.relationship("User", foreign_keys=[reviewed_by], backref="reports_reviewed")

