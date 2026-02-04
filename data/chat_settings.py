import sqlalchemy
from .session import SqlAlchemyBase
from sqlalchemy import orm


class ChatSettings(SqlAlchemyBase):
    """Настройки чата для пользователя"""
    
    __tablename__ = 'chat_settings'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    user_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=False, unique=True)
    sound_enabled = sqlalchemy.Column(sqlalchemy.Boolean, default=True)
    notifications_enabled = sqlalchemy.Column(sqlalchemy.Boolean, default=True)
    theme = sqlalchemy.Column(sqlalchemy.String, default="dark")  # dark, light
    font_size = sqlalchemy.Column(sqlalchemy.Integer, default=14)
    
    user = orm.relationship("User", backref="chat_settings")

