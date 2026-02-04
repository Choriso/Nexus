from data.session import SqlAlchemyBase
import sqlalchemy
from sqlalchemy import orm
from datetime import datetime


class Chat(SqlAlchemyBase):
    __tablename__ = "chats"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    user1_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"))
    user2_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"))

    messages = orm.relationship("Message", back_populates="chat")
