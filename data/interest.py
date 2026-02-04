import sqlalchemy
from .session import SqlAlchemyBase
from sqlalchemy import orm


class Interest(SqlAlchemyBase):

    __tablename__ = 'interests'

    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    title = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    description = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    user_id = sqlalchemy.Column(sqlalchemy.Integer,
                                sqlalchemy.ForeignKey("users.id"))
    user = orm.relationship('User', back_populates='interests')

