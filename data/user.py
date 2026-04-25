import sqlalchemy
from .session import SqlAlchemyBase
from sqlalchemy import orm
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin


class User(SqlAlchemyBase, UserMixin):

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.hashed_password, password)

    __tablename__ = 'users'

    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    name = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    email = sqlalchemy.Column(sqlalchemy.String,
                              index=True, unique=True, nullable=True)
    hashed_password = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    information = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    connection = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    image_path = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    allow_location = sqlalchemy.Column(sqlalchemy.Boolean, default=False)
    is_moderator = sqlalchemy.Column(sqlalchemy.Boolean, default=False)

    interests = orm.relationship("Interest", back_populates='user')
    messages = orm.relationship("Message", back_populates="author")
    personality_profile = orm.relationship("UserPersonalityProfile", backref="user", uselist=False, lazy='joined')
    extracted_interests = orm.relationship("AIExtractedInterests", backref="user", uselist=False)
    compatibilities_as_user1 = orm.relationship(
        "UserCompatibility",
        foreign_keys="UserCompatibility.user_id_1",
        backref="user1",
    )
    compatibilities_as_user2 = orm.relationship(
        "UserCompatibility",
        foreign_keys="UserCompatibility.user_id_2",
        backref="user2",
    )

    
    # Избранные интересы через промежуточную таблицу
    favorite_interests = orm.relationship(
        "Interest",
        secondary="favorite_interests",
        lazy="dynamic"
    )

