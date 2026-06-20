import sqlalchemy
from .session import SqlAlchemyBase


class FavoriteInterest(SqlAlchemyBase):    
    __tablename__ = 'favorite_interests'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    user_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=False)
    interest_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("interests.id"), nullable=False)
    
    __table_args__ = (
        sqlalchemy.UniqueConstraint('user_id', 'interest_id', name='unique_user_interest'),
    )

