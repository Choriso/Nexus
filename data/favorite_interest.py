import sqlalchemy
from .session import SqlAlchemyBase


class FavoriteInterest(SqlAlchemyBase):
    """Связь many-to-many между пользователями и понравившимися интересами"""
    
    __tablename__ = 'favorite_interests'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    user_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=False)
    interest_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("interests.id"), nullable=False)
    
    # Уникальная пара пользователь-интерес
    __table_args__ = (
        sqlalchemy.UniqueConstraint('user_id', 'interest_id', name='unique_user_interest'),
    )

