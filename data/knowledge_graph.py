import sqlalchemy
from .session import SqlAlchemyBase
from sqlalchemy import orm


class KnowledgeNode(SqlAlchemyBase):
    """Узлы графа знаний пользователя"""
    
    __tablename__ = 'knowledge_nodes'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    user_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id"), nullable=False)
    title = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    description = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    category = sqlalchemy.Column(sqlalchemy.String, nullable=True)  # work, hobby, want, etc.
    x = sqlalchemy.Column(sqlalchemy.Float, default=0.0)  # Позиция X на графе
    y = sqlalchemy.Column(sqlalchemy.Float, default=0.0)  # Позиция Y на графе
    
    user = orm.relationship("User", backref="knowledge_nodes")
    
    # Связи между узлами
    connections = orm.relationship(
        "KnowledgeConnection",
        foreign_keys="KnowledgeConnection.from_node_id",
        back_populates="from_node"
    )


class KnowledgeConnection(SqlAlchemyBase):
    """Связи между узлами графа"""
    
    __tablename__ = 'knowledge_connections'

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, autoincrement=True)
    from_node_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("knowledge_nodes.id"), nullable=False)
    to_node_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("knowledge_nodes.id"), nullable=False)
    label = sqlalchemy.Column(sqlalchemy.String, nullable=True)  # Подпись связи
    
    from_node = orm.relationship("KnowledgeNode", foreign_keys=[from_node_id], back_populates="connections")
    to_node = orm.relationship("KnowledgeNode", foreign_keys=[to_node_id])

