"""Иерархический граф интересов (Materialized Path + parent-child)."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from .session import SqlAlchemyBase


class InterestHierarchyNode(SqlAlchemyBase):
    """
    Узел иерархии интересов.

    path — materialized path вида /1/5/12/ для быстрого обхода предков.
    match_weight — доля совпадения при прямом попадании в этот узел (0..1).
    """

    __tablename__ = "interest_hierarchy_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    slug = Column(String(200), nullable=False, unique=True, index=True)
    parent_id = Column(Integer, ForeignKey("interest_hierarchy_nodes.id"), nullable=True)
    path = Column(String(500), nullable=False, default="/")
    depth = Column(Integer, default=0)
    match_weight = Column(Float, default=1.0)
    global_category = Column(String(50), nullable=True)

    parent = relationship("InterestHierarchyNode", remote_side=[id], backref="children")
    user_weights = relationship("UserInterestGraphWeight", back_populates="node")


class UserInterestGraphWeight(SqlAlchemyBase):
    """Накопленные веса интересов пользователя по узлам графа."""

    __tablename__ = "user_interest_graph_weights"
    __table_args__ = (
        UniqueConstraint("user_id", "node_id", name="uq_user_interest_node"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    node_id = Column(Integer, ForeignKey("interest_hierarchy_nodes.id"), nullable=False)
    weight = Column(Float, default=0.0)
    source_tag = Column(String(200), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    node = relationship("InterestHierarchyNode", back_populates="user_weights")
