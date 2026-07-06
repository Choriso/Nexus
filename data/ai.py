from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from pgvector.sqlalchemy import Vector
from .session import SqlAlchemyBase


class UserPersonalityProfile(SqlAlchemyBase):
    __tablename__ = "ai_user_personality_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    openness = Column(Float, default=0.5)
    conscientiousness = Column(Float, default=0.5)
    extraversion = Column(Float, default=0.5)
    agreeableness = Column(Float, default=0.5)
    neuroticism = Column(Float, default=0.5)

    embedding = Column(Vector(5), nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    mbti_type = Column(String(4), nullable=True)
    communication_style = Column(String(50), nullable=True)
    formality = Column(Float, default=0.5)
    enthusiasm = Column(Float, default=0.5)
    detail_oriented = Column(Float, default=0.5)

    traits = Column(JSON, nullable=True)
    values = Column(JSON, nullable=True)
    compatible_mbti_types = Column(JSON, nullable=True)
    collaboration_style = Column(String(50), nullable=True)
    confidence_score = Column(Float, default=0.0)
    last_analyzed = Column(DateTime, default=datetime.utcnow)
    conversation_count = Column(Integer, default=0)

    def get_big_five_vector(self):
        return [
            self.openness or 0.5,
            self.conscientiousness or 0.5,
            self.extraversion or 0.5,
            self.agreeableness or 0.5,
            self.neuroticism or 0.5,
        ]


class AIExtractedInterests(SqlAlchemyBase):
    __tablename__ = "ai_extracted_interests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    hobbies = Column(JSON, nullable=True, default=list)
    topics = Column(JSON, nullable=True, default=list)
    skills = Column(JSON, nullable=True, default=list)
    dislikes = Column(JSON, nullable=True, default=list)
    occupation = Column(String(200), nullable=True)
    work_style = Column(Text, nullable=True)
    short_term_goals = Column(JSON, nullable=True, default=list)
    long_term_goals = Column(JSON, nullable=True, default=list)
    preferences = Column(JSON, nullable=True)
    last_extraction = Column(DateTime, default=datetime.utcnow)


class UserSchwartzProfile(SqlAlchemyBase):
    """10 базовых ценностей по теории Schwartz (шкала 0.0–1.0)."""

    __tablename__ = "user_schwartz_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    self_direction = Column(Float, default=0.5)
    stimulation = Column(Float, default=0.5)
    hedonism = Column(Float, default=0.5)
    achievement = Column(Float, default=0.5)
    power = Column(Float, default=0.5)
    security = Column(Float, default=0.5)
    conformity = Column(Float, default=0.5)
    tradition = Column(Float, default=0.5)
    benevolence = Column(Float, default=0.5)
    universalism = Column(Float, default=0.5)

    values_json = Column(JSON, nullable=True)
    confidence_score = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    SCHWARTZ_KEYS = (
        "self_direction", "stimulation", "hedonism", "achievement", "power",
        "security", "conformity", "tradition", "benevolence", "universalism",
    )

    def to_vector(self) -> list[float]:
        return [getattr(self, key) or 0.5 for key in self.SCHWARTZ_KEYS]

    def is_populated(self, min_confidence: float = 0.1) -> bool:
        return (self.confidence_score or 0.0) >= min_confidence


class UserCompatibility(SqlAlchemyBase):
    __tablename__ = "ai_user_compatibility"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id_1 = Column(Integer, ForeignKey("users.id"), nullable=False)
    user_id_2 = Column(Integer, ForeignKey("users.id"), nullable=False)

    overall_score = Column(Float, default=0.0)
    romantic_score = Column(Float, default=0.0)
    professional_score = Column(Float, default=0.0)
    creative_score = Column(Float, default=0.0)
    interest_overlap = Column(Float, default=0.0)
    recommendations = Column(JSON, nullable=True)
    calculated_at = Column(DateTime, default=datetime.utcnow)
