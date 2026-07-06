"""
Поведенческий профиль: агрегация метрик сообщений и расчёт совместимости.
"""

from __future__ import annotations

import logging
import re

from sqlalchemy import func
from sqlalchemy.orm import Session

from data.behavior import UserBehaviorProfile
from data.message import Message

logger = logging.getLogger(__name__)

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def count_emojis(text: str | None) -> int:
    if not text:
        return 0
    return len(_EMOJI_RE.findall(text))


def compute_message_metadata(
    content: str | None,
    *,
    prev_timestamp=None,
    now=None,
) -> tuple[int, float | None, int]:
    """char_count, reply_time (sec), emoji_count."""
    from datetime import datetime

    text = content or ""
    char_count = len(text)
    emoji_count = count_emojis(text)
    reply_time = None
    if prev_timestamp and now:
        reply_time = max(0.0, (now - prev_timestamp).total_seconds())
    elif prev_timestamp:
        reply_time = max(0.0, (datetime.utcnow() - prev_timestamp).total_seconds())
    return char_count, reply_time, emoji_count


def refresh_user_behavior_profile(db: Session, user_id: int) -> UserBehaviorProfile | None:
    """Пересчитывает агрегаты по сообщениям пользователя."""
    stats = (
        db.query(
            func.avg(Message.char_count),
            func.avg(Message.reply_time),
            func.avg(Message.emoji_count),
            func.avg(func.extract("hour", Message.timestamp)),
            func.count(Message.id),
        )
        .filter(
            Message.author_id == user_id,
            Message.content.isnot(None),
            Message.message_type == "text",
        )
        .one()
    )

    avg_char, avg_reply, avg_emoji, avg_hour, msg_count = stats
    if not msg_count:
        return None

    profile = db.query(UserBehaviorProfile).filter_by(user_id=user_id).first()
    if not profile:
        profile = UserBehaviorProfile(user_id=user_id)
        db.add(profile)

    profile.avg_char_count = float(avg_char or 0.0)
    profile.avg_reply_time = float(avg_reply or 0.0)
    profile.avg_emoji_count = float(avg_emoji or 0.0)
    profile.avg_hour = float(avg_hour if avg_hour is not None else 12.0)
    profile.message_count = int(msg_count)
    db.commit()
    return profile


def _behavior_vector(profile: UserBehaviorProfile | None) -> list[float] | None:
    if profile is None or (profile.message_count or 0) < 3:
        return None
    return [
        min(profile.avg_char_count / 500.0, 1.0),
        min(profile.avg_reply_time / 3600.0, 1.0),
        min(profile.avg_emoji_count / 10.0, 1.0),
        profile.avg_hour / 23.0,
    ]


def calculate_behavioral_score(
    profile_a: UserBehaviorProfile | None,
    profile_b: UserBehaviorProfile | None,
    *,
    p: int = 2,
) -> float | None:
    """
    Minkowski-расстояние (p=2 → Euclidean) между нормализованными векторами.
    Возвращает score [0, 1] или None при cold-start.
    """
    vec_a = _behavior_vector(profile_a)
    vec_b = _behavior_vector(profile_b)
    if vec_a is None or vec_b is None:
        return None

    distance = sum(abs(a - b) ** p for a, b in zip(vec_a, vec_b)) ** (1.0 / p)
    max_distance = len(vec_a) ** (1.0 / p)
    return max(0.0, 1.0 - distance / max_distance)
