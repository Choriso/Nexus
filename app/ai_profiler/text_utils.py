"""
Утилиты для очистки пользовательского текста перед созданием embedding-векторов и ручными feature-вычислениями.

Обеспечивается единая логика предобработки для AIProfiler, preprocess.py и модульных тестов.
Не зависит от PyTorch.
"""

from __future__ import annotations

import re
from typing import Final

MAX_PROFILE_TEXT_CHARS: Final[int] = 200_000

def clean_user_text(text: str | None) -> str:
    """
    Очищает пользовательский текст для дальнейшего анализа:
    удаляет URL-адреса, нормализует пробелы и оставляет только полезные печатаемые символы.

    Args:
        text (str | None): Исходный текст, введённый пользователем.
    
    Returns:
        str: Безопасная строка для анализа. Если содержимое отсутствует, возвращает пустую строку.
    
    Примечания:
        - Не заменяет полноценную модерацию контента.
        - Основная цель — снизить поверхность возможных атак (например, длинные URL, управляющие символы)
          и убрать шум перед анализом.
        - Применяется ограничение длины строки для предотвращения атаки вида ReDoS и чрезмерного расхода памяти.
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text[:MAX_PROFILE_TEXT_CHARS]
    text = text.replace("\x00", "")
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9?!.,\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
