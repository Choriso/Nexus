"""
Sanitização de texto de utilizador para embeddings e features manuais.

Mantém a lógica partilhada entre ``AIProfiler``, ``preprocess.py`` e testes,
sem dependências de PyTorch.
"""

from __future__ import annotations

import re
from typing import Final

# Limite para mitigar ReDoS / custo de regex e memória em inputs maliciosos.
MAX_PROFILE_TEXT_CHARS: Final[int] = 200_000


def clean_user_text(text: str | None) -> str:
    """Remove URLs, normaliza espaços e filtra caracteres não imprimíveis úteis.

    Args:
        text: Texto bruto do utilizador ou None.

    Returns:
        String segura para análise; string vazia se não houver conteúdo útil.

    Note:
        Não substitui moderação de conteúdo; apenas reduz superfície de ataque
        e ruído (URLs longas, caracteres de controlo).
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Truncar antes de regex pesadas
    text = text[:MAX_PROFILE_TEXT_CHARS]
    text = text.replace("\x00", "")
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9?!.,\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
