"""
preprocess.py — Предобработка сырого JSON (text + scores) в JSON с features.

Использование:
    python preprocess.py \
        --input data/generated_data_ocean.json \
        --output data/train_data_precomputed.json

Требования:
    pip install sentence-transformers torch tqdm

Что делает скрипт:
    1. Загружает JSON с полями "text" и "scores"
    2. Вычисляет SBERT-эмбеддинги (384-мерные)
    3. Вычисляет ручные признаки (4-мерные): длина, заглавные, ! и ?
    4. Конкатенирует → 388-мерный вектор в поле "features"
    5. Копирует "scores" → "target"
    6. Сохраняет обогащённый JSON (пригоден для PersonalityDataset)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Garantir raiz do projeto no path ao correr ``python ml/preprocess.py``
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.ai_profiler.text_utils import clean_user_text


def get_manual_features(text: str) -> list[float]:
    """Extrai quatro escalares alinhados ao :class:`AIProfiler`."""
    if not text:
        return [0.0, 0.0, 0.0, 0.0]
    length = min(len(text) / 1000, 1.0)
    letters = [c for c in text if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper()) / (len(letters) + 1) if letters else 0.0
    excl = min(text.count("!") / 5, 1.0)
    ques = min(text.count("?") / 5, 1.0)
    return [length, caps, excl, ques]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Путь к сырому JSON")
    parser.add_argument("--output", required=True, help="Куда сохранить обогащённый JSON")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    with open(args.input, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"Загружено {len(raw_data)} записей")

    embedding_name = os.environ.get(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    model = SentenceTransformer(embedding_name, device=device)

    texts = [clean_user_text(item["text"]).lower() for item in raw_data]

    print("Вычисляю эмбеддинги...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )  # (N, 384)

    result = []
    for i, item in enumerate(tqdm(raw_data, desc="Собираю признаки")):
        cleaned = clean_user_text(item["text"])
        manual = get_manual_features(cleaned)
        features = np.concatenate([embeddings[i], manual]).tolist()  # (388,)
        result.append({
            "features": features,
            "target": item["scores"],
            "weight": item.get("weight", 1.0),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"Готово! Сохранено в {args.output}")
    print(f"Пример записи: features={len(result[0]['features'])}d, target={result[0]['target']}")


if __name__ == "__main__":
    main()
