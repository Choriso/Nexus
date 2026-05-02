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
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer


def get_manual_features(text: str) -> list:
    if not text:
        return [0.0, 0.0, 0.0, 0.0]
    length = min(len(text) / 1000, 1.0)
    letters = [c for c in text if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper()) / (len(letters) + 1) if letters else 0.0
    excl = min(text.count('!') / 5, 1.0)
    ques = min(text.count('?') / 5, 1.0)
    return [length, caps, excl, ques]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9?!.,\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


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

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

    texts = [clean_text(item["text"]).lower() for item in raw_data]

    print("Вычисляю эмбеддинги...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )  # (N, 384)

    result = []
    for i, item in enumerate(tqdm(raw_data, desc="Собираю признаки")):
        manual = get_manual_features(item["text"])
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
