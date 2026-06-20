"""
preprocess.py — Предобработка сырого JSON (text + scores) в JSON с features.

Использование:
    python preprocess.py \
        --input data/generated_data_ocean.json \
        --output data/train_data_precomputed.json

Описание:
    Скрипт предназначен для создания обучающего файла с эмбеддингами и ручными признаками.
    Для каждой записи из исходного JSON вычисляется SBERT-эмбеддинг (384-мерный) и 4 ручных признака,
    после чего формируется итоговый feature-вектор длины 388. Получившийся словарь сохраняется для дальнейшего использования в PersonalityDataset.
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.ai_profiler.text_utils import clean_user_text


def get_manual_features(text: str) -> list[float]:
    """
    Вычисляет четыре ручных числовых признака по тексту.

    Признаки:
        1. Относительная длина текста (макс. 1.0)
        2. Доля заглавных букв среди буквенных символов
        3. Количество восклицательных знаков (не более 1.0)
        4. Количество вопросительных знаков (не более 1.0)

    Args:
        text (str): Исходный текст для обработки.

    Returns:
        list[float]: Список из четырёх признаков [длина, CAPS, !, ?].
    """
    if not text:
        return [0.0, 0.0, 0.0, 0.0]
    length = min(len(text) / 1000, 1.0)
    letters = [c for c in text if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper()) / (len(letters) + 1) if letters else 0.0
    excl = min(text.count("!") / 5, 1.0)
    ques = min(text.count("?") / 5, 1.0)
    return [length, caps, excl, ques]


def main() -> None:
    """
    Основная функция для запуска предобработки данных и создания файла с фичами.

    Аргументы командной строки:
        --input (str): Путь к исходному JSON с ключами "text" и "scores".
        --output (str): Путь, по которому будет сохранён результат.
        --batch_size (int, optional): Размер batch для получения эмбеддингов, по умолчанию 64.

    Returns:
        None: Результат сохраняется на диск в формате JSON.
    """
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

    embedding_name = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    model = SentenceTransformer(embedding_name, device=device)

    texts = [clean_user_text(item["text"]).lower() for item in raw_data]

    print("Вычисляю эмбеддинги...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    result = []
    for i, item in enumerate(tqdm(raw_data, desc="Собираю признаки")):
        cleaned = clean_user_text(item["text"])
        manual = get_manual_features(cleaned)
        features = np.concatenate([embeddings[i], manual]).tolist()
        result.append({
            "features": features,
            "target": item["scores"],
            "weight": item.get("weight", 1.0),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"Сохранено в {args.output}")
    print(f"Пример записи: features={len(result[0]['features'])}d, target={result[0]['target']}")


if __name__ == "__main__":
    main()
