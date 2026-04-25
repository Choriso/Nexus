"""
prepare_dataset.py — Конвертация CSV → train_data.json с Label Smoothing
и автоматическим смешиванием с существующим датасетом.

Использование:
    python prepare_dataset.py --csv data/new_data.csv --output data/train_data.json
    python prepare_dataset.py --csv data/new_data.csv --output data/train_data.json --existing data/train_data.json
"""

import argparse
import csv
import json
import random
from pathlib import Path


SMOOTH_LOW  = 0.1   # 0 → 0.1
SMOOTH_HIGH = 0.9   # 1 → 0.9

SCORE_COLUMNS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def smooth(value: float) -> float:
    """Применяет Label Smoothing: 0 → 0.1, 1 → 0.9."""
    raw = float(value)
    if raw <= 0.0:
        return SMOOTH_LOW
    if raw >= 1.0:
        return SMOOTH_HIGH
    # Если значение уже непрерывное (не бинарное) — сглаживаем пропорционально
    return SMOOTH_LOW + raw * (SMOOTH_HIGH - SMOOTH_LOW)


def load_csv(csv_path: Path) -> list[dict]:
    """Читает CSV и возвращает список записей для train_data.json."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [col for col in ["cv_text"] + SCORE_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"В CSV отсутствуют колонки: {missing}")

        for i, row in enumerate(reader, start=1):
            text = row["cv_text"].strip()
            if not text:
                print(f"  [warn] строка {i}: пустой текст, пропущена")
                continue

            try:
                scores = [smooth(row[col]) for col in SCORE_COLUMNS]
            except ValueError as e:
                print(f"  [warn] строка {i}: ошибка преобразования ({e}), пропущена")
                continue

            records.append({"text": text, "scores": scores})

    return records


def load_existing(json_path: Path) -> list[dict]:
    """Загружает существующий train_data.json если он есть."""
    if not json_path.exists():
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Загружено {len(data)} записей из существующего датасета.")
    return data


def merge_and_shuffle(existing: list[dict], new_records: list[dict]) -> list[dict]:
    """Объединяет старые и новые записи, перемешивает."""
    combined = existing + new_records
    random.shuffle(combined)
    return combined


def save_json(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Конвертация CSV → train_data.json")
    parser.add_argument("--csv",      required=True,  help="Путь к входному CSV-файлу")
    parser.add_argument("--output",   required=True,  help="Путь к выходному JSON-файлу")
    parser.add_argument("--existing", default=None,   help="Существующий train_data.json для смешивания")
    parser.add_argument("--seed",     type=int, default=42, help="Random seed для воспроизводимости")
    args = parser.parse_args()

    random.seed(args.seed)

    csv_path    = Path(args.csv)
    output_path = Path(args.output)
    existing_path = Path(args.existing) if args.existing else output_path

    print(f"[1/4] Чтение CSV: {csv_path}")
    new_records = load_csv(csv_path)
    print(f"      Загружено {len(new_records)} новых записей.")

    print(f"[2/4] Загрузка существующего датасета: {existing_path}")
    existing_records = load_existing(existing_path)

    print(f"[3/4] Смешивание и перемешивание данных...")
    combined = merge_and_shuffle(existing_records, new_records)
    print(f"      Итого записей: {len(combined)} "
          f"(было: {len(existing_records)}, добавлено: {len(new_records)})")

    print(f"[4/4] Сохранение → {output_path}")
    save_json(combined, output_path)

    # Краткая статистика по Label Smoothing
    import numpy as np
    all_scores = np.array([r["scores"] for r in new_records])
    trait_names = ["O", "C", "E", "A", "N"]
    print("\n--- Статистика новых данных (после Label Smoothing) ---")
    for i, name in enumerate(trait_names):
        col = all_scores[:, i]
        print(f"  {name}: mean={col.mean():.3f}  min={col.min():.3f}  max={col.max():.3f}")

    print("\n✅ Готово.")


if __name__ == "__main__":
    main()