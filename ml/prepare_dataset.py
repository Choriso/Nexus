import argparse
import csv
import json
import random
from pathlib import Path

SCORE_COLUMNS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

def smooth_scores(scores, epsilon=0.2):
    """Сжимает [0,1] → [epsilon, 1-epsilon]"""
    return [s * (1 - 2*epsilon) + epsilon for s in scores]

def load_csv(csv_path: Path, weight: float = 1.0) -> list[dict]:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # У Kaggle данных обычно значения уже 0-1, просто берем их
            scores = [float(row[col]) for col in SCORE_COLUMNS]
            scores = smooth_scores(scores)
            records.append({
                "text": row["cv_text"].strip(),
                "scores": scores,
                "weight": weight # Вес для синтетики (обычно 1.0)
            })
    return records

def load_quality_json(json_path: Path, weight: float = 3.0) -> list[dict]:
    if not json_path.exists(): return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Помечаем твои качественные данные высоким весом
    for item in data:
        item["weight"] = weight
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Путь к Kaggle CSV")
    parser.add_argument("--json", required=True, help="Путь к твоему качественному JSON")
    parser.add_argument("--output", default="data/train_data.json")
    args = parser.parse_args()

    # 1. Загружаем синтетику с Kaggle (вес 1.0)
    print(f"Загрузка Kaggle CSV...")
    kaggle_records = load_csv(Path(args.csv), weight=1.0)

    # 2. Загружаем твои данные (вес 3.0)
    print(f"Загрузка качественного JSON...")
    quality_records = load_quality_json(Path(args.json), weight=10.0)

    # 3. Смешиваем
    combined = kaggle_records + quality_records
    random.shuffle(combined)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"✅ Готово! Итого {len(combined)} записей. Твои данные теперь весят в 3 раза больше.")

if __name__ == "__main__":
    main()