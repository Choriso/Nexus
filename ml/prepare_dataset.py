import argparse
import csv
import json
import random
from pathlib import Path

SCORE_COLUMNS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

def smooth_scores(scores: list[float], epsilon: float = 0.1) -> list[float]:
    """
    Сжимает значения факторов из диапазона [0,1] в [epsilon, 1-epsilon].

    Args:
        scores (list[float]): Список значений факторов в диапазоне [0, 1].
        epsilon (float, optional): Коэффициент сжатия. По умолчанию 0.1.

    Returns:
        list[float]: Список скорректированных значений факторов.
    """
    return [s * (1 - 2*epsilon) + epsilon for s in scores]

def load_csv(csv_path: Path, weight: float = 1.0) -> list[dict]:
    """
    Загружает данные из CSV-файла и создает записи с текстом, факторами и весом.

    Args:
        csv_path (Path): Путь к CSV-файлу.
        weight (float, optional): Вес для каждой записи. По умолчанию 1.0.

    Returns:
        list[dict]: Список словарей с ключами 'text', 'scores', 'weight'.
    """
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores = [float(row[col]) for col in SCORE_COLUMNS]
            scores = smooth_scores(scores)
            records.append({
                "text": row["cv_text"].strip(),
                "scores": scores,
                "weight": weight
            })
    return records

def load_quality_json(json_path: Path, weight: float = 3.0) -> list[dict]:
    """
    Загружает качественные данные из JSON и проставляет им заданный вес.

    Args:
        json_path (Path): Путь к JSON-файлу.
        weight (float, optional): Вес для каждой записи. По умолчанию 3.0.

    Returns:
        list[dict]: Список словарей с данными и ключом 'weight'.
    """
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["weight"] = weight
    return data

def main() -> None:
    """
    Объединяет и перемешивает синтетические и качественные данные, 
    сохраняет итоговую выборку в JSON-файл.

    Аргументы командной строки:
        --csv (str): Путь к файлу Kaggle CSV.
        --json (str): Путь к качественному JSON-файлу.
        --output (str, optional): Путь к выходному JSON-файлу (по умолчанию data/train_data.json).

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Путь к Kaggle CSV")
    parser.add_argument("--json", required=True, help="Путь к твоему качественному JSON")
    parser.add_argument("--output", default="data/train_data.json")
    args = parser.parse_args()

    print(f"Загрузка Kaggle CSV...")
    kaggle_records = load_csv(Path(args.csv), weight=1.0)

    print(f"Загрузка качественного JSON...")
    quality_records = load_quality_json(Path(args.json), weight=10.0)

    combined = kaggle_records + quality_records
    random.shuffle(combined)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Итого {len(combined)} записей сохранены в {args.output}")

if __name__ == "__main__":
    main()