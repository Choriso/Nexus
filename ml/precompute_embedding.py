import torch
import json
from tqdm import tqdm
from app.ai_profiler.core import AIProfiler


def precompute():
    # Инициализируем профилировщик (он сам подгрузит BERT на FAIDAI)
    profiler = AIProfiler()

    with open("data/train_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    precomputed = []
    print(f"Кеширование 11579 записей...")

    for item in tqdm(data):
        text = item["text"]

        # 1. Получаем BERT эмбеддинг (384)
        embedding = profiler.bert_model.encode(text, convert_to_tensor=True)

        # 2. Получаем ручные фичи (4)
        # В твоем core.py это обычно делает метод профилировщика
        # Если метода get_manual_features нет, посчитаем их прямо тут:
        manual = [
            len(text) / 500.0,
            text.count('!') / 10.0,
            text.count('?') / 10.0,
            sum(1 for c in text if c.isupper()) / (len(text) + 1)
        ]
        manual_tensor = torch.tensor(manual, device=embedding.device)

        # 3. Объединяем в итоговый вектор 388
        full_features = torch.cat([embedding, manual_tensor])

        precomputed.append({
            "features": full_features.detach().cpu().numpy().tolist(),
            "target": item["scores"]
        })

    torch.save(precomputed, "data/train_data_precomputed.pt")
    print("\n✅ Готово! Файл создан: data/train_data_precomputed.pt")


if __name__ == "__main__":
    precompute()