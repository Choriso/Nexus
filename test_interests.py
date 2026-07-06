import os
import re
from collections import defaultdict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# 1. Симулируем taxonomy.py
INTEREST_TAXONOMY = {
    "hobby": {
        "спорт и фитнес": ["бег", "спортзал", "тренировка", "йога", "фитнес", "пробежка"],
        "видеоигры": ["игры", "steam", "cs2", "гейминг", "консоль", "дота"],
        "музыка": ["музыка", "плейлист", "концерт", "инструмент", "гитара"],
        "путешествия": ["путешествия", "поездка", "travel", "отпуск"]
    },
    "work": {
        "разработка": ["python", "backend", "frontend", "devops", "sql", "писать код"],
        "дизайн": ["3d моделирование", "blender", "figma", "интерфейсы", "дизайн"],
        "менеджмент": ["лидерство", "управление командой", "план", "дедлайн"]
    },
    "psychology": {
        "саморазвитие": ["медитация", "осознанность", "книги", "продуктивность"],
        "отношения": ["общение", "эмпатия", "поддержка", "психология отношений"]
    }
}

print("Загрузка модели SentenceTransformer...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# 2. Симулируем ZeroShotInterestExtractor
class TestZeroShotExtractor:
    def __init__(self, model, taxonomy, threshold=0.45, max_per_group=2):
        self.model = model
        self.threshold = threshold
        self.max_per_group = max_per_group
        # Теперь мы сохраняем эмбеддинги КАЖДОГО слова-якоря отдельно!
        self.category_anchors = self._precompute(taxonomy)

    def _precompute(self, taxonomy):
        flat = {}
        for group, categories in taxonomy.items():
            for label, anchors in categories.items():
                # Эмбеддинг для каждого слова в подкатегории отдельно
                embs = self.model.encode(anchors, convert_to_tensor=True)
                flat[(group, label)] = embs
        return flat

    def extract(self, text: str):
        # Делим на чанки
        chunks = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 5]
        if not chunks:
            return {}

        chunk_embs = self.model.encode(chunks, convert_to_tensor=True)
        results = defaultdict(list)

        for (group, label), anchor_embs in self.category_anchors.items():
            # Считаем матрицу схожести: каждый чанк со всеми якорями подкатегории
            sims = util.cos_sim(chunk_embs, anchor_embs)
            # Берем абсолютный максимум сходства среди всех чанков и всех якорей этой группы
            max_sim = float(sims.max())

            if max_sim >= self.threshold:
                results[group].append((label, round(max_sim, 3)))

        # Сортировка и обрезка
        final_results = {}
        for group, items in results.items():
            items.sort(key=lambda x: x[1], reverse=True)
            final_results[group] = items[:self.max_per_group]

        return final_results

# Инициализируем наш тестовый экстрактор
extractor = TestZeroShotExtractor(bert_model, INTEREST_TAXONOMY, threshold=0.40)

# 3. Набор тестовых сценариев
test_cases = [
    ("Обожаю писать код на python. Вчера весь вечер дебажил бэкенд.", "Идеальный случай (Разработка)"),
    ("В выходные бегал в парке, устроил жесткую пробежку. А вечером засел в доту с друзьями.",
     "Смешанный случай (Спорт + Игры)"),
    ("Я изучаю психологию отношений и читаю книги по продуктивности.", "Психология и Саморазвитие"),
    ("кулинария рецепты выпечка торты", "Кейс вне таксономии (Шум)"),
    ("обожаю писать код на python и делать 3d моделирование в blender",
     "Edge-case Клода: Две темы в одном предложении без точек")
]

print("\n--- ЗАПУСК ТЕСТОВ ---")
for text, desc in test_cases:
    print(f"\n[Тест]: {desc}")
    print(f"Текст: \"{text}\"")
    res = extractor.extract(text)
    if not res:
        print("  => Результат: ПУСТО (Сработает Fallback на Tier 0!)")
    else:
        for group, items in res.items():
            print(f"  Группа [{group}]:")
            for label, score in items:
                print(f"    - {label} (confidence: {score})")