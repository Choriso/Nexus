import torch
import numpy as np
from app.ai_profiler.core import AIProfiler
from pathlib import Path

# Путь к артефактам (проверь, что путь совпадает с реальностью)
ARTIFACTS_DIR = Path("ml/artifacts")


def test_inference():
    # 1. Инициализируем профайлер.
    # Внутри __init__ он сам создаст PersonalityClassifier(input_size=388)
    # и загрузит веса из personality_model.pth
    profiler = AIProfiler()

    test_texts = [
        "Я обожаю планировать всё заранее и следовать списку задач.",
        "Меня всё бесит, этот мир несправедлив, я в ярости!",
        "РЕБЯТА! Я нашел способ как не спать трое суток и моделить без остановки! Метод проверен, я почти вижу звуки!!!",
        "МЕНЯ ВСЕ БЕСИТ ЭТОТ МИР НЕСПРАВЕДЛИВ Я В ЯРОСТИ!!!!!!!"
    ]

    print("\n--- ТЕСТ ГИБРИДНОЙ МОДЕЛИ (388 параметров + Boosting) ---")

    for text in test_texts:
        # Используем метод analyze_text, который:
        # - Считает BERT эмбеддинг
        # - Считает 4 ручные фичи (капс, знаки и т.д.)
        # - Склеивает их в 388 параметров
        # - Применяет Keyword Boosting
        scores = profiler.analyze_text(text)

        print(f"Текст: '{text}'")
        # Индексы: 0-O, 1-C, 2-E, 3-A, 4-N
        print(f"OCEAN: O={scores[0]:.2f}, C={scores[1]:.2f}, E={scores[2]:.2f}, A={scores[3]:.2f}, N={scores[4]:.2f}\n")


if __name__ == "__main__":
    test_inference()
