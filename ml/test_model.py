import torch
import numpy as np
from app.ai_profiler.core import AIProfiler
from pathlib import Path

ARTIFACTS_DIR = Path("ml/artifacts")


def test_inference() -> None:
    """
    Тестирует работу профайлера AIProfiler на различных текстах.

    Выводит результаты профилирования: значения шкал OCEAN, предположительный MBTI-тип, 
    стиль коммуникации и уровень уверенности для каждого тестового текста.

    Returns:
        None: Результаты выводятся в консоль.
    """
    profiler = AIProfiler()

    test_texts = [
        "Я обожаю планировать всё заранее и следовать списку задач.",
        "Меня всё бесит, этот мир несправедлив, я в ярости!",
        "РЕБЯТА! Я нашел способ как не спать трое суток и моделить без остановки! Метод проверен, я почти вижу звуки!!!",
        "Для оптимизации рендера в Blender я использую циклы и запекание текстур, а затем импортирую меши в движок через Python скрипт.",
        "Иногда мне кажется, что я совсем один в этом огромном мире. Всё валится из рук, и мне очень жаль, если я кого-то подвел своими ошибками.",
        "ЭЙ ВЫ ВСЕ!!!!!! СМОТРИТЕ НА МЕНЯ!!!!!!!! Я САМЫЙ КРУТОЙ И ЭНЕРГИЧНЫЙ ЧЕЛОВЕК В ЭТОМ ЧАТЕ!!!!!!!!"
    ]

    print("\n--- ТЕСТ NEXUS PROFILER ---")

    for text in test_texts:
        result: dict = profiler.analyze_profile(text)

        if not result:
            print(f"Ошибка анализа текста: {text[:30]}...")
            continue

        ocean: list = result["ocean"]
        mbti: str = result["mbti_type"]
        style: str = result["communication"]
        conf: float = result["confidence_score"]

        print(f"Текст: '{text}'")
        print(f"OCEAN: O={ocean[0]:.2f}, C={ocean[1]:.2f}, E={ocean[2]:.2f}, A={ocean[3]:.2f}, N={ocean[4]:.2f}")
        print(f"MBTI: {mbti} (Уверенность: {conf:.2%})")
        print(f"Стиль: {style}")
        print("-" * 50)


if __name__ == "__main__":
    test_inference()