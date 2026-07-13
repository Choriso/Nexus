import time
from pprint import pprint

# Импортируем реальный профилировщик из твоего проекта
from app.ai_profiler import get_profiler


def run_tests():
    print("⏳ Инициализация AIProfiler (загрузка моделей SBERT, это займет пару секунд)...")
    start_time = time.time()
    profiler = get_profiler()
    print(f"✅ Profiler загружен за {time.time() - start_time:.2f} сек.\n")

    print("=" * 60)
    print("🚀 ЗАПУСК ТЕСТОВ ГИБРИДНОГО МЭТЧИНГА")
    print("=" * 60)

    # Базовый вектор текущего пользователя (например, интроверт, логик)
    my_ocean = [0.8, 0.9, 0.2, 0.5, 0.3]

    # ---------------------------------------------------------
    # СЦЕНАРИЙ 1: Идеальный разработчик (Полное совпадение тегов)
    # ---------------------------------------------------------
    print("\n[СЦЕНАРИЙ 1] Идеальный разработчик")
    print("ОЖИДАНИЕ: Очень высокий скор (>80), тег 'разработка' должен дать буст.")

    dev_extracted = {
        "extraction_method": "neural",
        "skills": [{"subcategory": "разработка", "score": 0.88}],
        "hobbies": []
    }

    result_1 = profiler.calculate_hybrid_matching_score(
        search_query="написание кода на python",
        other_user_extracted_interests=dev_extracted,
        other_user_raw_text="Я бэкенд разработчик, пишу API",
        ocean_compatibility=85.0,  # Хорошая совместимость
        current_user_ocean=my_ocean,
        other_user_ocean=[0.7, 0.8, 0.3, 0.6, 0.2]
    )
    pprint(result_1)

    # ---------------------------------------------------------
    # СЦЕНАРИЙ 2: Геймер с высоким OCEAN (Проверка старого бага)
    # ---------------------------------------------------------
    print("\n[СЦЕНАРИЙ 2] Геймер с идеальным характером")
    print("ОЖИДАНИЕ: Низкий скор (<50). OCEAN = 100% НЕ должен перебить отсутствие интереса к коду.")

    gamer_extracted = {
        "extraction_method": "zero_shot",
        "skills": [],
        "hobbies": [{"subcategory": "видеоигры", "score": 0.95}]
    }

    result_2 = profiler.calculate_hybrid_matching_score(
        search_query="написание кода на python",
        other_user_extracted_interests=gamer_extracted,
        other_user_raw_text="Играю в CS2 на серверах каждый вечер",
        ocean_compatibility=100.0,  # Идеально подходят характерами
        current_user_ocean=my_ocean,
        other_user_ocean=[0.8, 0.9, 0.2, 0.5, 0.3]
    )
    pprint(result_2)

    # ---------------------------------------------------------
    # СЦЕНАРИЙ 3: Fallback (Нет тегов, но текст совпадает)
    # ---------------------------------------------------------
    print("\n[СЦЕНАРИЙ 3] Fallback (Новый пользователь, нет тегов от Celery)")
    print("ОЖИДАНИЕ: Средний скор. Система должна отработать только по SBERT и OCEAN.")

    result_3 = profiler.calculate_hybrid_matching_score(
        search_query="написание кода на python",
        other_user_extracted_interests=None,  # БД еще не обновилась
        other_user_raw_text="Обожаю программирование и алгоритмы",
        ocean_compatibility=70.0,
        current_user_ocean=my_ocean,
        other_user_ocean=[0.5, 0.5, 0.5, 0.5, 0.5]
    )
    pprint(result_3)

    # ---------------------------------------------------------
    # СЦЕНАРИЙ 4: Complementary OCEAN (Дополнение в работе)
    # ---------------------------------------------------------
    print("\n[СЦЕНАРИЙ 4] Разработчик-экстраверт (Проверка комплементарности)")
    print("ОЖИДАНИЕ: Выше скор за счет разницы в экстраверсии (полезно для командной работы).")

    # Тот же разработчик из Сценария 1, но сильный экстраверт
    extrovert_ocean = [0.7, 0.8, 0.9, 0.6, 0.2]  # extraversion = 0.9 (разница с my_ocean > 0.4)

    result_4 = profiler.calculate_hybrid_matching_score(
        search_query="написание кода на python",
        other_user_extracted_interests=dev_extracted,
        other_user_raw_text="Я бэкенд разработчик, люблю выступать на митапах",
        ocean_compatibility=85.0,
        current_user_ocean=my_ocean,
        other_user_ocean=extrovert_ocean
    )
    pprint(result_4)


if __name__ == "__main__":
    # Выполнять только если скрипт запущен напрямую
    run_tests()