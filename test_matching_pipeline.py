import os
import sys
import logging

# Выставляем пути к проекту
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Включаем логирование, чтобы видеть отладочные сообщения Cursor
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineDiagnostic")

from data.session import global_init, create_session
from data.user import User
from data.ai import AIExtractedInterests
from app.ai_profiler.interest_graph import (
    resolve_tag_to_slug,
    build_query_weights,
    calculate_graph_interest_score,
    _extract_tags_from_profile
)
from app.ai.matching_engine import calculate_multidimensional_compatibility
from config import config


def run_diagnostic():
    print("\n" + "=" * 60)
    print("🚀 НАЧАЛО ГЛУБОКОЙ ДИАГНОСТИКИ КОНВЕЙЕРА МАТЧИНГА")
    print("=" * 60)

    # 1. Инициализация БД
    # 1. Инициализация БД
    print("\n[ШАГ 1] Подключение к базе данных...")
    # Вытаскиваем строку подключения из твоего централизованного конфига
    db_string = getattr(config, 'SQLALCHEMY_DATABASE_URI', None) or getattr(config, 'DATABASE_URL', None)
    if not db_string and hasattr(config, 'seed_db_config'):
        # Если это фабричный конфиг для сида, соберем строку вручную
        cfg = config.seed_db_config()
        db_string = f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"

    if not db_string:
        print("❌ ОШИБКА: Не удалось найти строку подключения к БД в config.py!")
        return

    print(f"  • Инициализируем БД со строкой: {db_string[:25]}***")
    global_init(db_string)  # ТЕПЕРЬ ПЕРЕДАЕМ СТРОКУ!
    db_sess = create_session()

    # 2. Проверяем тесты, которые вызывают "ужас"
    test_tags = ["игры", "разработка", "дизайн", "программирование", "python"]

    print("\n[ШАГ 2] Проверка резолвера тегов в слаги графа:")
    for tag in test_tags:
        slug = resolve_tag_to_slug(tag)
        print(f"  • Ввод: '{tag}' ---> Разрешился в slug: '{slug}'")
        if not slug:
            print(f"    ⚠️ КРИТИЧЕСКИ: Тег '{tag}' не смог привязаться к онтологии!")

    # 3. Ищем тестовых пользователей из сидера
    print("\n[ШАГ 3] Поиск тестовых пользователей в БД...")
    users = db_sess.query(User).all()
    if not users:
        print("❌ ОШИБКА: База данных пуста! Запусти сначала python tools/seed_db.py")
        return
    print(f"  ✅ Найдено пользователей в базе: {len(users)}")

    # Выберем двух пользователей для проверки
    me = users[0]
    others = users[1:]
    print(f"  • Тестируем матчинг от лица: ID {me.id} ({me.name})")

    # 4. Проверяем извлечение тегов из профилей
    print("\n[ШАГ 4] Проверка извлечения тегов из AI-профилей (AIExtractedInterests):")
    for u in users[:4]:
        ext = db_sess.query(AIExtractedInterests).filter(AIExtractedInterests.user_id == u.id).first()
        if not ext:
            print(f"  ⚠️ У пользователя ID {u.id} ({u.name}) ОТСУТСТВУЕТ запись в ai_extracted_interests!")
            continue

        tags = _extract_tags_from_profile(ext)
        print(
            f"  • ID {u.id} ({u.name}): колонки -> hobbies: {ext.hobbies}, topics: {ext.topics}, skills: {ext.skills}")
        print(f"    🎯 Извлечено тегов функцией: {list(tags)}")

    # 5. Имитируем реальный поисковый запрос по разным категориям
    print("\n[ШАГ 5] Имитация живых поисковых запросов сайта:")
    queries_to_test = [
        {"q": "игры", "expected_at_least_one": True},
        {"q": "разработка", "expected_at_least_one": True},
        {"q": "дизайн", "expected_at_least_one": True}
    ]

    # Подгружаем карту имен нод (как это делается в реальном эндпоинте)
    from data.interest_hierarchy import InterestHierarchyNode
    nodes = db_sess.query(InterestHierarchyNode).all()
    hierarchy_node_names = {node.id: node.name for node in nodes}

    for query in queries_to_test:
        search_term = query["q"]
        print(f"\n⚡ Тестируем ввод в поиск слова: '{search_term}'")

        query_tags = {search_term}
        query_weights = build_query_weights(db_sess, query_tags)
        print(f"  Построенная карта весов для запроса: {query_weights}")

        if not query_weights:
            print(f"  ❌ Стоп. Карта весов пуста. Алгоритм не понял тег '{search_term}'")
            continue

        matches_found = 0
        for candidate in others:
            ext_profile = db_sess.query(AIExtractedInterests).filter(
                AIExtractedInterests.user_id == candidate.id).first()

            # Запускаем расчет
            score, matched_tags = calculate_graph_interest_score(
                db=db_sess,
                query_tags=query_tags,
                other_user_id=candidate.id,
                other_extracted=ext_profile,
                query_weights=query_weights,
                hierarchy_node_names=hierarchy_node_names
            )

            if score > 0.05:
                matches_found += 1
                print(
                    f"  ✅ Кандидат ID {candidate.id} ({candidate.name}) -> Score: {score:.4f}, Совпали: {matched_tags}")

        print(f"  Итого найдено совпадений по графу для '{search_term}': {matches_found}")

    # 6. Эмуляция сквозного многомерного движка (matching_engine.py)
    print("\n[ШАГ 6] Проверка интеграции с многомерным движком (calculate_multidimensional_compatibility):")
    if len(users) >= 2:
        candidate = users[1]
        print(f"  Скрещиваем ID {me.id} и ID {candidate.id} через многомерный движок...")
        try:
            # Имитируем передачу данных, как в profile.py
            res = calculate_multidimensional_compatibility(
                db=db_sess,
                current_user_id=me.id,
                other_user_id=candidate.id,
                query_tags={"игры", "разработка"},
                other_extracted=db_sess.query(AIExtractedInterests).filter(
                    AIExtractedInterests.user_id == candidate.id).first(),
                ocean_score_normalized = 0.5,  # Средние значения по Big Five (нейтрально)
                my_schwartz = None,  # Нулевые векторы для ценностей Шварца
                other_schwartz = None,
                my_behavior = None,  # Пустые словари для поведенческого паттерна
                other_behavior = None
            )
            print(f"  🎉 Результат движка: {res}")
        except Exception as e:
            print(f"  ❌ КРАШ ДВИЖКА: {str(e)}")
            import traceback
            traceback.print_exc()

    db_sess.close()
    print("\n" + "=" * 60)
    print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА!")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostic()