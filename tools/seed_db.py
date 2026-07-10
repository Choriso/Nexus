"""Seed базы данных PostgreSQL для разработки с учетом новой архитектуры ИИ."""
import os
import sys

# Определяем путь к папке, в которой лежит текущий скрипт (tools), и берем её родительскую директорию (Nexus)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import random
import sys
import psycopg2
from faker import Faker

from config import config
from data.session import global_init, create_session
from data.user import User

# Импортируем ТОЛЬКО после настройки путей
from app.ai.personality_analyzer import analyze_user_profile
from app.ai_profiler.interest_graph import register_user_tags, ensure_hierarchy_seeded

DB_CONFIG = config.seed_db_config()
fake = Faker(["ru_RU"])

ARCHETYPES = {
    "Профи": {
        "msgs": [
            "Нам нужно декомпозировать эти сложные задачи на спринты и выставить эстимейты.",
            "Кодстайл сильно хромает, нужно обязательно поправить отступы и импорты.",
            "Я внимательно изучил техническую документацию, там явно есть баг в деплой API.",
            "Необходимо провести качественный рефакторинг этого микросервиса на бэкенде."
        ],
        "cat": "work",
        # ЯВНЫЕ теги для графа интересов
        "graph_tags": ["разработка", "python", "backend", "архитектура"]
    },
    "Душа компании": {
        "msgs": [
            "Привет! Как ваши дела? Погнали завтра все вместе на крутой IT-митап!",
            "Обожаю пробовать новые фреймворки и библиотеки, это всегда так весело и интересно!",
            "Давайте соберемся всей нашей дружной командой в пятницу вечером в баре!",
            "Вау, это просто потрясающая новость! Вы все такие классные, обожаю наш чат!"
        ],
        "cat": "psychology",
        "graph_tags": ["общение", "психология", "soft skills", "лидерство"]
    },
    "Творец": {
        "msgs": [
            "Я тут придумал совершенно безумную, но очень красивую идею для нового UI/UX дизайна...",
            "А что если нам полностью заменить текущий бэкенд и переписать критические узлы на Rust?",
            "Вдохновение пришло сегодня глубокой ночью, сразу набросал свежий концепт в Figma.",
            "Искусство должно вдохновлять, поэтому я добавил несколько неоновых элементов в интерфейс."
        ],
        "cat": "hobby",
        "graph_tags": ["дизайн", "figma", "творчество", "искусство", "3d-моделирование"]
    },
    "Скейтер/Геймер": {
        "msgs": [
            "Го катку в CS2 прямо сейчас? Я как раз создал отличный приватный сервер для своих.",
            "Вчера выбил невероятно редкий и дорогой скин из нового кейса, зацените скриншот!",
            "Кто-то сталкивался с такой проблемой, как полностью убрать внезапные лаги в Steam?",
            "Мы вчера с пацанами катали три на два, было очень потно, но мы затащили этот раунд!"
        ],
        "cat": "hobby",
        "graph_tags": ["игры", "cs2", "гейминг", "киберспорт"]
    },
}

TOPICS = {
    "work": ["Python", "Flask", "PostgreSQL", "Архитектура", "Deployment"],
    "hobby": ["3D Modeling", "Blender", "CS2", "Unity", "Digital Art"],
    "psychology": ["Эмпатия", "Тайм-менеджмент", "Лидерство", "Медитация", "Soft Skills"],
}


def run_mega_seed(num_users: int = 10) -> None:
    if not DB_CONFIG["password"]:
        print("❌ Error: defina SEED_DB_PASSWORD...", file=sys.stderr)
        sys.exit(1)

    global_init(config.DATABASE_URL)

    # 🔥 Создаём иерархию ОДИН раз до всех регистраций
    session = create_session()
    try:
        ensure_hierarchy_seeded(session)
        session.commit()
        print("✅ Иерархия графа интересов создана/проверена")
    finally:
        session.close()

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        print("--- 🚀 Запуск генерации цифровых личностей... ---")

        # Создаём дефолтного пользователя ID=1
        session = create_session()
        my_user = session.query(User).filter(User.id == 1).first()
        if not my_user:
            cur.execute(
                "INSERT INTO users (id, name, email, hashed_password) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (1, "Данила Фещенко", "danila@nexus.com", "scrypt:32768:8:1$fake_hash")
            )
        session.close()

        for i in range(num_users):
            arch_name = random.choice(list(ARCHETYPES.keys()))
            arch_data = ARCHETYPES[arch_name]

            full_name = f"{fake.name()} ({arch_name})"
            email = fake.unique.email()
            pwd = "scrypt:32768:8:1$fake_hash_value"

            cur.execute(
                "INSERT INTO users (name, email, hashed_password) VALUES (%s, %s, %s) RETURNING id",
                (full_name, email, pwd),
            )
            u_id = cur.fetchone()[0]
            print(f"\n👤 Пользователь ID={u_id}: {full_name}")

            # Ноды знаний
            category = arch_data["cat"]
            tags = random.sample(TOPICS[category], min(3, len(TOPICS[category])))
            for tag in tags:
                x, y = random.uniform(100, 800), random.uniform(100, 600)
                cur.execute(
                    "INSERT INTO knowledge_nodes (user_id, title, description, category, x, y) VALUES (%s, %s, %s, %s, %s, %s)",
                    (u_id, tag, f"Интерес к {tag}", category, x, y),
                )

            # Чат
            cur.execute(
                "INSERT INTO chats (user1_id, user2_id) VALUES (%s, %s) RETURNING id",
                (1, u_id),
            )
            chat_id = cur.fetchone()[0]

            # Сообщения
            for msg_content in arch_data["msgs"]:
                cur.execute(
                    "INSERT INTO messages (chat_id, author_id, content, message_type) VALUES (%s, %s, %s, %s)",
                    (chat_id, u_id, msg_content, "text"),
                )

            # ai_extracted_interests
            graph_tags = arch_data.get("graph_tags", tags)
            graph_tags_json = json.dumps(graph_tags)
            cur.execute(
                """
                INSERT INTO ai_extracted_interests (user_id, hobbies, topics, skills, dislikes, last_extraction)
                VALUES (%s, %s, %s, '[]', '[]', NOW())
                ON CONFLICT (user_id) DO UPDATE SET hobbies = EXCLUDED.hobbies, topics = EXCLUDED.topics, last_extraction = NOW()
                """,
                (u_id, graph_tags_json, graph_tags_json)
            )

            # 🔥 Регистрируем теги в графе (в одной сессии)
            session = create_session()
            try:
                register_user_tags(session, u_id, graph_tags)
                session.commit()

                # Проверяем, что записалось
                from data.interest_hierarchy import UserInterestGraphWeight
                count = session.query(UserInterestGraphWeight).filter_by(user_id=u_id).count()
                print(f"   ✅ Зарегистрировано {count} весов для тегов: {graph_tags}")
            except Exception as e:
                session.rollback()
                print(f"   ⚠️ Ошибка регистрации: {e}")
            finally:
                session.close()

            # ИИ-анализ (опционально)
            try:
                analysis_result = analyze_user_profile.apply(args=[u_id], kwargs={"force": True}).result
                if isinstance(analysis_result, dict):
                    print(f"   🧠 MBTI: {analysis_result.get('mbti_type', 'N/A')}")
            except Exception as e:
                print(f"   ⚠️ Ошибка анализа: {e}")

        print("\n✅ СИДИНГ ЗАВЕРШЕН!")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    # Генерируем 10 продвинутых профилей
    run_mega_seed(10)