"""Seed базы данных PostgreSQL с гарантированно резолвимыми тегами."""
import os
import sys
import json
import random
import psycopg2
from faker import Faker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from data.session import global_init, create_session
from data.user import User
from app.ai_profiler.interest_graph import register_user_tags, ensure_hierarchy_seeded

DB_CONFIG = config.seed_db_config()
fake = Faker(["ru_RU"])

# Архетипы с ТОЧНО резолвимыми тегами (те, что есть в CANONICAL_SLUG_FIXES или _HIERARCHY_SEED)
# Архетипы с ИМЕНАМИ ИЗ _HIERARCHY_SEED (display_name), которые точно есть в базе
# Архетипы с РЕАЛЬНЫМИ СЛАГАМИ из _HIERARCHY_SEED
ARCHETYPES = {
    "Профи": {
        "msgs": [
            "Декомпозируем задачи на спринты, нужны эстимейты.",
            "Кодстайл хромает, поправьте отступы.",
            "Техническая документация – баг в деплой API.",
            "Рефакторинг микросервиса на бэкенде.",
        ],
        "cat": "work",
        # Реальные слаги из _HIERARCHY_SEED
        "tags": ["it_development", "backend_python", "python_flask", "api_design", "databases"]
    },
    "Душа компании": {
        "msgs": [
            "Привет! Как дела? Идём на IT-митап!",
            "Обожаю новые фреймворки, это весело!",
            "Соберёмся командой в баре в пятницу!",
            "Вау, вы классные, обожаю наш чат!",
        ],
        "cat": "psychology",
        "tags": ["psychology_relations", "soft_skills_empathy", "emotional_intelligence", "networking"]
    },
    "Творец": {
        "msgs": [
            "Придумал безумную идею для UI/UX дизайна...",
            "А что если переписать бэкенд на Rust?",
            "Набросал свежий концепт в Figma.",
            "Искусство должно вдохновлять.",
        ],
        "cat": "hobby",
        "tags": ["creativity_art", "digital_art", "figma_tool", "graphic_design", "3d_modeling_game"]
    },
    "Скейтер/Геймер": {
        "msgs": [
            "Го катку в CS2 прямо сейчас!",
            "Выбил редкий скин, зацените!",
            "Кто сталкивался с лагами в Steam?",
            "Мы вчера катали три на два, затащили!",
        ],
        "cat": "hobby",
        "tags": ["gaming", "cs2_game", "cybersport", "dota2_game", "valorant_game"]
    },
}

def run_mega_seed(num_users: int = 10) -> None:
    if not DB_CONFIG["password"]:
        print("❌ Error: SEED_DB_PASSWORD не задан")
        sys.exit(1)

    global_init(config.DATABASE_URL)

    # Инициализируем иерархию графа (если ещё не)
    session = create_session()
    try:
        ensure_hierarchy_seeded(session)
        session.commit()
        print("✅ Иерархия графа проверена")
    except Exception as e:
        session.rollback()
        print(f"⚠️ Ошибка иерархии: {e}")
    finally:
        session.close()

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        print("--- 🚀 Генерация новых профилей ---")

        # Убедимся, что есть пользователь ID=1
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

            # 1. Пользователь
            cur.execute(
                "INSERT INTO users (name, email, hashed_password) VALUES (%s, %s, %s) RETURNING id",
                (full_name, email, pwd),
            )
            u_id = cur.fetchone()[0]
            print(f"\n👤 ID={u_id} {full_name}")

            # 2. Knowledge nodes (необязательно, но для фронтенда)
            category = arch_data["cat"]
            tags = arch_data["tags"][:3]  # первые 3 тега для узлов
            for tag in tags:
                x, y = random.uniform(100, 800), random.uniform(100, 600)
                cur.execute(
                    """INSERT INTO knowledge_nodes (user_id, title, description, category, x, y)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (u_id, tag, f"Интерес к {tag}", category, x, y),
                )

            # 3. Чат с ID=1
            cur.execute(
                "INSERT INTO chats (user1_id, user2_id) VALUES (%s, %s) RETURNING id",
                (1, u_id),
            )
            chat_id = cur.fetchone()[0]

            # 4. Сообщения
            for msg_content in arch_data["msgs"]:
                cur.execute(
                    "INSERT INTO messages (chat_id, author_id, content, message_type) VALUES (%s, %s, %s, %s)",
                    (chat_id, u_id, msg_content, "text"),
                )

            # 5. AIExtractedInterests с гарантированно резолвимыми тегами
            resolved_tags = arch_data["tags"]  # они точно зарезолвятся
            hobbies_json = json.dumps(resolved_tags)
            topics_json = json.dumps(resolved_tags[:2])  # немного для topics
            skills_json = json.dumps(resolved_tags[:2])
            cur.execute(
                """INSERT INTO ai_extracted_interests (user_id, hobbies, topics, skills, dislikes, last_extraction)
                   VALUES (%s, %s, %s, %s, '[]', NOW())
                   ON CONFLICT (user_id) DO UPDATE SET
                       hobbies = EXCLUDED.hobbies,
                       topics = EXCLUDED.topics,
                       skills = EXCLUDED.skills,
                       last_extraction = NOW()""",
                (u_id, hobbies_json, topics_json, skills_json),
            )

            # 6. РЕГИСТРАЦИЯ ВЕСОВ В ГРАФЕ (самое важное!)
            session = create_session()
            try:
                register_user_tags(session, u_id, resolved_tags)
                session.commit()
                from data.interest_hierarchy import UserInterestGraphWeight
                count = session.query(UserInterestGraphWeight).filter_by(user_id=u_id).count()
                print(f"   ✅ Зарегистрировано {count} весов")
            except Exception as e:
                session.rollback()
                print(f"   ⚠️ Ошибка регистрации весов: {e}")
            finally:
                session.close()

            # 7. Опционально: AI анализ личности (может быть медленно, можно закомментировать)
            # print(f"   🧠 Запуск AI анализа...")
            # try:
            #     from app.ai.personality_analyzer import analyze_user_profile
            #     result = analyze_user_profile.apply(args=[u_id], kwargs={"force": True}).result
            #     if isinstance(result, dict) and "error" not in result:
            #         print(f"   ✅ MBTI: {result.get('mbti_type')}")
            # except Exception as e:
            #     print(f"   ⚠️ Ошибка анализа: {e}")

        print("\n--- ✅ СИД ГОТОВ ---")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run_mega_seed(10)