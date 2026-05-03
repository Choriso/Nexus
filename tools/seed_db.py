"""Seed opcional da base PostgreSQL para desenvolvimento (sem credenciais em código)."""

import os
import random
import sys

import psycopg2
from faker import Faker

DB_CONFIG = {
    "dbname": os.environ.get("SEED_DB_NAME", "ai_profiler_db"),
    "user": os.environ.get("SEED_DB_USER", "my_app_user"),
    "password": os.environ.get("SEED_DB_PASSWORD", ""),
    "host": os.environ.get("SEED_DB_HOST", "127.0.0.1"),
    "port": os.environ.get("SEED_DB_PORT", "5432"),
}

fake = Faker(["ru_RU"])

ARCHETYPES = {
    "Профи": {
        "ocean": [0.4, 0.9, 0.3, 0.4, 0.8],
        "msgs": [
            "Нам нужно декомпозировать задачи на спринты.",
            "Кодстайл хромает, нужно поправить отступы.",
            "Я изучил документацию, там есть баг в API.",
        ],
        "cat": "work",
    },
    "Душа компании": {
        "ocean": [0.8, 0.3, 1.0, 0.9, 0.2],
        "msgs": [
            "Привет! Как дела? Пойдем завтра на митап?",
            "Обожаю пробовать новые фреймворки, это так весело!",
            "Давайте соберемся всей командой в пятницу!",
        ],
        "cat": "psychology",
    },
    "Творец": {
        "ocean": [1.0, 0.4, 0.6, 0.5, 0.4],
        "msgs": [
            "Я тут придумал безумную идею для дизайна...",
            "А что если мы заменим весь бэкенд на Rust?",
            "Вдохновение пришло ночью, набросал новый концепт.",
        ],
        "cat": "hobby",
    },
    "Скейтер/Геймер": {
        "ocean": [0.7, 0.2, 0.8, 0.4, 0.3],
        "msgs": [
            "Го катку в CS2? Я создал приватный сервер.",
            "Вчера выбил редкий скин, зацени!",
            "Кто-то знает, как убрать лаги в Steam?",
        ],
        "cat": "hobby",
    },
}

TOPICS = {
    "work": ["Python", "Flask", "PostgreSQL", "Архитектура", "Deployment"],
    "hobby": ["3D Modeling", "Blender", "CS2", "Unity", "Digital Art"],
    "psychology": ["Эмпатия", "Тайм-менеджмент", "Лидерство", "Медитация", "Soft Skills"],
}


def run_mega_seed(num_users: int = 10) -> None:
    if not DB_CONFIG["password"]:
        print(
            "Erro: defina SEED_DB_PASSWORD (e opcionalmente SEED_DB_USER, SEED_DB_NAME, ...).",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        print("--- Запуск генерации цифровых личностей... ---")

        my_id = 1

        for _ in range(num_users):
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

            o, c, e, a, n = arch_data["ocean"]
            o, c, e, a, n = [
                max(0.1, min(1.0, val + random.uniform(-0.1, 0.1)))
                for val in [o, c, e, a, n]
            ]

            cur.execute(
                """
                INSERT INTO ai_user_personality_profiles (user_id, openness, conscientiousness, extraversion, agreeableness, neuroticism)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                (u_id, o, c, e, a, n),
            )

            category = arch_data["cat"]
            tags = random.sample(TOPICS[category], 3)

            for tag in tags:
                x, y = random.uniform(100, 800), random.uniform(100, 600)
                cur.execute(
                    """
                    INSERT INTO knowledge_nodes (user_id, title, description, category, x, y) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                """,
                    (u_id, tag, f"Интерес к {tag}", category, x, y),
                )

            cur.execute(
                "INSERT INTO chats (user1_id, user2_id) VALUES (%s, %s) RETURNING id",
                (my_id, u_id),
            )
            chat_id = cur.fetchone()[0]

            for msg_content in arch_data["msgs"]:
                cur.execute(
                    """
                    INSERT INTO messages (chat_id, author_id, content, message_type) 
                    VALUES (%s, %s, %s, %s)
                """,
                    (chat_id, u_id, msg_content, "text"),
                )

            print(f"Добавлен: {full_name} | Категория: {category}")

        print("\n--- ГОТОВО! Проверь поиск на сайте. ---")

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    run_mega_seed(12)
