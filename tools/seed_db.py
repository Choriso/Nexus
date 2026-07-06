"""Seed базы данных PostgreSQL для разработки с учетом новой архитектуры ИИ."""
import os
import sys

# Определяем путь к папке, в которой лежит текущий скрипт (tools), и берем её родительскую директорию (Nexus)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import random
import sys
import psycopg2
from faker import Faker

from config import config
# Импортируем задачу напрямую для синхронного запуска в процессе сидинга
from app.ai.personality_analyzer import analyze_user_profile
from data.session import global_init, create_session
from data.user import User

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
    },
    "Душа компании": {
        "msgs": [
            "Привет! Как ваши дела? Погнали завтра все вместе на крутой IT-митап!",
            "Обожаю пробовать новые фреймворки и библиотеки, это всегда так весело и интересно!",
            "Давайте соберемся всей нашей дружной командой в пятницу вечером в баре!",
            "Вау, это просто потрясающая новость! Вы все такие классные, обожаю наш чат!"
        ],
        "cat": "psychology",
    },
    "Творец": {
        "msgs": [
            "Я тут придумал совершенно безумную, но очень красивую идею для нового UI/UX дизайна...",
            "А что если нам полностью заменить текущий бэкенд и переписать критические узлы на Rust?",
            "Вдохновение пришло сегодня глубокой ночью, сразу набросал свежий концепт в Figma.",
            "Искусство должно вдохновлять, поэтому я добавил несколько неоновых элементов в интерфейс."
        ],
        "cat": "hobby",
    },
    "Скейтер/Геймер": {
        "msgs": [
            "Го катку в CS2 прямо сейчас? Я как раз создал отличный приватный сервер для своих.",
            "Вчера выбил невероятно редкий и дорогой скин из нового кейса, зацените скриншот!",
            "Кто-то сталкивался с такой проблемой, как полностью убрать внезапные лаги в Steam?",
            " Мы вчера с пацанами катали три на два, было очень потно, но мы затащили этот раунд!"
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

    # Инициализируем SQLAlchemy для проверки пользователей
    global_init(config.DATABASE_URL)

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()
        print("--- 🚀 Запуск генерации цифровых личностей через конвейер ИИ... ---")

        # Проверяем или создаем дефолтного пользователя с ID=1 (ваш аккаунт для чатов)
        session = create_session()
        my_user = session.query(User).filter(User.id == 1).first()
        if not my_user:
            print("Пользователь с ID=1 не найден. Создаем базового пользователя...")
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

            # 1. Создаем пользователя
            cur.execute(
                "INSERT INTO users (name, email, hashed_password) VALUES (%s, %s, %s) RETURNING id",
                (full_name, email, pwd),
            )
            u_id = cur.fetchone()[0]

            # 2. Генерируем связи и ноды знаний (как в вашем старом скрипте)
            category = arch_data["cat"]
            tags = random.sample(TOPICS[category], min(3, len(TOPICS[category])))

            for tag in tags:
                x, y = random.uniform(100, 800), random.uniform(100, 600)
                cur.execute(
                    """
                    INSERT INTO knowledge_nodes (user_id, title, description, category, x, y) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                """,
                    (u_id, tag, f"Интерес к {tag}", category, x, y),
                )

            # Создаем чат между вами (ID=1) и новым пользователем
            cur.execute(
                "INSERT INTO chats (user1_id, user2_id) VALUES (%s, %s) RETURNING id",
                (1, u_id),
            )
            chat_id = cur.fetchone()[0]

            # 3. Наполняем историю сообщений (это критически важно для работы анализатора!)
            for msg_content in arch_data["msgs"]:
                cur.execute(
                    """
                    INSERT INTO messages (chat_id, author_id, content, message_type) 
                    VALUES (%s, %s, %s, %s)
                """,
                    (chat_id, u_id, msg_content, "text"),
                )

            # 4. 🔥 ТРИГГЕР КОНВЕЙЕРА ИИ
            # Используем .apply(), чтобы запустить задачу Celery СИНХРОННО прямо в этом потоке.
            # Нам не нужно запускать celery worker в отдельном терминале ради сидинга.
            print(f"🧠 Запуск ИИ-анализа для: {full_name}...")
            analysis_result = analyze_user_profile.apply(args=[u_id], kwargs={"force": True}).result
            # ... твой код вызова аналитики ...
            # СИЛОВОЙ КРАШ-ФИКС ДЛЯ СИДЕРА: Записываем теги напрямую в профиль кандидата
            # (замени 'ai_extracted_interests' на имя твоей таблицы, если оно отличается)
            # СИЛОВОЙ КРАШ-ФИКС ДЛЯ СИДЕРА ПОД РЕАЛЬНУЮ СХЕМУ
            try:
                import json
                # Превращаем теги в JSON-массив строк: '["Python", "Flask"]'
                tags_json = json.dumps(tags)

                # Записываем теги одновременно в hobbies и topics для надежности матчинга
                cur.execute(
                    """
                    INSERT INTO ai_extracted_interests (
                        user_id, hobbies, topics, skills, dislikes, last_extraction
                    )
                    VALUES (%s, %s, %s, '[]', '[]', NOW())
                    ON CONFLICT (user_id) 
                    DO UPDATE SET 
                        hobbies = EXCLUDED.hobbies,
                        topics = EXCLUDED.topics,
                        last_extraction = NOW();
                    """,
                    (u_id, tags_json, tags_json)
                )
                print(f"   🎯 Принудительно заполнены hobbies/topics для кандидата: {tags}")
            except Exception as e_tag:
                print(f"   ⚠️ Не удалось принудительно вшить теги: {e_tag}")
            if "error" in analysis_result:
                print(f"⚠️ Ошибка анализа пользователя {u_id}: {analysis_result['error']}")
            else:
                print(f"✅ Успешно проанализирован! MBTI: {analysis_result.get('mbti_type')}")

        print("\n--- 🎉 СИДИНГ УСПЕШНО ЗАВЕРШЕН! База данных заполнена и связана векторно. ---")

    except Exception as e:
        print(f"❌ Критическая ошибка при сидинге: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Сгенерируем 10 продвинутых профилей
    run_mega_seed(10)