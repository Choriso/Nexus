"""Seed БД: 200 пользователей с максимально разными интересами и характерами."""
import os
import sys
import json
import random
import math
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from data.session import global_init, create_session
from data.user import User
from data.ai import UserPersonalityProfile, AIExtractedInterests, UserSchwartzProfile, GlobalWeightsConfig, DynamicAlias
from data.interest_hierarchy import InterestHierarchyNode, UserInterestGraphWeight
from data.chat import Chat
from data.message import Message
from app.ai_profiler.interest_graph import ensure_hierarchy_seeded, register_user_tags
from app.ai_profiler.contextual_adapter import get_contextual_adapter

fake = None
try:
    from faker import Faker
    fake = Faker(["ru_RU"])
except ImportError:
    pass

random.seed(42)

# ─── 16 архетипов, покрывающих ВСЕ ветки иерархии ──────────────────────────

ARCHETYPES = [
    # (имя, список слогов, категория, msgs)
    ("Python-бэкендер", [
        "backend_python", "python_flask", "python_sqlalchemy", "python_asyncio",
        "python_fastapi", "docker", "ci_cd", "it_development",
    ], "work", [
        "Рефакторю легаси на FastAPI, покрываю тестами.",
        "Декомпозирую бэк на микросервисы с очередями.",
        "Смотрю на Python 3.13 — насколько там GIL отключили?",
        "Пишу код стайл гайд для команды бэкенда.",
    ]),
    ("Фронтендер-дизайнер", [
        "frontend_dev", "javascript", "typescript", "react",
        "css", "ux_ui_design", "graphic_design", "animation",
    ], "work", [
        "Верстаю адаптивный дашборд на React + TS.",
        "Спорят о Tailwind vs CSS-in-JS, я за CSS-переменные.",
        "Пишу дизайн-систему для корпоративного портала.",
        "Разбираюсь с accessibility в вебе.",
    ]),
    ("Мобильный разработчик", [
        "mobile_dev", "react_native", "ios_dev", "android_dev",
        "javascript", "it_development",
    ], "work", [
        "Портирую нативное приложение на React Native.",
        "Оптимизирую рендер списков в FlatList.",
        "Добавляю биометрию в мобильное приложение.",
        "Разбираюсь с push-уведомлениями на обеих платформах.",
    ]),
    ("DevOps-инженер", [
        "devops", "docker", "kubernetes", "ci_cd",
        "python_flask", "it_development",
    ], "work", [
        "Настраиваю GitLab CI с канареечным деплоем.",
        "Оптимизирую Docker-образы, мультистейдж билды.",
        "Поднимаю K8s кластер для микросервисов.",
        "Terraform-им инфраструктуру в Yandex Cloud.",
    ]),
    ("Дата-сайентист", [
        "data_science", "machine_learning", "deep_learning", "nlp",
        "backend_python", "python_asyncio",
    ], "work", [
        "Обучаю трансформер на кастомном датасете.",
        "Пайплайн фичей: от сырых логов до embedding-ов.",
        "A/B тестируем новую модель ранжирования.",
        "Визуализирую SHAP values для стейкхолдеров.",
    ]),
    ("Геймдев-энтузиаст", [
        "game_development", "game_engines", "unity", "unreal_engine", "godot",
        "3d_modeling", "blender", "digital_art",
    ], "hobby", [
        "Пишу прототип головоломки на Unity.",
        "Текстурю low-poly модель для инди-игры.",
        "Разбираюсь с шейдерами в Unreal Engine 5.",
        "Собираю команду для геймджема.",
    ]),
    ("Киберспортсмен", [
        "competitive_gaming", "gaming", "indie_games",
        "game_development", "game_engines",
    ], "hobby", [
        "Го катку, залетайте в пати!",
        "Новый патч сбалансил мету, надо тестить.",
        "Смотрь стримы с мейджора, топовый замес.",
        "Фармю рейтинг в соло-очереди.",
    ]),
    ("Музыкальный продюсер", [
        "music_production", "music_genres", "hip_hop", "electronic_music", "jazz",
        "rock_music", "classical_music",
    ], "hobby", [
        "Свёл новый трек в Ableton, нужен мэстеринг.",
        "Собираю плейлист для вечерних кодинг-сессий.",
        "Ищу вокалиста для экспериментального проекта.",
        "Разбираю гармонию джазовых стандартов.",
    ]),
    ("Киноман", [
        "cinema_genres", "scifi", "horror", "drama", "comedy",
        "animation_films", "cinema_video",
    ], "hobby", [
        "Вчера пересмотрел всего Нолана — имба.",
        "Составляю список лучших хорроров десятилетия.",
        "Аниме или западная анимация? У каждого своя магия.",
        "Спорим, что научная фантастика 80-х была лучше?",
    ]),
    ("Книжный червь", [
        "literature_reading", "reading_genres", "fantasy", "science_fiction_lit",
        "mystery", "romance",
    ], "hobby", [
        "Сейчас читаю цикл «Три тела» — мозг кипит.",
        "Ищу рекомендации в жанре магического реализма.",
        "Аудиокниги или бумага? Я за бумагу.",
        "Перечитываю классику: Достоевский всё ещё актуален.",
    ]),
    ("Художник-иллюстратор", [
        "creativity_art", "digital_art", "character_design", "animation",
        "graphic_design", "photography",
    ], "hobby", [
        "Нарисовала концепт-арт персонажа в Procreate.",
        "Ищу стиль: селл-шейд или реализм для комикса.",
        "Запустил серию скетчей каждый день.",
        "Фотографирую городскую архитектуру на плёнку.",
    ]),
    ("Фитнес-активист", [
        "sports_active_life", "fitness_training", "yoga", "hiking_outdoor",
        "team_sports", "football", "basketball",
    ], "life", [
        "Новый жим лёжа — 100 кг, прогресс пошёл!",
        "Йога по утрам — лучший старт дня.",
        "Планирую поход в горы на неделю.",
        "Субботний футбол с ребятами, всех порвём.",
    ]),
    ("Психолог-саморазвитие", [
        "psychology_relations", "mental_health", "relationships",
        "personal_growth", "mindfulness", "therapy",
        "self_development", "personal_growth", "mindfulness",
    ], "life", [
        "Практикую осознанность и медитацию каждый день.",
        "Читаю про когнитивные искажения в отношениях.",
        "Провожу сессию глубинной рефлексии.",
        "Эмпатия — главный навык 21 века.",
    ]),
    ("Учёный-исследователь", [
        "science_education", "physics", "chemistry", "biology",
        "astronomy", "mathematics", "linguistics",
    ], "work", [
        "Читаю статьи по квантовой запутанности.",
        "Экспериментирую с диффузионными моделями в биоинфе.",
        "Математика — язык вселенной, спорно?",
        "Изучаю этимологию санскрита для лингвистики.",
    ]),
    ("Стартапер-инвестор", [
        "finance_business", "startups", "entrepreneurship", "investing",
        "trading", "cryptocurrency", "it_development",
    ], "work", [
        "Паттерн для пре-сида: ищу ко-фаундера.",
        "Портфель на 70% в индексах, 30% в крипте.",
        "Unit-экономика: CAC, LTV, юнит-тесты бизнеса.",
        "Провожу дейли со стартап-командой.",
    ]),
    ("Хозяин-джек-всех-рук", [
        "home_lifestyle", "cooking", "gardening", "interior_design",
        "sustainability", "time_management",
    ], "life", [
        "Завёл гидропонику на подоконнике — урожай салата.",
        "Готовлю пасту с трюфельным маслом по новому рецепту.",
        "Сделал перепланировку квартиры в SketchUp.",
        "Сортирую мусор и компостирую органику.",
    ]),
]

# Распределение OCEAN-характеров: 5 кластеров
OCEAN_CLUSTERS = [
    {"name": "Лидер",       "o": 0.7, "c": 0.8, "e": 0.7, "a": 0.4, "n": 0.3, "mbti": "ENTJ", "comm": "direct", "formality": 0.6, "enthusiasm": 0.7, "detail": 0.8,
     "schwartz": {"self_direction": 0.8, "stimulation": 0.7, "hedonism": 0.4, "achievement": 0.9, "power": 0.8, "security": 0.3, "conformity": 0.2, "tradition": 0.2, "benevolence": 0.4, "universalism": 0.3}},
    {"name": "Творец",      "o": 0.9, "c": 0.3, "e": 0.5, "a": 0.6, "n": 0.5, "mbti": "INFP", "comm": "expressive", "formality": 0.3, "enthusiasm": 0.8, "detail": 0.5,
     "schwartz": {"self_direction": 0.9, "stimulation": 0.8, "hedonism": 0.7, "achievement": 0.5, "power": 0.2, "security": 0.3, "conformity": 0.1, "tradition": 0.3, "benevolence": 0.7, "universalism": 0.8}},
    {"name": "Аналитик",    "o": 0.4, "c": 0.9, "e": 0.2, "a": 0.5, "n": 0.6, "mbti": "INTJ", "comm": "analytical", "formality": 0.8, "enthusiasm": 0.2, "detail": 0.9,
     "schwartz": {"self_direction": 0.7, "stimulation": 0.2, "hedonism": 0.2, "achievement": 0.8, "power": 0.5, "security": 0.6, "conformity": 0.5, "tradition": 0.4, "benevolence": 0.3, "universalism": 0.5}},
    {"name": "Душа компании","o": 0.6, "c": 0.5, "e": 0.9, "a": 0.8, "n": 0.2, "mbti": "ESFJ", "comm": "friendly", "formality": 0.3, "enthusiasm": 0.9, "detail": 0.3,
     "schwartz": {"self_direction": 0.5, "stimulation": 0.6, "hedonism": 0.8, "achievement": 0.6, "power": 0.3, "security": 0.5, "conformity": 0.6, "tradition": 0.5, "benevolence": 0.9, "universalism": 0.6}},
    {"name": "Хранитель",   "o": 0.3, "c": 0.7, "e": 0.3, "a": 0.7, "n": 0.7, "mbti": "ISFJ", "comm": "supportive", "formality": 0.7, "enthusiasm": 0.3, "detail": 0.7,
     "schwartz": {"self_direction": 0.3, "stimulation": 0.2, "hedonism": 0.3, "achievement": 0.5, "power": 0.3, "security": 0.8, "conformity": 0.8, "tradition": 0.8, "benevolence": 0.8, "universalism": 0.4}},
]

COLLAB_STYLES = ["solo", "collaborative", "mentor", "leader", "supporter"]
COMM_STYLES = ["direct", "expressive", "analytical", "friendly", "supportive"]
COMPATIBLE_MBTI = {
    "ENTJ": ["INTP", "INTJ", "ENFP"],
    "INFP": ["ENFJ", "ENTJ", "INFJ"],
    "INTJ": ["ENFP", "ENTP", "ENTJ"],
    "ESFJ": ["ISFP", "ISTP", "INTJ"],
    "ISFJ": ["ESFP", "ESTP", "ENFP"],
}


def _make_ocean_noise(base: dict, noise: float = 0.15) -> dict:
    """Add random noise to OCEAN traits while keeping in [0, 1]."""
    return {
        k: max(0.05, min(0.95, v + random.uniform(-noise, noise)))
        for k, v in base.items() if k in ("o", "c", "e", "a", "n")
    }


def _embedding_from_ocean(ocean: dict) -> list[float]:
    """Generate a 5D pgvector embedding from OCEAN traits."""
    return [ocean["o"], ocean["c"], ocean["e"], ocean["a"], ocean["n"]]


def run_seed(num_users: int = 200) -> None:
    global_init(config.DATABASE_URL)
    db = create_session()

    try:
        # 1 ─── Убедиться, что иерархия посеяна ─────────────────────────
        ensure_hierarchy_seeded(db)
        db.commit()
        print("[OK] Иерархия графа проверена")

        # 2 ─── Создать/обновить пользователя ID=1 (тестовый текущий) ────
        user1 = db.query(User).filter(User.id == 1).first()
        if not user1:
            user1 = User(id=1, name="Данила Фещенко", email="danila@nexus.com", hashed_password="scrypt:32768:8:1$fake_hash")
            db.add(user1)
            db.flush()
        user1_id = 1

        # 3 ─── Собираем узлы иерархии ──────────────────────────────────
        all_nodes: list[InterestHierarchyNode] = db.query(InterestHierarchyNode).all()
        slug_to_node = {n.slug: n for n in all_nodes}
        all_slugs = list(slug_to_node.keys())
        node_ids = {n.slug: n.id for n in all_nodes}

        # 4 ─── Генерация seed-пользователей ────────────────────────────
        print(f"\n--- Генерация {num_users} пользователей ---")

        created_count = 0
        for i in range(num_users):
            # Выбираем архетип
            archetype = random.choice(ARCHETYPES)
            arch_slugs = archetype[1]
            arch_cat = archetype[2]
            arch_msgs = archetype[3]

            # Выбираем кластер OCEAN (с небольшим шансом случайный)
            if random.random() < 0.15:
                ocean_base = {
                    "o": random.random(),
                    "c": random.random(),
                    "e": random.random(),
                    "a": random.random(),
                    "n": random.random(),
                }
                cluster = None
            else:
                cluster = random.choice(OCEAN_CLUSTERS)
                ocean_base = {k: cluster[k] for k in ("o", "c", "e", "a", "n")}

            ocean = _make_ocean_noise(ocean_base)

            # Добавляем немного дополнительных слогов (не из архетипа) для разнообразия
            extra_slugs = random.sample([s for s in all_slugs if s not in arch_slugs], k=random.randint(0, 3))
            user_slugs = list(set(arch_slugs + extra_slugs))

            # Генерируем имя
            if fake:
                full_name = f"{fake.first_name()} {fake.last_name()}"
                email = f"{i+2}_{full_name.lower().replace(' ', '.')}@nexus.com"
            else:
                full_name = f"User{i+2}"
                email = f"user{i+2}@nexus.com"
            pwd = "scrypt:32768:8:1$fake_hash_value"

            # Создаём пользователя
            user = User(
                name=full_name,
                email=email,
                hashed_password=pwd,
                information=f"User archetype: {archetype[0]}, ocean: {cluster['name'] if cluster else 'random'}",
                is_moderator=False,
            )
            db.add(user)
            db.flush()
            u_id = user.id

            # 5 ─── AIExtractedInterests ────────────────────────────────
            skills_list = [s for s in user_slugs if slug_to_node.get(s) and slug_to_node[s].depth >= 2][:4]
            pref_json = {
                "archetype": archetype[0],
                "cluster": cluster["name"] if cluster else "random",
                "tags": user_slugs,
            }
            ai_ext = AIExtractedInterests(
                user_id=u_id,
                hobbies=json.dumps(user_slugs),
                topics=json.dumps(user_slugs[:5]),
                skills=json.dumps(skills_list if skills_list else user_slugs[:3]),
                dislikes=json.dumps([]),
                preferences=json.dumps(pref_json),
                last_extraction=datetime.utcnow(),
            )
            db.add(ai_ext)

            # 6 ─── Personality Profile ─────────────────────────────────
            if cluster:
                mbti = cluster["mbti"]
                comm_style = cluster["comm"]
                formality = cluster["formality"]
                enthusiasm = cluster["enthusiasm"]
                detail = cluster["detail"]
            else:
                mbti = random.choice(list(COMPATIBLE_MBTI.keys()))
                comm_style = random.choice(COMM_STYLES)
                formality = random.random()
                enthusiasm = random.random()
                detail = random.random()

            traits_json = json.dumps({
                "archetype": archetype[0],
                "openness_desc": "high" if ocean["o"] > 0.6 else "low",
                "conscientiousness_desc": "high" if ocean["c"] > 0.6 else "low",
                "extraversion_desc": "high" if ocean["e"] > 0.6 else "low",
            })
            values_json = json.dumps(cluster["schwartz"] if cluster else {k: random.random() for k in
                ("self_direction", "stimulation", "hedonism", "achievement", "power", "security", "conformity", "tradition", "benevolence", "universalism")})

            profile = UserPersonalityProfile(
                user_id=u_id,
                openness=ocean["o"],
                conscientiousness=ocean["c"],
                extraversion=ocean["e"],
                agreeableness=ocean["a"],
                neuroticism=ocean["n"],
                mbti_type=mbti,
                communication_style=comm_style,
                formality=formality,
                enthusiasm=enthusiasm,
                detail_oriented=detail,
                traits=traits_json,
                values=values_json,
                compatible_mbti_types=json.dumps(COMPATIBLE_MBTI.get(mbti, [])),
                collaboration_style=random.choice(COLLAB_STYLES),
                confidence_score=random.uniform(0.6, 0.95),
                last_analyzed=datetime.utcnow(),
                conversation_count=random.randint(1, 50),
            )
            profile.embedding = _embedding_from_ocean(ocean)
            db.add(profile)

            # 7 ─── Schwartz Profile ────────────────────────────────────
            sv = cluster["schwartz"] if cluster else {k: random.random() for k in
                ("self_direction", "stimulation", "hedonism", "achievement", "power", "security", "conformity", "tradition", "benevolence", "universalism")}
            schwartz = UserSchwartzProfile(
                user_id=u_id,
                self_direction=sv["self_direction"],
                stimulation=sv["stimulation"],
                hedonism=sv["hedonism"],
                achievement=sv["achievement"],
                power=sv["power"],
                security=sv["security"],
                conformity=sv["conformity"],
                tradition=sv["tradition"],
                benevolence=sv["benevolence"],
                universalism=sv["universalism"],
                values_json=json.dumps({k: round(v, 3) for k, v in sv.items()}),
                confidence_score=random.uniform(0.6, 0.95),
            )
            db.add(schwartz)

            # 8 ─── Регистрация весов в графе ──────────────────────────
            register_user_tags(db, u_id, user_slugs)

            # Добавляем случайный шум к весам графа (±0.1) для вариативности
            from data.interest_hierarchy import UserInterestGraphWeight
            user_weights = db.query(UserInterestGraphWeight).filter_by(user_id=u_id).all()
            for uw in user_weights:
                noise = random.uniform(-0.1, 0.1)
                uw.weight = max(0.05, min(1.0, uw.weight + noise))
                db.add(uw)
            db.commit()

            # 9 ─── Чат с пользователем 1 ───────────────────────────────
            chat = Chat(user1_id=user1_id, user2_id=u_id)
            db.add(chat)
            db.flush()
            chat_id = chat.id

            # Сообщения
            for text in arch_msgs:
                msg = Message(
                    chat_id=chat_id,
                    author_id=u_id,
                    content=text,
                    message_type="text",
                )
                db.add(msg)

            # Добавляем 1-2 случайных сообщения от user1
            for _ in range(random.randint(1, 2)):
                reply = random.choice([
                    "Круто, расскажи подробнее!",
                    "Интересно, а как ты к этому пришёл?",
                    "Хорошая мысль, согласен.",
                    "А я вот вообще не разбираюсь в этом.",
                    "Вау, звучит захватывающе!",
                ])
                msg = Message(
                    chat_id=chat_id,
                    author_id=user1_id,
                    content=reply,
                    message_type="text",
                )
                db.add(msg)

            # 10 ─── Knowledge nodes (для фронтенда) ────────────────────
            for tag in user_slugs[:4]:
                node = slug_to_node.get(tag)
                if not node:
                    continue
                from data.knowledge_graph import KnowledgeNode
                kn = KnowledgeNode(
                    user_id=u_id,
                    title=node.name,
                    description=f"Интерес к {node.name}",
                    category=arch_cat,
                    x=random.uniform(100, 800),
                    y=random.uniform(100, 600),
                )
                db.add(kn)

            created_count += 1
            if created_count % 20 == 0:
                db.commit()
                print(f"  [{created_count}/{num_users}] пользователей создано...")

        db.commit()
        print(f"\n[OK] Создано {created_count} пользователей")

        # 11 ─── GlobalWeightsConfig ────────────────────────────────────
        gwc = GlobalWeightsConfig.get_or_create(db)
        print(f"   Global weights: ocean={gwc.weight_ocean}, graph={gwc.weight_graph}, jaccard={gwc.weight_jaccard}")

        # 12 ─── Cтатистика ────────────────────────────────────────────
        total_users = db.query(User).count()
        total_profiles = db.query(UserPersonalityProfile).count()
        total_weights = db.query(UserInterestGraphWeight).count()
        total_chats = db.query(Chat).count()
        print(f"\nСтатистика БД:")
        print(f"   Users: {total_users}")
        print(f"   Personality profiles: {total_profiles}")
        print(f"   Graph weights: {total_weights}")
        print(f"   Chats: {total_chats}")

        # Распределение MBTI
        import sqlalchemy as sa
        mbti_counts = db.query(UserPersonalityProfile.mbti_type, sa.func.count()).group_by(
            UserPersonalityProfile.mbti_type).all()
        print(f"   MBTI distribution: {dict(mbti_counts)}")

        # Распределение архетипов (из preferences)
        arch_counts = {}
        for ext in db.query(AIExtractedInterests.preferences).all():
            if ext.preferences:
                pref = json.loads(ext.preferences) if isinstance(ext.preferences, str) else ext.preferences
                a = pref.get("archetype", "unknown")
                arch_counts[a] = arch_counts.get(a, 0) + 1
        print(f"   Archetype distribution: {dict(sorted(arch_counts.items()))}")

    except Exception as e:
        db.rollback()
        print(f"[ERROR] Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed БД тестовыми пользователями")
    parser.add_argument("-n", "--num", type=int, default=200, help="Количество пользователей")
    args = parser.parse_args()
    run_seed(args.num)
