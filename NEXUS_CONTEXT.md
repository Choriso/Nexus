# Nexus — Project Architecture & Technical Reference

> **Назначение:** Единый технический справочник платформы Nexus. Содержит описание архитектуры, схемы БД, потоков данных, конфигурации и всех AI-компонентов. Предназначен для разработчиков и LLM-ассистентов.

---

## 1. ОБЗОР ПРОЕКТА И СТЕК

### 1.1. Суть проекта

**Nexus** — социальная платформа для поиска единомышленников на основе психологической совместимости и общих интересов. Три ключевых слоя:

| Слой | Назначение |
|------|------------|
| **Психологический профиль** | Автоматический анализ личности: OCEAN (Big Five), MBTI, ценности Шварц, поведенческие метрики |
| **Граф знаний интересов** | Визуальный граф узлов (`knowledge_nodes`) + иерархия интересов + AI-извлечённые интересы |
| **Двухконтурный поиск** | Быстрый pgvector OCEAN-поиск + гибридный скоринг (Graph × OCEAN × Jaccard) с микро-градиентной подстройкой |

### 1.2. Технологический стек

```
┌──────────────────────────────────────────────────────────────┐
│  Клиент: HTML/JS шаблоны (Jinja2) + Chart.js + D3.js (граф)  │
├──────────────────────────────────────────────────────────────┤
│  Flask 3.x │ Flask-Login │ Flask-SocketIO │ Flask-CORS        │
│  Blueprints: auth, profile, interests, chat, moderation,      │
│              analytics, routes (main)                         │
├──────────────────────────────────────────────────────────────┤
│  Celery 5.x ←→ Redis (broker + result backend)               │
├──────────────────────────────────────────────────────────────┤
│  PyTorch 2.x + SentenceTransformers + Transformers            │
│  Модели: PersonalityClassifier (OCEAN), MBTIClassifier        │
├──────────────────────────────────────────────────────────────┤
│  YandexGPT (API) — генерация match-отчётов                   │
│  Ollama (опционально) — fallback для LLM-запросов             │
├──────────────────────────────────────────────────────────────┤
│  SQLAlchemy 2.x │ Alembic / Flask-Migrate                     │
│  PostgreSQL 16 + pgvector 0.2.x (Docker: pgvector/pgvector)  │
│  Redis 7.x (кэш + брокер Celery)                             │
└──────────────────────────────────────────────────────────────┘
```

**Python:** 3.12+
**Точка входа:** `App.py` → `create_app()` из `app/__init__.py` → `socketio.run(app)`
**Конфигурация:** единый модуль `config.py` (корень проекта). Все env-переменные читаются только там; остальные модули импортируют `config`.

**Инфраструктура (Docker):**
```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: my_app_user
      POSTGRES_DB: nexus_db
```

**Redis** (ожидается локально): брокер `redis://localhost:6379/0`, кэш `redis://localhost:6379/1`.

---

## 2. АРХИТЕКТУРА И СТРУКТУРА ДАННЫХ

### 2.1. Карта директорий

```
Nexus/
├── App.py                          # Точка входа WSGI
├── config.py                       # Единая конфигурация (Config, DevelopmentConfig, …)
├── .env                            # env-переменные (не в Git)
├── NEXUS_CONTEXT.md                # Этот файл
│
├── app/                            # Flask-приложение
│   ├── __init__.py                 # create_app(), регистрация blueprints, logging
│   ├── extensions.py               # login_manager, socketio, cors, migrate
│   ├── auth.py                     # Регистрация / авторизация
│   ├── profile.py                  # Профиль, аватар, API графа, matching, viewProfile
│   ├── chat.py                     # REST + Socket.IO чаты, триггер AI-анализа
│   ├── interests.py                # Лента интересов, matching по узлам графа
│   ├── moderation.py               # Модерация, жалобы
│   ├── analytics.py                # Дашборд модератора
│   ├── db.py                       # get_db_session() — контекстный менеджер сессий
│   ├── routes.py                   # Главная страница, редиректы
│   │
│   ├── ai/                         # AI-слой (Celery, LLM, matching)
│   │   ├── match_report.py         # ★ Генерация AI-ревью совместимости (YandexGPT → Ollama → fallback)
│   │   ├── personality_analyzer.py # ★ Celery-задачи (analyze_user_profile, update_compatibility)
│   │   ├── celery_tasks.py         # Re-export задач для Celery worker
│   │   ├── matching_engine.py      # Legacy matching (под вопросом)
│   │   ├── ollama_service.py       # Сервис для Ollama API
│   │   ├── models.py               # Legacy: ai_conversation_analysis, ai_training_metrics
│   │   ├── data_processor.py       # TF-IDF препроцессор (legacy)
│   │   └── profiler_singleton.py   # Re-export get_profiler()
│   │
│   └── ai_profiler/                # ★ Ядро AI-профилирования
│       ├── __init__.py             # get_profiler() — thread-safe singleton
│       ├── core.py                 # ★ AIProfiler, PersonalityClassifier, MBTIClassifier
│       ├── text_utils.py           # clean_user_text() — очистка текста
│       ├── contextual_adapter.py   # ContextualAdapter — SBERT-обогащение, taxonomy
│       ├── providers.py            # ★ LLMProvider, FailoverCascade, YandexGPTProvider
│       ├── dynamic_enrichment.py   # ★ DynamicTagEnricher — Two-Stage Semantic Routing
│       ├── search_ranking.py       # ★ Multi-metric scoring, micro_gradient_step
│       ├── root_personalities.py   # ★ Root archetypes (work/entertainment/life), scoring
│       ├── interest_graph.py       # Граф интересов: resolve_tags, overlap, jaccard
│       ├── interest_extractor.py   # CustomInterestClassifier, ZeroShot-экстрактор
│       ├── schwartz_analyzer.py    # Анализ ценностей Шварц
│       ├── behavior_analyzer.py    # Анализ поведенческих паттернов
│       ├── semantic_ontology.py    # SEMANTIC_ONTOLOGY — канонические алиасы tag→slug
│       ├── taxonomy.py             # INTEREST_TAXONOMY — якоря zero-shot
│       └── constants.py            # Общие константы
│
├── data/                           # ★ SQLAlchemy ORM-модели (единый слой данных)
│   ├── session.py                  # SqlAlchemyBase, global_init(), create_session()
│   ├── __all_models.py             # Импорт всех моделей для Alembic
│   ├── user.py                     # users + metric_weight_*_offset
│   ├── ai.py                       # ★ UserPersonalityProfile, AIExtractedInterests,
│   │                               #   UserSchwartzProfile, UserCompatibility,
│   │                               #   DynamicAlias, GlobalWeightsConfig
│   ├── behavior.py                 # UserBehaviorProfile
│   ├── interest_hierarchy.py       # InterestHierarchyNode, UserInterestGraphWeight
│   ├── knowledge_graph.py          # KnowledgeNode, KnowledgeConnection
│   ├── interest.py                 # interests (социальная лента)
│   ├── favorite_interest.py        # favorite_interests (M2M)
│   ├── chat.py                     # chats
│   ├── message.py                  # messages
│   ├── chat_settings.py            # chat_settings
│   └── report.py                   # reports (жалобы)
│
├── ml/                             # Обучение моделей
│   ├── train_ocean.py              # Обучение PersonalityClassifier
│   ├── train_mbti.py               # Обучение MBTIClassifier
│   ├── preprocess.py               # Подготовка датасета, SBERT-эмбеддинги
│   └── artifacts/                  # *.pth веса моделей
│
├── tools/
│   ├── seed_db.py                  # ★ Наполнение БД (200 users, archetypes, chats)
│   └── test_node_match.py          # ★ Тестер графового matching
│
├── migrations/                     # Alembic (включая pgvector)
├── templates/                      # Jinja2-шаблоны (14 файлов)
│   ├── base.html                   # Базовый layout
│   ├── index.html                  # ★ Главная: граф + сайдбар кандидатов + AI Toast
│   ├── profile.html                # Свой профиль (OCEAN, Schwartz)
│   ├── view_profile.html           # Чужой профиль (псих. карточка, интересы)
│   ├── chat.html, login.html, …
│   └── analytics.html, moderation.html
│
├── static/
│   ├── src/layouts/base.css        # Основные стили
│   ├── src/styles/variables.css    # ★ Все CSS-переменные (цвета, радиусы, тени)
│   ├── src/components/             # button.css, card.css, input.css
│   └── CSS/                        # Специфичные стили (чат, лента, …)
│
└── test_metrics.py                 # ★ 55 тестов метрик (graph, schwartz, root_personality, jaccard)
```

### 2.2. Полная ER-схема

```mermaid
erDiagram
    User ||--o| UserPersonalityProfile : "has"
    User ||--o| AIExtractedInterests : "has"
    User ||--o| UserSchwartzProfile : "has"
    User ||--o| UserBehaviorProfile : "has"
    User ||--o{ UserCompatibility : "user_id_1"
    User ||--o{ UserCompatibility : "user_id_2"
    User ||--o{ KnowledgeNode : "owns"
    User ||--o{ Interest : "creates"
    User ||--o{ UserInterestGraphWeight : "weights"
    User ||--o{ Message : "author"
    User ||--o{ Chat : "participant"
    Chat ||--o{ Message : "contains"
    Chat ||--o{ ChatSettings : "settings"
    KnowledgeNode ||--o{ KnowledgeConnection : "from_node"
    KnowledgeNode ||--o{ KnowledgeConnection : "to_node"
    InterestHierarchyNode ||--o{ UserInterestGraphWeight : "has_weights"
    InterestHierarchyNode ||--o{ InterestHierarchyNode : "parent"

    User {
        int id PK
        string name
        string email
        string hashed_password
        string information
        string connection
        string image_path
        bool allow_location
        bool is_moderator
        float metric_weight_ocean_offset
        float metric_weight_graph_offset
        float metric_weight_jaccard_offset
    }

    UserPersonalityProfile {
        int user_id PK_FK
        float openness
        float conscientiousness
        float extraversion
        float agreeableness
        float neuroticism
        vector embedding "Vector(5)"
        string mbti_type "INTJ, ENFP, …"
        string communication_style
        float formality
        float enthusiasm
        float detail_oriented
        string collaboration_style
        json traits
        json values
        json compatible_mbti_types
        float confidence_score
        int conversation_count
        datetime last_analyzed
    }

    UserSchwartzProfile {
        int user_id PK_FK
        float self_direction
        float stimulation
        float hedonism
        float achievement
        float power
        float security
        float conformity
        float tradition
        float benevolence
        float universalism
        json values_json
        float confidence_score
    }

    UserBehaviorProfile {
        int user_id PK_FK
        float avg_char_count
        float avg_reply_time
        float avg_emoji_count
        float avg_hour
        int message_count
    }

    AIExtractedInterests {
        int user_id PK_FK
        json hobbies
        json topics
        json skills
        json dislikes
        string occupation
        text work_style
        json short_term_goals
        json long_term_goals
        json preferences
    }

    UserCompatibility {
        int id PK
        int user_id_1 FK
        int user_id_2 FK
        float overall_score
        float romantic_score
        float professional_score
        float creative_score
        float interest_overlap
        json recommendations
    }

    KnowledgeNode {
        int id PK
        int user_id FK
        string title
        string description
        string category "work | hobby | psychology | want"
        float x "позиция на графе"
        float y "позиция на графе"
    }

    KnowledgeConnection {
        int id PK
        int from_node_id FK
        int to_node_id FK
        string label
    }

    InterestHierarchyNode {
        int id PK
        string name
        string slug "уникальный идентификатор"
        int parent_id FK
        string path "materialized path: /1/5/12/"
        int depth
        float match_weight
        string global_category
        vector embedding "Vector(384)"
    }

    UserInterestGraphWeight {
        int id PK
        int user_id FK
        int node_id FK
        float weight
        string source_tag
    }

    DynamicAlias {
        int id PK
        string raw_tag "исходный тег"
        string slug "разрешенный слаг"
        string tag_hash "MD5"
        float confidence
        text enriched_context
        string source "ollama | direct | усечение"
    }

    GlobalWeightsConfig {
        int id PK "always 1"
        float weight_ocean
        float weight_graph
        float weight_jaccard
        float learning_rate
    }
```

### 2.3. Модуль `data/` — ORM-слой

Все таблицы описаны в `data/*.py`, наследуют `SqlAlchemyBase` из `data/session.py`.

| Файл | Таблицы | Роль |
|------|---------|------|
| `user.py` | `users` | Аутентификация, смещения весов метрик для поиска |
| `ai.py` | `ai_user_personality_profiles` | OCEAN + MBTI + embedding Vector(5) |
| `ai.py` | `ai_extracted_interests` | JSON-интересы, цели, профессия |
| `ai.py` | `user_schwartz_profiles` | 10 ценностей Шварц |
| `ai.py` | `ai_user_compatibility` | Попарные оценки совместимости |
| `ai.py` | `dynamic_aliases` | Кэш разрешения тегов → слаг |
| `ai.py` | `global_weights_config` | Глобальные веса метрик (1 строка) |
| `behavior.py` | `user_behavior_profiles` | Агрегаты стиля общения |
| `knowledge_graph.py` | `knowledge_nodes` | Визуальные узлы графа |
| `knowledge_graph.py` | `knowledge_connections` | Связи между узлами |
| `interest_hierarchy.py` | `interest_hierarchy_nodes` | Иерархия интересов |
| `interest_hierarchy.py` | `user_interest_graph_weights` | Персональные веса узлов |
| `message.py` | `messages` | Сообщения чатов (источник AI-анализа) |
| `chat.py` | `chats` | Диалоги 1:1 |
| `interest.py` | `interests` | Публикации интересов (социальная лента) |
| `favorite_interest.py` | `favorite_interests` | M2M избранное |
| `report.py` | `reports` | Жалобы модерации |
| `chat_settings.py` | `chat_settings` | Настройки чатов |

---

## 3. BLUEPRINTS И ЭНДПОИНТЫ

### 3.1. Регистрация

```python
# app/__init__.py
from .routes import main_bp
from .auth import auth_bp
from .profile import profile_bp
from .interests import interests_bp
from .chat import chat_bp
from .moderation import moderation_bp
from .analytics import analytics_bp
```

### 3.2. Ключевые эндпоинты

| Маршрут | Метод | Blueprint | Описание |
|---------|-------|-----------|----------|
| `/` | GET | `main` | Стартовая страница |
| `/intereses` | GET | `main` | Главная страница с графом |
| `/viewProfile?user_id=N` | GET | `profile` | Профиль другого пользователя (MBTI, OCEAN, Schwartz, интересы) |
| `/profile` | GET/POST | `profile` | Свой профиль с AI-психотипом |
| `/api/graph/match/<node_id>` | GET | `profile` | Поиск кандидатов по узлу графа |
| `/api/graph/report/<user_id>?node_id=N` | GET | `profile` | AI-ревью совместимости (match_report) |
| `/knowledge_graph_data` | GET | `profile` | Данные для визуализации графа |
| `/knowledge_graph/node` | POST | `profile` | Создать узел графа |
| `/knowledge_graph/node/<id>` | PUT | `profile` | Обновить узел |
| `/knowledge_graph/node/<id>` | DELETE | `profile` | Удалить узел |
| `/create_chat/<user_id>` | POST | `chat` | Создать чат с пользователем |
| `/api/graph/report/<user_id>` | GET | `profile` | Получить AI-отчёт совместимости |
| `/upload_avatar` | POST | `profile` | Загрузить аватар |
| `/favorite/<interest_id>` | POST | `interests` | Добавить/убрать избранное |

---

## 4. AI-КОНВЕЙЕР (DETAILED PIPELINE)

### 4.1. Двухфазная архитектура

```
ФАЗА 1 — Анализ сообщений (WRITE):
[Сообщение] → app/chat.py → analyze_user_profile.delay(user_id)
                                       │
                                       ▼
                              AIProfiler.analyze_profile(text)
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                  UserPersonalityProfile    AIExtractedInterests
                  OCEAN + MBTI + embed      hobbies, skills, goals
                          │
                          ▼
                  UserSchwartzProfile       UserBehaviorProfile
                  (если модуль включён)      (avg_char, reply_time, …)
                          │
                          ▼
                  update_compatibility.delay(user_id)
                          │
                          ▼
                  pgvector cosine_distance()
                  → UserCompatibility (пары)


ФАЗА 2 — Поиск и ранжирование (READ):
[Клик по узлу графа]
       │
       ▼
GET /api/graph/match/<node_id>
       │
       ├─► resolve_tags_batch()        # Теги узла → слаг иерархии
       ├─► calculate_graph_interest_score()  # Graph Score (overlap)
       ├─► compute_ocean_similarity()        # OCEAN Score
       ├─► compute_jaccard_interest_similarity()  # Jaccard Score
       ├─► compute_root_personality_score()  # Root Personality Score
       │
       ├─► комбинация: blend_weight * root_score + ...
       │    + micro_gradient_step()   # подстройка под историю пользователя
       │
       └─► возврат: [{user_id, user_name, compatibility, matched_tags, match_reason}, …]
```

### 4.2. Двухконтурная система весов

| Контур | Описание | Веса |
|--------|----------|------|
| **Глобальный (медленный)** | `GlobalWeightsConfig` — усреднённые веса всех пользователей | ocean=0.35, graph=0.40, jaccard=0.25 |
| **Персональный (быстрый)** | `User.metric_weight_*_offset` — индивидуальные смещения | базовые веса + offset |

**Микро-градиентный шаг** (`search_ranking.py`): после каждого matching-запроса веса пользователя корректируются на `learning_rate` в зависимости от того, на какие профили пользователь кликнул.

### 4.3. Генерация Match-отчёта (`app/ai/match_report.py`)

```
[Пользователь нажимает «Анализ ИИ»]
       │
       ▼
GET /api/graph/report/<target_user_id>?node_id=<node_id>
       │
       ├─► Загрузить UserPersonalityProfile (оба пользователя)
       ├─► Загрузить UserSchwartzProfile (оба)
       ├─► Загрузить UserBehaviorProfile (оба)
       ├─► Загрузить AIExtractedInterests (оба)
       │
       ├─► _build_prompt_payload()
       │      ┌─────────────────────────────────────────┐
       │      │ MBTI, OCEAN, Schwartz top-3,            │
       │      │ коммуникация, сотрудничество,            │
       │      │ поведение, цели, интересы,               │
       │      │ общие теги                               │
       │      └─────────────────────────────────────────┘
       │
       ├─► 1. YandexGPT (приоритет)
       │        _yandexgpt_generate(prompt, system_prompt)
       │        └─► API: foundationModels/v1/completion
       │
       ├─► 2. Ollama (если OLLAMA_ENABLED)
       │
       └─► 3. Статический шаблонный fallback
              build_default_match_report()
              └─► "Ваш пересекающийся интерес к сфере «X»…"
```

**System prompt для LLM:**
```
Ты — эксперт-психолог и AI-аналитик социальной платформы Nexus.
Напиши краткое (2-3 предложения), живое и максимально персонализированное
обоснование совместимости двух пользователей.
— Используй ТОЛЬКО факты из блока «ДАННЫЕ»
— Запрещены шаблонные фразы вроде «Мы подобрали тебе людей...»
— Интеллектуальный, вовлекающий, дружелюбный стиль
— Не использовать markdown, эмодзи, кавычки
```

### 4.4. Dynamic Tag Enricher (`app/ai_profiler/dynamic_enrichment.py`)

Two-Stage Semantic Routing для разрешения нечётких тегов:

```
Сырой тег (например "cs2", "я люблю музыку")
       │
       ▼
Stage 1 — Прямое совпадение
  ├─► dynamic_aliases (кэш)
  ├─► exact match по slug
  └─► fallback к Stage 2
       │
       ▼
Stage 2 — LLM-резолв
  ├─► 1. YandexGPT: сопоставить тег с иерархией
  ├─► 2. Ollama: fallback если YandexGPT недоступен
  └─► 3. Усечение: обрезать до 100 символов как slug
       │
       ▼
  Кэшировать в dynamic_aliases
```

### 4.5. Root Personality Scoring (`app/ai_profiler/root_personalities.py`)

Три корневых архетипа, определяющих базовый профиль пользователя:

| Архетип | Категория | OCEAN-профиль | Доминанта Шварц |
|---------|-----------|---------------|-----------------|
| **work** | Работа | [0.55, 0.85, 0.40, 0.55, 0.25] | achievement: 0.90, security: 0.65 |
| **entertainment** | Хобби | [0.80, 0.35, 0.70, 0.50, 0.40] | stimulation: 0.85, hedonism: 0.80 |
| **life** | Психология | [0.75, 0.50, 0.40, 0.80, 0.55] | universalism: 0.90, benevolence: 0.85 |

**Определение категории из KnowledgeNode:**
```
Если узел имеет поле category → KNOWLEDGE_CATEGORY_TO_ROOT[category]
Иначе → наследование от родительского узла по title
Иначе → fallback "work"
```

**Blend:** `root_personality_score = blend_weight * совпадение_архетипа + (1 - blend_weight) * weighted_average(ocean_sim, schwartz_sim)`

### 4.6. LLM Providers (`app/ai_profiler/providers.py`)

Абстрактная фабрика с failover-каскадом:

```
LLMProvider (abstract)
├── YandexGPTProvider    → api_key из config.YANDEX_GPT_API_KEY
├── OllamaProvider       → endpoint: http://localhost:11434
├── DeepSeekProvider     → (заготовка)
├── OpenAIProvider       → (заготовка)
└── GroqProvider         → (заготовка)

FailoverCascade:
  build_cascade(config) → [YandexGPT, Ollama] (по enabled-флагам)
  .classify(prompt)     → перебирает провайдеров, возвращает первый успешный ответ
```

---

## 5. ГРАФ ЗНАНИЙ И ИЕРАРХИЯ ИНТЕРЕСОВ

### 5.1. Три слоя интересов

| Слой | Таблица | Источник | UI |
|------|---------|----------|-----|
| **Визуальный граф** | `knowledge_nodes` + `knowledge_connections` | Seed / ручное создание | Canvas с узлами (SVG + D3.js) |
| **Иерархия интересов** | `interest_hierarchy_nodes` | Seed (tools/seed_db.py) | Невидим, основа для скоринга |
| **AI-извлечение** | `ai_extracted_interests` | AIProfiler.extract_interests() | Профиль пользователя |

### 5.2. KnowledgeNode

```python
KnowledgeNode:
  id, user_id, title, description
  category: str    # "work" | "hobby" | "psychology" | "want"
  x, y: float      # позиция на визуальном графе
```

**category → root_category** (через KNOWLEDGE_CATEGORY_TO_ROOT в root_personalities.py):
- `work` → `"work"`
- `hobby` → `"entertainment"`
- `psychology` → `"life"`
- `want` → `"work"` (fallback)

### 5.3. Иерархия (InterestHierarchyNode)

Материализованный путь (`path = "/1/5/12/"`), embedding Vector(384) для семантического поиска:

```
Корень (/)
├── it_development (/1/)
│   ├── programming (/1/5/)
│   │   ├── python (/1/5/12/)
│   │   └── javascript (/1/5/13/)
│   ├── mobile_dev (/1/6/)
│   └── devops (/1/7/)
├── design (/2/)
├── sports (/3/)
└── psychology (/4/)
```

### 5.4. Метрики скоринга

| Метрика | Функция | Диапазон | Описание |
|---------|---------|----------|----------|
| **Graph Score** | `calculate_graph_interest_score()` | [0, 1] | Пересечение весов в иерархии |
| **OCEAN** | `compute_ocean_similarity()` | [0, 1] | Косинусная близость OCEAN-векторов + комплементарность |
| **Jaccard** | `compute_jaccard_interest_similarity()` | [0, 1] | Jaccard по AIExtractedInterests |
| **Schwartz** | `compute_schwartz_similarity()` | [0, 1] | Косинус по 10 ценностям |
| **Root Personality** | `compute_root_personality_score()` | [0, 1] | Близость к корневому архетипу |

---

## 6. UI / FRONTEND

### 6.1. Главная страница (`templates/index.html`)

```
┌──────────────────────────────────────────────────────┐
│  ┌────────────────────────────┐  ┌──────────────────┐│
│  │   Граф знаний (SVG+D3)    │  │  Подходящие люди  ││
│  │   Интерактивный Canvas     │  │                   ││
│  │   drag/zoom/edit           │  │  [88%] Иван       ││
│  │                            │  │       [Проф][AI]  ││
│  │   [Создать узел]           │  │  ┌──тег1 +ещё 2──┐││
│  │                            │  │  │               ││
│  └────────────────────────────┘  │  [92%] Мария     ││
│                                   │       [Проф][AI]  ││
│                                   └──────────────────┘│
│                                                       │
│  ┌── AI Review Toast (bottom-left) ──────────────┐   │
│  │ ✨ Nexus AI — Анализ                     [✕]  │   │
│  │ Ваш пересекающийся интерес к сфере ...        │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

**AI Review Toast:** фиксированный тост в левом нижнем углу. Появляется при нажатии «Анализ ИИ» в карточке кандидата. Не блокирует просмотр списка. Содержит typewriter-анимацию текста. Авто-закрывается при клике на новый узел графа.

### 6.2. Карточка кандидата

```
┌─────────────────────────────────────────────┐
│  [88%]  Иван                    [Проф][AI]   │
│  ┌─── программирование ─── +ещё 2 ─────────┐│
│  │  python, django, backend                 ││
│  └──────────────────────────────────────────┘│
└─────────────────────────────────────────────┘
```

- Первый тег всегда виден
- Остальные скрыты под `+ещё N`, раскрываются по клику

### 6.3. Профили

**Свой профиль** (`templates/profile.html`):
- OCEAN radar chart (Chart.js)
- MBTI, стиль общения, сотрудничество, формальность, энтузиазм
- Schwartz value bars (топ-5)
- AI-извлечённые интересы (хобби, навыки, цели)

**Чужой профиль** (`templates/view_profile.html`):
- Аватар, имя, био, контакты
- Психологическая карточка: MBTI, OCEAN (топ-2 черты), Schwartz bars (топ-5)
- AI-интересы: хобби, навыки, темы, цели, профессия, стиль работы

### 6.4. CSS Architecture

Единая система CSS-переменных в `static/src/styles/variables.css`:

```
--color-bg              # Фон страницы
--color-bg-deep         # Тёмный фон графа
--color-panel           # Панели
--color-card            # Карточки кандидатов
--color-text            # Основной текст
--color-text-heading    # Заголовки (яркий белый)
--color-text-soft       # Второстепенный текст
--color-accent-purple   # Акцент графа
--color-success         # Зелёный (теги)
--color-danger          # Красный (удаление, ошибки)
--radius-md / --radius-lg / --radius-pill
--space-1 … --space-8
--shadow-graph / --shadow-toast
```

---

## 7. КОНФИГУРАЦИЯ

### 7.1. Критичные env-переменные

```bash
# База данных
DATABASE_URL=postgresql://my_app_user:pass@127.0.0.1:5432/nexus_db

# Redis
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
REDIS_CACHE_URL=redis://localhost:6379/1

# Безопасность
SECRET_KEY=your-secret-key-here

# ML-модели
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
LOCAL_ARTIFACTS_DIR=./ml/artifacts
PERSONALITY_MODEL_FILENAME=personality_model_best.pth
MBTI_MODEL_FILENAME=mbti_model.pth
MBTI_NEURAL_BLEND_WEIGHT=0.7

# LLM (YandexGPT)
YANDEX_GPT_API_KEY=your-api-key
YANDEX_GPT_API_BASE=https://llm.api.cloud.yandex.net
YANDEX_GPT_MODEL=gpt://b1gd7uvpjf1qlla85o97/yandexgpt-5.1/latest

# Ollama (опционально)
OLLAMA_ENABLED=False
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# AI-анализ
MIN_MESSAGES_FOR_ANALYSIS=3        # dev
ANALYSIS_MESSAGE_TRIGGER=5          # dev
ANALYSIS_COOLDOWN=0                 # dev

# Метрики
ROOT_PERSONALITY_BLEND_WEIGHT=0.3
PERSONALITY_OCEAN_WEIGHT=0.6
PERSONALITY_SCHWARTZ_WEIGHT=0.4
```

### 7.2. Профили конфигурации

| Класс | FLASK_ENV | Особенности |
|-------|-----------|-------------|
| `DevelopmentConfig` | development | DEBUG=True, MIN_MESSAGES=3, cooldown=0 |
| `ProductionConfig` | production | Secure cookies, строгие лимиты |
| `TestingConfig` | testing | SQLite in-memory |

---

## 8. CELERY-ЗАДАЧИ

| Имя задачи | Функция | Триггер | Описание |
|-----------|---------|---------|----------|
| `ai.analyze_user_profile` | `analyze_user_profile()` | Каждое новое сообщение | Запускает полный AI-пайплайн для пользователя |
| `ai.update_compatibility` | `update_compatibility()` | После analyze_user_profile | Пересчитывает pgvector-совместимость для всех пар |

**Запуск worker:**
```bash
celery -A app.ai.personality_analyzer worker --loglevel=info
```

**Seed-режим (синхронный, без Celery):**
```python
analyze_user_profile.apply(args=[u_id], kwargs={"force": True}).result
```

---

## 9. СИДИНГ БД (`tools/seed_db.py`)

**Назначение:** наполнение PostgreSQL тестовыми данными + полный прогон AI-пайплайна.

**Параметры:**
- 200 пользователей
- 16 архетипов (из ROOT_ARCHETYPES + комбинации)
- 5 OCEAN-кластеров
- Schwartz-профили (на основе архетипов)
- pgvector эмбеддинги
- Шум к весам графа (±0.1)
- Knowledge nodes (3 на пользователя)
- Чаты + сообщения

**Алгоритм:**
```
FOR each user:
  1. INSERT users
  2. INSERT knowledge_nodes (3 random tags)
  3. INSERT knowledge_connections
  4. INSERT chats (с пользователем #1)
  5. INSERT messages (архетипные тексты)
  6. analyze_user_profile SYNC → OCEAN + MBTI + интересы
```

---

## 10. ТЕСТЫ

### test_metrics.py — 55 тестов

| Группа | Количество | Описание |
|--------|-----------|----------|
| Graph Score | 5 | calculate_graph_interest_score, hierarchical overlap |
| Schwartz | 8 | compute_schwartz_similarity, cosine |
| Root Personality | 10 | compute_root_personality_score, find_root_category |
| Jaccard | 5 | compute_jaccard_interest_similarity |
| OCEAN | 5 | compute_ocean_similarity |
| Mapping integrity | 12 | KNOWLEDGE_CATEGORY_TO_ROOT, slug → archetype |
| Edge cases | 10 | Пустые профили, None, граничные значения |

**Запуск:**
```bash
pytest test_metrics.py -v
```

---

## 11. ИЗВЕСТНЫЕ АРХИТЕКТУРНЫЕ ПРОБЛЕМЫ

| Проблема | Файлы | Описание |
|----------|-------|----------|
| Рассинхронизация SEMANTIC_ONTOLOGY и InterestHierarchyNode | `semantic_ontology.py`, `interest_graph.py` | Канонические слаги не совпадают, graph_score = 0 для пользователей без precomputed weights |
| resolve_tag_to_slug хрупок | `dynamic_enrichment.py` | Нет fuzzy/semantic fallback для опечаток и синонимов |
| Двойная система совместимости | `personality_analyzer.py`, `profile.py` | pgvector vs Python-SBERT — результаты могут расходиться |
| register_user_tags не покрывает всех | `interest_graph.py`, `seed_db.py` | graph_score часто = 0 для не-seeded пользователей |
| Нет rate-limiting на AI-анализ | `personality_analyzer.py` | Каждое сообщение триггерит анализ (потенциально дорого) |

---

## 12. QUICK REFERENCE ДЛЯ РАЗРАБОТЧИКА

1. **Канонические ORM-модели AI** — `data/ai.py`, не `app/ai/models.py`
2. **`embedding` = OCEAN[5]**, не SBERT; размерность Vector(5)
3. **Celery-задачи** в `app/ai/personality_analyzer.py`; `celery_tasks.py` — re-export
4. **`get_profiler()`** — единственный способ получить `AIProfiler` (singleton)
5. **Очистка текста** — через `clean_user_text()` из `text_utils.py`
6. **Конфиг** — только `config.py`; не дублировать чтение env
7. **При добавлении новой модели** — регистрировать в `data/__all_models.py`
8. **CSS-переменные** — только в `variables.css`; не хардкодить цвета
9. **AI-ревью** — приоритет: YandexGPT → Ollama → шаблонный fallback
10. **Новый blueprint** — регистрировать в `app/__init__.py`

---

*Версия документа: 2.0. Последнее обновление: июль 2026.*
