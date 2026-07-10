# Nexus — Project Architecture & AI Context

> **Назначение документа:** единый контекстный файл для LLM-ассистентов. После прочтения модель должна понимать архитектуру, потоки данных, схему БД и точки расширения без повторного обхода репозитория.

---

## 1. ОБЗОР ПРОЕКТА И СТЕК

### 1.1. Суть проекта

**Nexus** — социальная платформа, объединяющая три слоя интеллекта:

| Слой | Назначение |
|------|------------|
| **Психологический профиль** | Автоматический анализ личности (OCEAN / Big Five + MBTI) из текстовых сообщений пользователя |
| **Граф знаний интересов** | Визуальный персональный граф (`knowledge_nodes`, `knowledge_connections`) + JSON-извлечённые интересы (`AIExtractedInterests`) |
| **Векторная совместимость** | Поиск ближайших «психологических соседей» через pgvector (косинусное расстояние по 5-мерному OCEAN-вектору) |

Пользователь общается в чатах → тексты накапливаются → Celery-воркер запускает ML-анализ → результаты сохраняются в PostgreSQL → совместимость пересчитывается на уровне СУБД → фронтенд показывает рекомендации, граф интересов и профиль.

### 1.2. Технологический стек

```
┌─────────────────────────────────────────────────────────────┐
│  Клиент (HTML/JS templates + Socket.IO)                     │
├─────────────────────────────────────────────────────────────┤
│  Flask 3.x  │  Flask-Login  │  Flask-SocketIO  │  Flask-CORS│
│  Blueprints: auth, profile, interests, chat, moderation,    │
│              analytics, routes (main)                         │
├─────────────────────────────────────────────────────────────┤
│  Celery 5.x  ←→  Redis (broker + result backend)            │
│  Задачи: ai.analyze_user_profile, ai.update_compatibility  │
├─────────────────────────────────────────────────────────────┤
│  PyTorch 2.x + SentenceTransformers + Transformers          │
│  Модели: PersonalityClassifier (OCEAN), MBTIClassifier       │
├─────────────────────────────────────────────────────────────┤
│  SQLAlchemy 2.x  │  Alembic / Flask-Migrate                 │
│  PostgreSQL 16 + pgvector 0.2.x  (Docker: pgvector/pgvector) │
└─────────────────────────────────────────────────────────────┘
```

**Python:** 3.12+  
**Точка входа:** `App.py` → `create_app()` из `app/__init__.py` → `socketio.run(app)`  
**Конфигурация:** единый модуль `config.py` (корень проекта). Все env-переменные читаются только там; остальные модули импортируют `config` или `get_config()`.

**Ключевые зависимости** (`docs/requirements.txt`):
- `flask`, `flask-login`, `flask-socketio`, `flask-migrate`, `flask-cors`
- `celery`, `redis`
- `sqlalchemy`, `psycopg2-binary`, `pgvector`
- `torch`, `sentence-transformers`, `transformers`
- `scikit-learn`, `numpy`, `pandas`, `nltk`

**Инфраструктура (Docker):**
```yaml
# docker-compose.yml
services:
  db:
    image: pgvector/pgvector:pg16
    ports: ["5432:5432"]
    environment:
      POSTGRES_USER: my_app_user
      POSTGRES_DB: nexus_db
```

**Redis** (не в compose, ожидается локально): broker Celery `redis://localhost:6379/0`, кэш `redis://localhost:6379/1`.

---

## 2. АРХИТЕКТУРА И СТРУКТУРА ДАННЫХ

### 2.1. Карта директорий

```
Nexus/
├── App.py                  # Точка входа WSGI / dev-сервер
├── config.py               # Единая конфигурация (Config, DevelopmentConfig, …)
├── app/                    # Flask-приложение
│   ├── __init__.py         # create_app(), регистрация blueprints
│   ├── extensions.py       # login_manager, socketio, cors, migrate
│   ├── auth.py             # Регистрация / авторизация
│   ├── profile.py          # Профиль, аватар, API графа знаний, matching
│   ├── chat.py             # REST + Socket.IO чаты, триггер AI-анализа
│   ├── interests.py        # Лента интересов, API matching по узлам графа
│   ├── moderation.py       # Модерация, жалобы
│   ├── analytics.py        # Дашборд модератора
│   ├── db.py               # get_db_session() — context manager для сессий
│   ├── ai/
│   │   ├── personality_analyzer.py  # ★ Celery-задачи (ядро пайплайна)
│   │   ├── celery_tasks.py          # Re-export задач
│   │   ├── data_processor.py        # TF-IDF препроцессор (legacy/обучение)
│   │   ├── models.py                # ConversationAnalysis, TrainingMetrics
│   │   └── profiler_singleton.py    # Re-export get_profiler()
│   └── ai_profiler/
│       ├── core.py                  # ★ AIProfiler, PersonalityClassifier, MBTIClassifier
│       ├── text_utils.py            # clean_user_text() — единая очистка текста
│       ├── contextual_adapter.py    # ContextualAdapter — enrichment, SBERT wrapper (app/ai_profiler/contextual_adapter.py)
│       ├── interest_graph.py       # Interest graph: resolve_tag_to_slug, build_query_weights, calculate_graph_interest_score
│       ├── interest_extractor.py   # CustomInterestClassifier, Neural/ZeroShot extractors, label builders
│       ├── semantic_ontology.py    # SEMANTIC_ONTOLOGY, FINETUNE_POSITIVE_PAIRS — canonical aliases for tag->slug
│       ├── taxonomy.py             # INTEREST_TAXONOMY — anchors for zero-shot extractor
│       ├── schwartz_analyzer.py    # (optional) value/schwartz scoring module
│       ├── behavior_analyzer.py    # (optional) behavior -> compatibility signals
│       └── __init__.py            # get_profiler() — thread-safe singleton
├── data/                   # SQLAlchemy ORM-модели (единый слой данных)
│   ├── session.py          # SqlAlchemyBase, global_init(), create_session()
│   ├── __all_models.py     # Импорт всех моделей для Alembic/metadata
│   ├── user.py, chat.py, message.py, interest.py, …
│   ├── ai.py                    # ★ UserPersonalityProfile, AIExtractedInterests, UserCompatibility
│   ├── interest_hierarchy.py    # InterestHierarchyNode, UserInterestGraphWeight — граф интересов и per-user weights
│   ├── behavior.py              # UserBehaviorProfile — агрегаты поведения (если есть)
│   └── knowledge_graph.py       # KnowledgeNode, KnowledgeConnection
├── ml/                     # Обучение моделей
│   ├── train_ocean.py      # Обучение PersonalityClassifier
│   ├── train_mbti.py       # Обучение MBTIClassifier
│   ├── preprocess.py       # Подготовка датасета, SBERT-эмбеддинги
│   └── artifacts/          # *.pth веса, training_history.png
├── tools/
│   ├── seed_db.py          # ★ Наполнение БД + синхронный AI-анализ
│   ├── watch_profile.py    # Отладка профилей
│   └── chat.py             # CLI-утилита чата
├── migrations/             # Alembic (в т.ч. pgvector)
└── templates/, static/       # Frontend
```

### 2.2. ER-схема ключевых сущностей

```mermaid
erDiagram
    User ||--o{ Message : "author"
    User ||--o| UserPersonalityProfile : "has"
    User ||--o| AIExtractedInterests : "has"
    User ||--o{ KnowledgeNode : "owns"
    User ||--o{ UserCompatibility : "user_id_1"
    User ||--o{ UserCompatibility : "user_id_2"
    Chat ||--o{ Message : "contains"
    User ||--o{ Chat : "user1_id"
    User ||--o{ Chat : "user2_id"
    KnowledgeNode ||--o{ KnowledgeConnection : "from_node"
    KnowledgeNode ||--o{ KnowledgeConnection : "to_node"

    UserPersonalityProfile {
        int user_id PK_FK
        float openness
        float conscientiousness
        float extraversion
        float agreeableness
        float neuroticism
        vector embedding "Vector(5)"
        string mbti_type
        json traits
        json values
    }

    AIExtractedInterests {
        int user_id PK_FK
        json hobbies
        json topics
        json skills
        json preferences
        json short_term_goals
    }

    UserCompatibility {
        int user_id_1 FK
        int user_id_2 FK
        float overall_score
        json recommendations
    }
```

### 2.3. Модуль `data/` — ORM-слой

**Принцип:** все таблицы описаны в `data/*.py`, наследуют `SqlAlchemyBase` из `data/session.py`. Импорт всех моделей — через `data/__all_models.py` (нужен для `create_all` и Alembic).

| Файл | Таблицы | Роль |
|------|---------|------|
| `user.py` | `users` | Аутентификация (UserMixin), связи с профилем, сообщениями, совместимостью |
| `chat.py` | `chats` | Диалог 1:1 (`user1_id`, `user2_id`) |
| `message.py` | `messages` | Текстовые/файловые сообщения; **источник данных для AI** |
| `interest.py` | `interests` | Пользовательские публикации интересов (социальная лента) |
| `knowledge_graph.py` | `knowledge_nodes`, `knowledge_connections` | Узлы и рёбра персонального графа (координаты x,y для UI) |
| `ai.py` | `ai_user_personality_profiles`, `ai_extracted_interests`, `ai_user_compatibility` | **Ядро AI-данных** |
| `favorite_interest.py` | `favorite_interests` | M2M избранное |
| `report.py` | `reports` | Жалобы модерации |
| `chat_settings.py` | `chat_settings` | Настройки чатов |
| `session.py` | — | Фабрика сессий, `global_init(DATABASE_URL)` |

**Важно для LLM:** `app/ai/models.py` содержит только legacy-таблицы (`ai_conversation_analysis`, `ai_training_metrics`) и re-export из `data.ai`. Канонические AI-модели — **только в `data/ai.py`**.

### 2.4. Модуль `app/` — веб-слой

Blueprints регистрируются в `create_app()`:

```python
# app/__init__.py (упрощённо)
app.register_blueprint(main_bp)      # routes.py — стартовая страница
app.register_blueprint(auth_bp)
app.register_blueprint(profile_bp)
app.register_blueprint(interests_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(moderation_bp)
app.register_blueprint(analytics_bp)
```

**Триггер AI-анализа** — в `app/chat.py`, после сохранения сообщения:
```python
analyze_user_profile.delay(author_id)  # асинхронно через Celery
```

**Сессии БД в HTTP:** `app/db.py` → `get_db_session()` (context manager).  
**Сессии БД в Celery:** `DBTask` в `personality_analyzer.py` создаёт сессию на каждый вызов задачи.

### 2.5. Модуль `ml/` — обучение

| Скрипт | Назначение |
|--------|------------|
| `preprocess.py` | Корпус → SBERT-эмбеддинги + ручные признаки → JSON/PT датасет |
| `train_ocean.py` | Обучение `PersonalityClassifier`, метрики R²/MAE, сохранение в `ml/artifacts/personality_model_best.pth` |
| `train_mbti.py` | Обучение `MBTIClassifier` → `ml/artifacts/mbti_model.pth` |
| `precompute_embedding.py` | Предвычисление эмбеддингов |
| `test_model.py` | Валидация модели |

**Артефакты** (`config.PERSONALITY_MODEL_PATH`, `config.MBTI_MODEL_PATH`):
- `personality_model_best.pth` — веса OCEAN-классификатора
- `mbti_model.pth` — веса MBTI-классификатора

**Метрики (из README):** R² ≈ 0.68, MAE ≈ 0.09, датасет ~1726 примеров.

---

## 3. ДЕТАЛЬНОЕ ОПИСАНИЕ ИИ-КОНВЕЙЕРА (AI PIPELINE)

### 3.1. Общая схема потока данных

```
[Пользователь пишет сообщение]
         │
         ▼
app/chat.py ──POST /messages──► INSERT messages (data/message.py)
         │
         ▼
analyze_user_profile.delay(user_id)     ← Celery + Redis
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Celery Task: ai.analyze_user_profile                      │
│  (app/ai/personality_analyzer.py)                          │
│                                                            │
│  1. SELECT messages WHERE author_id = user_id              │
│     ORDER BY timestamp DESC LIMIT MAX_MESSAGES (50)        │
│  2. full_text = join(message.content)                      │
│  3. profiler = get_profiler()  → AIProfiler singleton      │
│  4. analysis = profiler.analyze_profile(full_text)         │
│  5. UPSERT UserPersonalityProfile + AIExtractedInterests   │
│  6. profile.embedding = ocean  (Vector(5))                 │
│  7. COMMIT                                                 │
│  8. update_compatibility.delay(user_id)                    │
└────────────────────────────────────────────────────────────┘


# Интеграционные заметки (кратко):
- ContextualAdapter (app/ai_profiler/contextual_adapter.py) инициализируется в AIProfiler и отдаёт sbert_model + методы enrich_text/prepare_for_encoding; используется для обогащения поисковых запросов и подготовки якорей для ZeroShotInterestExtractor.
- matching_engine (app/ai/matching_engine.py) собирает компоненты совместимости: graph_score (вызывая app.ai_profiler.interest_graph.calculate_graph_interest_score), schwartz и поведение; profile.py endpoints используют calculate_multidimensional_compatibility/graph scoring и дополняют pgvector-результатами.
- register_user_tags() вызывается в profile.py перед матчингом для регистрации/записи query-тегов в UserInterestGraphWeight, чтобы обеспечить наличие per-user weights для скоринга.
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Celery Task: ai.update_compatibility                      │
│  pgvector cosine_distance() на уровне PostgreSQL           │
│  → UPSERT ai_user_compatibility                            │
└────────────────────────────────────────────────────────────┘
```

### 3.2. Задача `analyze_user_profile`

**Файл:** `app/ai/personality_analyzer.py`  
**Имя Celery:** `ai.analyze_user_profile`  
**Базовый класс:** `DBTask` — автоматическое создание/закрытие SQLAlchemy-сессии.

**Алгоритм:**

1. **Загрузка пользователя** — `User` по `user_id`; выход с `{"error": "User not found"}` если нет.
2. **Сбор корпуса** — последние `config.MAX_MESSAGES_PER_ANALYSIS` (по умолчанию 50) сообщений автора:
   ```python
   messages = db.query(Message).filter_by(author_id=user_id)
       .order_by(Message.timestamp.desc()).limit(50).all()
   full_text = " ".join([m.content for m in messages if m.content])
   ```
3. **Проверка** — пустой `full_text` → `{"error": "No text data for analysis"}`.
4. **Инференс** — `get_profiler().analyze_profile(full_text)`.
5. **Запись `UserPersonalityProfile`:**
   - Скalarные колонки OCEAN: `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`
   - **`embedding = ocean`** — список из 5 float `[O, C, E, A, N]` напрямую в колонку `Vector(5)`
   - MBTI, стиль коммуникации, traits, values, confidence_score, last_analyzed
6. **Запись `AIExtractedInterests`** — hobbies, topics, skills, dislikes, goals, preferences из `analysis["interests"]`.
7. **`db.commit()`**
8. **Цепочка:** `update_compatibility.delay(user_id)`

**Конфигурационные пороги** (`config.py`):
- `MIN_MESSAGES_FOR_ANALYSIS` — минимум сообщений (dev: 3, prod: 10)
- `ANALYSIS_MESSAGE_TRIGGER` — триггер по количеству (dev: 5, prod: 15)
- `ANALYSIS_COOLDOWN` — cooldown между анализами (сек)
- `MAX_MESSAGES_PER_ANALYSIS` — 50

> **Примечание:** текущая реализация вызывает анализ на **каждое** новое сообщение (`force=True` по умолчанию). Rate-limiting на уровне конфига (`AI_ANALYSIS_RATE_LIMIT`) определён, но в задаче не проверяется явно — потенциальная точка оптимизации.

### 3.3. Класс `AIProfiler` (`app/ai_profiler/core.py`)

**Singleton:** `app/ai_profiler/__init__.py` → `get_profiler()` с double-checked locking. Модели загружаются один раз на процесс Celery-воркера.

#### 3.3.1. Загружаемые модели

| Компонент | Модель / архитектура | Вход | Выход |
|-----------|---------------------|------|-------|
| **SBERT** | `SentenceTransformer(config.EMBEDDING_MODEL)` — default: `paraphrase-multilingual-MiniLM-L12-v2` | текст | вектор 384 dim |
| **PersonalityClassifier** | PyTorch: Linear→ResidualBlock→Linear, ordinal 5×10 bins | 388 dim | 5 OCEAN scores [0,1] |
| **MBTIClassifier** | PyTorch MLP 384→256→128→16 | 384 dim (только SBERT) | 16 классов MBTI |

**Путь к весам:** `config.PERSONALITY_MODEL_PATH`, `config.MBTI_MODEL_PATH` → `ml/artifacts/*.pth`.

#### 3.3.2. Метод `analyze_profile(text)` — пошагово

```
text
  │
  ├─► clean_user_text()          # app/ai_profiler/text_utils.py
  │     • удаление URL, null-bytes, лишних символов
  │     • лимит 200 000 символов
  │
  ├─► get_manual_features()      # 4 признака:
  │     [длина/1000, доля CAPS, count(!)/5, count(?)/5]
  │
  ├─► bert_model.encode()        # SBERT → tensor [1, 384]
  │
  ├─► torch.cat([emb, manual])   # combined_input [1, 388]
  │
  ├─► PersonalityClassifier.predict_scores()
  │     • ordinal softmax → weighted bin centers
  │     • clip [0.05, 0.95]
  │     → ocean: [O, C, E, A, N]
  │
  ├─► MBTI (Neural Blend):
  │     • mbti_model(emb) → softmax → mbti_probs (если .pth есть)
  │     • infer_mbti(ocean) → rule-based тип + _ocean_to_soft_probs()
  │     • blended = w * mbti_probs + (1-w) * ocean_soft
  │       w = config.MBTI_NEURAL_BLEND_WEIGHT (default 0.7)
  │     → mbti_type (argmax)
  │
  ├─► infer_communication_style()  # formality, enthusiasm, detail_oriented
  ├─► extract_interests()          # rule-based: hobbies, topics, skills, …
  └─► return dict {ocean, mbti_type, communication, interests, …}
```

#### 3.3.3. OCEAN — семантика шкал

Пять **независимых непрерывных** значений в диапазоне **[0.0, 1.0]**:

| Индекс | Признак | Интерпретация |
|--------|---------|---------------|
| 0 | Openness (O) | Открытость новому опыту |
| 1 | Conscientiousness (C) | Добросовестность, организованность |
| 2 | Extraversion (E) | Экстраверсия |
| 3 | Agreeableness (A) | Доброжелательность |
| 4 | Neuroticism (N) | Нейротизм / эмоциональная нестабильность |

**Архитектура предсказания:** не регрессия напрямую, а **ordinal classification** — 10 бинов на каждый признак, затем взвешенное математическое ожидание (`bins_to_score`).

#### 3.3.4. MBTI — семантика

- **Категориальный** признак: одна из 16 строк (`INTJ`, `ENFP`, …).
- **Два источника:**
  1. Нейросеть `MBTIClassifier` на SBERT-эмбеддинге (384 dim)
  2. Rule-based `infer_mbti(ocean)` — пороги по OCEAN (E/I ← extraversion, N/S ← openness, T/F ← agreeableness, J/P ← conscientiousness; neuroticism > 0.75 → принудительно I)
- **Слияние:** `MBTI_NEURAL_BLEND_WEIGHT` (default 0.7 в пользу нейросети).

#### 3.3.5. Упаковка OCEAN в `embedding`

```python
# personality_analyzer.py, строки 116-121
profile.openness = ocean[0]
profile.conscientiousness = ocean[1]
# …
profile.embedding = ocean  # list[float] длиной 5 → pgvector Vector(5)
```

**Критически важно:** `embedding` — это **не SBERT-вектор**. Это прямое отображение 5 OCEAN-скоров в pgvector-столбец для ANN-поиска совместимости. SBERT используется только на этапе инференса и в endpoint'ах семантического matching по интересам.

### 3.4. Запуск Celery

```bash
# Worker (из корня проекта)
celery -A app.ai.personality_analyzer worker --loglevel=info

# Или через re-export:
celery -A app.ai.celery_tasks worker --loglevel=info
```

При сидинге (`tools/seed_db.py`) worker **не нужен** — используется синхронный вызов:
```python
analyze_user_profile.apply(args=[u_id], kwargs={"force": True}).result
```

---

## 4. ИНФРАСТРУКТУРА РАБОТЫ С ВЕКТОРАМИ (pgvector & HNSW)

### 4.1. Расширение pgvector

**Миграция:** `migrations/versions/0acf81f89c90_add_pgvector.py`

```python
op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

op.add_column('ai_user_personality_profiles',
    sa.Column('embedding', Vector(5), nullable=True))

op.create_index(
    'idx_user_personality_embedding_hnsw',
    'ai_user_personality_profiles',
    ['embedding'],
    postgresql_using='hnsw',
    postgresql_with={'m': 16, 'ef_construction': 64},
    postgresql_ops={'embedding': 'vector_cosine_ops'}
)
```

**ORM-тип** (`data/ai.py`):
```python
from pgvector.sqlalchemy import Vector
embedding = Column(Vector(5), nullable=True)
```

### 4.2. HNSW-индекс

| Параметр | Значение | Назначение |
|----------|----------|------------|
| Имя | `idx_user_personality_embedding_hnsw` | — |
| Алгоритм | HNSW | Approximate Nearest Neighbor |
| Метрика | `vector_cosine_ops` | Косинусное расстояние |
| `m` | 16 | Макс. связей на узел графа индекса |
| `ef_construction` | 64 | Точность построения индекса |

HNSW обеспечивает сублинейный поиск ближайших профилей при росте числа пользователей.

### 4.3. Архитектурный сдвиг: совместимость в СУБД

**Было (legacy):** попарный расчёт в Python через `AIProfiler.calculate_compatibility()` — взвешенное еuclidean-расстояние между OCEAN-векторами, матричные операции NumPy.

**Стало (Celery-пайплайн):** расчёт полностью в SQL через `pgvector.sqlalchemy`:

```python
# app/ai/personality_analyzer.py — update_compatibility
similar_profiles = db.query(
    UserPersonalityProfile.user_id,
    UserPersonalityProfile.mbti_type,
    UserPersonalityProfile.embedding.cosine_distance(my.embedding).label("distance")
).filter(
    UserPersonalityProfile.user_id != user_id,
    UserPersonalityProfile.embedding.isnot(None)
).all()
```

**Семантика `cosine_distance`:**
- Диапазон: **[0, 2]**
- 0 = идентичные векторы
- 2 = противоположные направления

**Нормализация в `overall_score` [0.0, 1.0]:**
```python
score_normalized = max(0.0, 1.0 - (float(distance) / 2.0))
```

**Запись в `ai_user_compatibility`:**
```python
compat.overall_score = score_normalized
compat.romantic_score = score_normalized      # пока = overall
compat.professional_score = score_normalized  # пока = overall
compat.creative_score = score_normalized      # пока = overall
compat.interest_overlap = score_normalized    # пока = overall
compat.recommendations = {
    "summary": "Compatibility calculated via pgvector cosine distance",
    "mbti_pair": [my.mbti_type, other.mbti_type],
}
```

**Upsert-логика:** предзагрузка существующих пар `(user_id, other_id)` в `compat_map`, обновление или создание новых `UserCompatibility`.

### 4.4. Двойная система расчёта совместимости (важно для LLM)

| Контекст | Механизм | Файл |
|----------|----------|------|
| **Фоновый пересчёт всех пар** | pgvector `cosine_distance` в SQL | `personality_analyzer.update_compatibility` |
| **HTTP API matching по графу** | Python `AIProfiler.calculate_compatibility()` + SBERT similarity | `app/profile.py`, `app/interests.py` |

Endpoint `/api/graph/match/<node_id>` (profile.py) комбинирует:
- 70% — косинусное сходство SBERT между заголовками узлов графа
- 30% — взвешенное euclidean-сходство OCEAN с учётом комплементарности (`complementary` indices)

**Это не баг документации** — два пути существуют параллельно. При рефакторинге следует унифицировать.

### 4.5. Тест pgvector

`test_pgvector.py` — интеграционный тест:
1. Создаёт двух пользователей с близкими OCEAN-векторами
2. Записывает `embedding` напрямую
3. Выполняет `.cosine_distance()` через SQLAlchemy
4. Сверяет с NumPy (`1 - cos_sim`)

---

## 5. ГРАФ ЗНАНИЙ И ДОПОЛНИТЕЛЬНЫЕ МОДУЛИ

### 5.1. Два слоя «интересов»

| Слой | Таблица | Источник данных | UI |
|------|---------|-----------------|-----|
| **Социальная лента** | `interests` | Пользователь публикует вручную | Лента, избранное |
| **AI-извлечение** | `ai_extracted_interests` | `AIProfiler.extract_interests()` | JSON для персонализации |
| **Визуальный граф** | `knowledge_nodes`, `knowledge_connections` | Seed / ручное создание | Canvas с узлами (x, y) |

#### `AIExtractedInterests` (`data/ai.py`)

```python
hobbies          # JSON list — «спорт», «coding», …
topics           # JSON list — «AI», «психология», …
skills           # JSON list — «python», «дизайн», …
dislikes         # JSON list
occupation       # string | null
work_style       # text | null
short_term_goals # JSON list
long_term_goals  # JSON list
preferences      # JSON dict — language_mix, message_length, …
last_extraction  # datetime
```

Rule-based извлечение в `AIProfiler.extract_interests()` — словари ключевых слов (RU+EN), частотный анализ токенов.

#### `KnowledgeNode` / `KnowledgeConnection`

```python
KnowledgeNode: user_id, title, description, category, x, y
KnowledgeConnection: from_node_id, to_node_id, label
```

Frontend строит интерактивный граф; API matching ищет пользователей с похожими узлами.

### 5.2. Чат-система

**Модель `Chat`:** два участника (`user1_id`, `user2_id`), без group-chat.

**Модель `Message`:**
```python
chat_id, author_id, content, file_url, timestamp,
message_type  # "text", …
reply_to_id   # self-FK для тредов
```

**Транспорт:**
- REST API в `app/chat.py` (CRUD сообщений)
- **Flask-SocketIO** для real-time (`app/extensions.py` → `socketio`)

**Связь с AI:** каждое сохранённое сообщение → `analyze_user_profile.delay(author_id)`.

### 5.3. Сидинг БД (`tools/seed_db.py`)

**Назначение:** наполнение пустой PostgreSQL реалистичными данными + полный прогон AI-пайплайна.

**Архетипы пользователей:**
| Архетип | Категория | Характер сообщений |
|---------|-----------|-------------------|
| Профи | work | Планирование, кодстайл, рефакторинг |
| Душа компании | psychology | Митапы, эмоции, командный дух |
| Творец | hobby | UI/UX, Rust, Figma |
| Скейтер/Геймер | hobby | CS2, Steam, скины |

**Алгоритм `run_mega_seed(num_users=10)`:**
```
FOR each archetype user:
  1. INSERT users (psycopg2 raw SQL)
  2. INSERT knowledge_nodes (3 random tags from TOPICS)
  3. INSERT chats (user_id=1 ↔ new_user)  # чат с владельцем
  4. INSERT messages (4 msgs per archetype)
  5. analyze_user_profile.apply([u_id]).result  # SYNC Celery
     → OCEAN + MBTI + interests + embedding
     → update_compatibility (async chain)
```

**Требования:** `SEED_DB_PASSWORD` в env, PostgreSQL доступен, ML-артефакты на месте, SBERT скачается при первом запуске.

### 5.4. Прочие модули

| Модуль | Назначение |
|--------|------------|
| `app/auth.py` | Регистрация, логин, Flask-Login sessions |
| `app/moderation.py` | Жалобы (`Report`), действия модератора |
| `app/analytics.py` | Метрики платформы (только `is_moderator`) |
| `app/debug_bp.py` | Отладочные endpoint'ы |
| `tools/watch_profile.py` | CLI-мониторинг изменений профиля |
| `AI/config.py` | Legacy-конфиг (deprecated, использовать `config.py`) |

---

## 6. КОНФИГУРАЦИЯ (СПРАВОЧНИК ДЛЯ LLM)

### 6.1. Критичные env-переменные

```bash
DATABASE_URL=postgresql://my_app_user:PASSWORD@127.0.0.1:5432/nexus_db
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
SECRET_KEY=...
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
LOCAL_ARTIFACTS_DIR=./ml/artifacts
PERSONALITY_MODEL_FILENAME=personality_model_best.pth
MBTI_MODEL_FILENAME=mbti_model.pth
NEXUS_MBTI_NEURAL_BLEND_WEIGHT=0.7
```

### 6.2. Профили конфигурации

| Класс | FLASK_ENV | Особенности |
|-------|-----------|-------------|
| `DevelopmentConfig` | development | DEBUG=True, RATELIMIT_ENABLED=False, MIN_MESSAGES=3 |
| `ProductionConfig` | production | Secure cookies, строгие лимиты |
| `TestingConfig` | testing | SQLite in-memory, USE_LOCAL_AI_MODELS=True |

---

## 7. ДИАГРАММА ПОЛНОГО ЖИЗНЕННОГО ЦИКЛА ДАННЫХ

```
                    ┌──────────────┐
                    │   Browser    │
                    └──────┬───────┘
                           │ HTTP / WS
                    ┌──────▼───────┐
                    │  Flask App   │
                    │  app/chat.py │
                    └──────┬───────┘
                           │ INSERT
                    ┌──────▼───────┐
                    │   messages   │◄──── data/message.py
                    └──────┬───────┘
                           │ .delay()
                    ┌──────▼───────────────┐
                    │  Redis (Celery)      │
                    └──────┬───────────────┘
                           │
              ┌────────────▼────────────┐
              │ analyze_user_profile    │
              │  ├─ get_profiler()      │
              │  ├─ AIProfiler.analyze  │
              │  └─ COMMIT profiles     │
              └────────────┬────────────┘
                           │ .delay()
              ┌────────────▼────────────┐
              │ update_compatibility    │
              │  └─ pgvector SQL query  │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
│ personality    │ │ extracted     │ │ compatibility    │
│ profiles       │ │ interests     │ │ (overall_score)  │
│ + embedding    │ │ (JSON)        │ │                  │
│ Vector(5)      │ │               │ │                  │
└────────┬───────┘ └───────────────┘ └──────────────────┘
         │
         ▼ HNSW index
┌────────────────┐
│ ANN search     │  cosine_distance(embedding, ?)
│ (pgvector)     │
└────────────────┘
```

---

## 8. ПРАВИЛА ДЛЯ LLM-ПОМОЩНИКОВ (QUICK REFERENCE)

1. **Канонические ORM-модели AI** — `data/ai.py`, не `app/ai/models.py`.
2. **`embedding` = OCEAN[5]**, не SBERT; размерность Vector(5).
3. **Celery-задачи** определены в `app/ai/personality_analyzer.py`; `celery_tasks.py` — только re-export.
4. **`get_profiler()`** — единственный способ получить `AIProfiler` (singleton).
5. **Очистка текста** — всегда через `clean_user_text()` из `text_utils.py`.
6. **Совместимость в БД** — только через `update_compatibility` + pgvector; HTTP endpoints могут использовать legacy Python-расчёт.
7. **Миграции** — Alembic через Flask-Migrate; pgvector требует PostgreSQL (не SQLite для prod).
8. **Конфиг** — только `config.py` в корне; не дублировать env-чтение.
9. **При добавлении новых ORM-моделей** — регистрировать в `data/__all_models.py`.
10. **ML-веса** — `ml/artifacts/*.pth`; пути через `config.PERSONALITY_MODEL_PATH`.

---

## 9. ИЗВЕСТНЫЕ ТОЧКИ РАСШИРЕНИЯ

| Задача | Куда смотреть |
|--------|---------------|
| Добавить новый признак личности | `PersonalityClassifier`, `train_ocean.py`, `data/ai.py` (новые колонки + migration) |
| Изменить метрику совместимости | `update_compatibility()` + возможно размерность `Vector(N)` |
| Добавить rate-limit на анализ | `analyze_user_profile()` + Redis/cache |
| Унифицировать compatibility | Заменить Python-расчёт в `profile.py`/`interests.py` на чтение из `ai_user_compatibility` |
| Улучшить extract_interests | `AIProfiler.extract_interests()` или LLM-API (`ANTHROPIC_API_KEY` в config) |
| Новый blueprint | `app/__init__.py` → `register_blueprint` |


### Выявленные архитектурные проблемы (кратко)
- resolve_tag_to_slug хрупок к опечаткам и синонимам; требует fuzzy/semantic fallback (app/ai_profiler/interest_graph.py).
- Рассинхронизация SEMANTIC_ONTOLOGY и _HIERARCHY_SEED: канонические слаги не совпадают (app/ai_profiler/semantic_ontology.py, interest_graph.py).
- graph_score часто = 0 для пользователей без precomputed weights; register_user_tags/seed не покрывают всех (app/ai_profiler/interest_graph.py, tools/seed_db.py).
- Двойная система совместимости (pgvector в БД vs Python-SBERT в endpoints) приводит к расхождению результатов (app/ai/personality_analyzer.py, app/profile.py).

---

*Документ сгенерирован на основе анализа кодовой базы Nexus. Версия стека: Flask 3.1, Celery 5.6, PyTorch 2.11, pgvector 0.2.5, PostgreSQL 16.*

---

