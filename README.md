# Nexus — Социальная платформа на основе психологической совместимости

**Nexus** — платформа для поиска единомышленников, использующая AI-профилирование личности (OCEAN Big Five, MBTI, ценности Шварц), граф знаний интересов и гибридный алгоритм совместимости.

---

## Содержание

- [Архитектура](#архитектура)
- [Технологический стек](#технологический-стек)
- [Основные возможности](#основные-возможности)
- [AI-конвейер](#ai-конвейер)
- [Схема базы данных](#схема-базы-данных)
- [Обучение нейросетей](#обучение-нейросетей)
- [Результаты модели](#результаты-модели)
- [Установка и запуск](#установка-и-запуск)
- [Конфигурация](#конфигурация)
- [API-эндпоинты](#api-эндпоинты)
- [Разработка](#разработка)
- [Тестирование](#тестирование)

---

## Архитектура

```mermaid
graph TB
    subgraph Client["Клиент (Browser)"]
        A1["HTML / Jinja2"]
        A2["D3.js (граф знаний)"]
        A3["Chart.js (OCEAN radar)"]
        A4["Socket.IO (real-time чат)"]
    end

    subgraph Flask["Flask Application"]
        B["Blueprints: auth, profile, chat, interests, moderation, analytics\nFlask-Login · Flask-SocketIO · Flask-CORS · Flask-Migrate"]
    end

    subgraph AI["AI / ML Layer"]
        C1["AI Profiler (singleton)"]
        C1a["PersonalityClassifier (OCEAN)"]
        C1b["MBTIClassifier (16 типов)"]
        C1c["ContextualAdapter (SBERT enrichment)"]
        C1 --> C1a & C1b & C1c

        C2["LLM Providers (failover cascade)"]
        C2a["YandexGPT (primary)"]
        C2b["Ollama (fallback)"]
        C2 --> C2a & C2b

        C3["Search & Ranking"]
        C3a["Graph Score"]
        C3b["OCEAN Similarity"]
        C3c["Jaccard Similarity"]
        C3d["Schwartz Similarity"]
        C3e["Root Personality Score"]
        C3 --> C3a & C3b & C3c & C3d & C3e
    end

    subgraph BG["Background Tasks (Celery)"]
        D1["Redis (broker)"]
        D2["analyze_user_profile"]
        D3["update_compatibility"]
        D1 <--> D2 --> D3
    end

    subgraph Data["Data Layer"]
        E1["SQLAlchemy 2.x"]
        E2["PostgreSQL 16 + pgvector"]
        E3["Alembic migrations"]
        E4["Redis Cache"]
    end

    Client --> Flask --> AI & BG --> Data
```

---

## Технологический стек

### Backend
| Компонент | Технология | Версия |
|-----------|-----------|--------|
| Язык | Python | 3.12+ |
| Веб-фреймворк | Flask | 3.x |
| Аутентификация | Flask-Login | — |
| Realtime | Flask-SocketIO | — |
| База данных | PostgreSQL | 16 |
| Векторное расширение | pgvector | 0.2.x |
| ORM | SQLAlchemy | 2.x |
| Миграции | Alembic / Flask-Migrate | — |
| Очереди | Celery | 5.x |
| Брокер | Redis | 7.x |
| CORS | Flask-CORS | — |

### Machine Learning
| Компонент | Технология |
|-----------|-----------|
| Фреймворк | PyTorch 2.x |
| Эмбеддинги | SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Классификация OCEAN | PersonalityClassifier (Residual Blocks + Ordinal Regression) |
| Классификация MBTI | MBTIClassifier (MLP 384→256→128→16) |
| NLP | Transformers, NLTK, scikit-learn |

### LLM
| Провайдер | Назначение | Статус |
|-----------|-----------|--------|
| YandexGPT | Генерация match-отчётов, разрешение тегов | Primary |
| Ollama | Fallback для LLM-запросов | Optional |

### DevOps
| Инструмент | Назначение | Версия |
|-----------|-----------|--------|
| Docker Engine | Контейнеризация | >= 24.0 |
| Docker Compose | Оркестрация | >= 2.24 |
| Nginx | Reverse proxy + SSL | 1.27-alpine |
| Gunicorn + Eventlet | WSGI + async workers | 23.0 / 0.38 |
| pip / venv | Управление зависимостями | — |
| pytest | Тестирование | — |

---

## Основные возможности

### 1. AI-профилирование личности

Автоматический анализ личности из текстовых сообщений:

- **OCEAN (Big Five)**: 5 непрерывных шкал [0.0, 1.0] — Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
- **MBTI**: 16 типов личности (INTJ, ENFP, ...) — нейросетевой + rule-based бленд
- **Ценности Шварц**: 10 базовых ценностей (self-direction, stimulation, hedonism, ...)
- **Поведенческие метрики**: средняя длина сообщения, время ответа, частота эмодзи
- **Стиль общения**: формальность, энтузиазм, ориентация на детали, стиль сотрудничества
- **Корневой архетип**: работа / хобби / психология — на основе категорий узлов графа

### 2. Граф знаний интересов

- Визуальный интерактивный граф (SVG + D3.js)
- Узлы с категориями: `work`, `hobby`, `psychology`, `want`
- Drag-and-drop, создание/редактирование/удаление узлов
- Иерархия интересов с материализованными путями

### 3. Поиск и совместимость

Двухконтурная система:

| Контур | Механизм | Результат |
|--------|----------|-----------|
| Фоновый (Celery) | pgvector cosine_distance по OCEAN-векторам | `UserCompatibility` в БД |
| Интерактивный (HTTP) | Гибридный скор: Graph × OCEAN × Jaccard × Schwartz × Root | Мгновенный поиск по узлу графа |

Каждый скор учитывает персональные смещения весов (`User.metric_weight_*_offset`) и корректируется микро-градиентным шагом.

### 4. AI-ревью совместимости

При нажатии «Анализ ИИ» в карточке кандидата генерируется персонализированное обоснование:

1. **YandexGPT** — синхронный вызов API
2. **Ollama** — fallback
3. **Шаблонный** — статический fallback

Текст выводится в плавающем тосте в левом нижнем углу, не блокируя просмотр других кандидатов.

### 5. Чат-система

- REST + Socket.IO для real-time
- Диалоги 1:1
- Отправка файлов и изображений
- Каждое сообщение триггерит AI-анализ автора

### 6. UI-компоненты

- **Главная страница**: граф знаний + сайдбар с кандидатами + AI Toast
- **Свой профиль**: OCEAN radar (Chart.js), MBTI, Schwartz-ценности, AI-интересы
- **Профиль другого пользователя**: психологическая карточка с полным разбором
- **Единая CSS-система**: все цвета через CSS-переменные в `variables.css`

---

## AI-конвейер

### Фаза 1: Анализ сообщений (WRITE)

```mermaid
sequenceDiagram
    participant U as Пользователь
    participant S as Сервер
    participant AI as AIProfiler
    participant DB as База данных

    U->>S: POST /messages
    S->>AI: analyze_user_profile.delay(user_id)
    AI->>AI: analyze_profile(text)
    par PersonalityProfile
        AI-->>DB: OCEAN + MBTI + embed
    and ExtractedInterests
        AI-->>DB: hobbies, skills, goals
    end
    AI->>AI: update_compatibility.delay(user_id)
    AI->>DB: pgvector cosine_distance
    DB-->>S: UserCompatibility
```

### Фаза 2: Поиск (READ)

```mermaid
flowchart LR
    A["Клик узла"] --> B["GET /api/graph/match/&lt;node_id&gt;"]
    B --> C["Graph Score"]
    B --> D["OCEAN Similarity"]
    B --> E["Jaccard Similarity"]
    C & D & E --> F["Root Personality Score"]
    F --> G["Комбинация с весами<br/>+ micro_gradient_step"]
    G --> H["Отсортированный список кандидатов"]
```

### Фаза 3: Match-отчёт

```mermaid
flowchart TB
    A["Нажатие «Анализ ИИ»"] --> B["GET /api/graph/report/&lt;user_id&gt;?node_id=N"]
    B --> C["_build_prompt_payload()"]
    C --> D["OCEAN + MBTI + Schwartz<br/>Поведение + цели<br/>Общие теги"]
    D --> E{"LLM Cascade"}
    E --> F["YandexGPT (primary)"]
    E --> G["Ollama / Шаблон (fallback)"]
    F & G --> H["AI Toast (bottom-left)"]
```

---

## Схема базы данных

### Ключевые таблицы

| Таблица | Назначение | Ключевые поля |
|---------|-----------|---------------|
| `users` | Пользователи | `name`, `email`, `information`, `connection`, `metric_weight_*_offset` |
| `ai_user_personality_profiles` | OCEAN + MBTI | `openness..neuroticism`, `embedding(Vector(5))`, `mbti_type`, `communication_style` |
| `ai_extracted_interests` | AI-интересы | `hobbies(JSON)`, `skills(JSON)`, `goals(JSON)`, `occupation` |
| `user_schwartz_profiles` | Ценности Шварц | `self_direction..universalism` (10 полей Float) |
| `user_behavior_profiles` | Поведение | `avg_char_count`, `avg_reply_time`, `avg_emoji_count` |
| `ai_user_compatibility` | Совместимость | `overall_score`, `romantic_score`, `professional_score` |
| `knowledge_nodes` | Узлы графа | `title`, `category`, `x`, `y` |
| `knowledge_connections` | Связи графа | `from_node_id`, `to_node_id`, `label` |
| `interest_hierarchy_nodes` | Иерархия | `slug`, `path`, `embedding(Vector(384))` |
| `user_interest_graph_weights` | Веса интересов | `weight`, `source_tag` |
| `dynamic_aliases` | Кэш тегов | `raw_tag`, `slug`, `confidence` |
| `global_weights_config` | Веса метрик | `weight_ocean`, `weight_graph`, `weight_jaccard` |
| `blocked_users` | Заблокированные | `user_id`, `blocked_user_id` |

### pgvector

OCEAN-векторы хранятся в колонке `embedding Vector(5)` с HNSW-индексом:

```sql
CREATE INDEX idx_user_personality_embedding_hnsw
ON ai_user_personality_profiles
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

Поиск ближайших соседей выполняется прямо в SQL:
```python
profile.embedding.cosine_distance(my.embedding)
```

### ER-диаграмма

```mermaid
erDiagram
    users ||--o{ ai_user_personality_profiles : has
    users ||--o{ ai_extracted_interests : has
    users ||--o{ user_schwartz_profiles : has
    users ||--o{ user_behavior_profiles : has
    users ||--o{ ai_user_compatibility : "matched with"
    users ||--o{ knowledge_nodes : creates
    users ||--o{ user_interest_graph_weights : configures
    knowledge_nodes ||--o{ knowledge_connections : connects

    users {
        int id PK
        string name
        string email
        float metric_weight_ocean_offset
        float metric_weight_graph_offset
        float metric_weight_jaccard_offset
        float metric_weight_schwartz_offset
        boolean allow_location
        string image_path
    }

    ai_user_personality_profiles {
        int id PK
        int user_id FK
        float openness
        float conscientiousness
        float extraversion
        float agreeableness
        float neuroticism
        vector embedding "Vector(5)"
        string mbti_type
        string communication_style
    }

    knowledge_nodes {
        int id PK
        int user_id FK
        string title
        string category
        float x
        float y
    }

    interest_hierarchy_nodes {
        int id PK
        string slug
        string path
        vector embedding "Vector(384)"
    }
```

---

## Обучение нейросетей

Модуль `ml/` содержит полный пайплайн обучения трёх нейросетевых моделей, используемых в AI-ядре платформы.

### OCEAN PersonalityClassifier

```mermaid
flowchart LR
    subgraph Data["Подготовка данных"]
        GEN["Сгенерированный JSON\n(generated_data_ocean.json)"]
        PREP["preprocess.py\nSBERT эмбеддинг (384d)\n+ 4 ручных признака"]
        PREP_OUT["train_data_precomputed.pt\n(388d тензоры)"]
        GEN --> PREP --> PREP_OUT
    end

    subgraph Arch["Архитектура сети"]
        INPUT["Input (388)"] --> PROJ["Linear 388→256\nLayerNorm + GELU + Dropout"]
        PROJ --> RES["ResidualBlock (256)\nLinear→LN→GELU→Dropout→Linear→LN\n+ skip connection"]
        RES --> BOT["Bottleneck 256→128\nLayerNorm + GELU + Dropout"]
        BOT --> CLASS["Linear 128→50\n(5 traits × 10 bins)"]
        CLASS --> OUT["Reshape: (B, 5, 10)\nOrdinal Softmax → взвешенное ожидание"]
    end

    Data --> Arch
```

**Архитектура:**
- **Вход**: 388-мерный вектор (384 SBERT + 4 ручных признака: длина сообщения, доля CAPS, кол-во `!`, кол-во `?`)
- **Projection**: `Linear(388, 256) → LayerNorm → GELU → Dropout(0.2)`
- **ResidualBlock**: `Linear(256,256) → LayerNorm → GELU → Dropout(0.15) → Linear(256,256) → LayerNorm → + skip connection → GELU`
- **Bottleneck**: `Linear(256, 128) → LayerNorm → GELU → Dropout(0.2)`
- **Classifier head**: `Linear(128, 5 × 10)` → reshape to `(B, 5, 10)`
- **Выход**: ординальная регрессия по 10 бинам для каждой из 5 черт, конвертация в непрерывное значение [0, 1] через взвешенное softmax-ожидание

**Функция потерь**: Кастомный `ordinal_loss` — бинарная кросс-энтропия по кумулятивным вероятностям `P(Y ≤ k)` с label smoothing 0.02.

**Оптимизатор**: AdamW (lr=3e-4, weight_decay=1e-5), ReduceLROnPlateau (factor=0.5, patience=12), EarlyStopping (patience=40 по R²).

**Данные:**
- Сгенерированный `generated_data_ocean.json` — синтетические примеры текстов с OCEAN-оценками
- Итого: ~17 000 записей, сплит 85/15 train/val

**Процесс обучения:**
```bash
cd ml
python preprocess.py --input data/generated_data_ocean.json --output data/train_data_precomputed.json
python precompute_embedding.py       # → train_data_precomputed.pt
python train_ocean.py                # → artifacts/personality_model_best.pth
```

---

### MBTIClassifier

**Архитектура:**

```mermaid
flowchart LR
    TEXT["Текст сообщения"] --> SBERT["paraphrase-multilingual-\nMiniLM-L12-v2"]
    SBERT --> EMB["384d эмбеддинг"]
    EMB --> L1["Linear 384→256\nLayerNorm + GELU + Dropout(0.3)"]
    L1 --> L2["Linear 256→128\nLayerNorm + GELU + Dropout(0.2)"]
    L2 --> L3["Linear 128→16"]
    L3 --> SOFT["Softmax\n16 MBTI типов"]
```

**MLP**: 384 → 256 (LayerNorm, GELU, Dropout 0.3) → 128 (LayerNorm, GELU, Dropout 0.2) → 16 (логиты)

**Функция потерь**: CrossEntropyLoss с WeightedRandomSampler (балансировка редких типов).

**Оптимизатор**: AdamW (lr=5e-4, weight_decay=0.01), EarlyStopping (patience=12 по val_acc).

**Данные**: Сгенерированный набор текстов с размеченным MBTI. Сплит 80/20 стратифицированный.

**Запуск:**
```bash
python train_mbti.py  # → artifacts/mbti_model.pth
```

---

### Interest Head (Zero-shot классификатор интересов)

```mermaid
flowchart LR
    TEXT["Текст пользователя"] --> SBERT["SBERT\n384d"]
    TAX["Таксономия интересов\n(глобальная категория, подкатегория)"]
    TAX --> ANCHORS["Якорные тексты"]
    ANCHORS --> SBERT2["SBERT\n384d"]
    SBERT --> COS["Cosine Similarity\nс каждым якорем"]
    SBERT2 --> COS
    COS --> THRESH["Порог confidence\n→ matched tags"]
```

**CustomInterestClassifier**: линейная проекция 384 → 128 → `num_classes` с Softmax. Обучается на якорях из `INTEREST_TAXONOMY` (15 эпох, lr=0.001).

**Запуск:**
```bash
python train_interest_head.py  # → artifacts/interest_head.pth
```

---

### Fine-tuning SBERT

Шаблон для дообучения эмбеддингов на парах (сленг → каноническое описание) через `MultipleNegativesRankingLoss`:

```bash
python finetune_sbert.py \
    --pairs data/slang_pairs.json \
    --output ml/artifacts/sbert-finetuned \
    --epochs 3 \
    --batch-size 16
```

---

## Результаты модели

### OCEAN PersonalityClassifier

| Метрика | Значение |
|---------|----------|
| **R² (лучший)** | 0.6724 |
| **MAE (лучший)** | 0.0621 |
| **Функция потерь** | Ordinal Binary Cross-Entropy |
| **Архитектура** | ResidualBlock 256 → Bottleneck 128 → 5×10 Ordinal |
| **Данные** | ~17 000 примеров (85/15 split) |
| **Эпох до best** | ~60 |

### MBTI Classifier

| Метрика | Значение |
|---------|----------|
| **Val Accuracy** | ~72% на отложенной выборке |
| **Балансировка** | WeightedRandomSampler по частотам классов |
| **Архитектура** | MLP 384→256→128→16 |
| **Данные** | Сгенерированный датасет |

### График обучения

![Training History](ml/artifacts/training_history.png)

*График обновляется после каждого запуска `train_ocean.py`. На графике: Loss (train/val), R² Score, Learning Rate.*

---

## Установка и запуск

### Требования

- Docker Engine >= 24.0 + Docker Compose >= 2.24
- Python 3.12+ (для локальной разработки без Docker)
- PostgreSQL 16 + pgvector (автоматически через Docker)
- Redis 7+ (автоматически через Docker)
- (Опционально) Ollama с установленной моделью

### 1. Клонирование

```bash
git clone https://github.com/your-username/nexus.git
cd nexus
```

### 2. Виртуальное окружение

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Зависимости

```bash
pip install -r docs/requirements.txt
```

### 4. База данных

Через Docker:
```bash
docker run -d --name nexus-pg \
  -e POSTGRES_USER=my_app_user \
  -e POSTGRES_DB=nexus_db \
  -e POSTGRES_PASSWORD=your-pass \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 5. Конфигурация

Создать `.env` в корне проекта:

```bash
DATABASE_URL=postgresql://my_app_user:your-pass@127.0.0.1:5432/nexus_db
SECRET_KEY=your-secret-key
YANDEX_GPT_API_KEY=your-api-key   # для match-отчётов
```

Полный список переменных — в `config.py`.

### 6. Запуск (Docker — рекомендовано)

```bash
# Development
make dev
# или
docker compose up -d

# Production
make prod
# или
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 7. Миграции

```bash
# Автоматически при старте контейнера (entrypoint.sh)
# Вручную:
flask db upgrade
```

### 8. Локальный запуск (без Docker)

```bash
# Dev-сервер
python App.py

# Celery worker (в отдельном терминале)
celery -A app.ai.personality_analyzer worker --loglevel=info
```

### 8. Наполнение БД (опционально)

```bash
python tools/seed_db.py
```

Создаёт 200 тестовых пользователей с архетипами, чатами и AI-профилями.

---

## Конфигурация

Ключевые переменные окружения (полный список в `config.py`):

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `DATABASE_URL` | — | URL подключения к PostgreSQL |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Брокер Celery |
| `SECRET_KEY` | — | Ключ шифрования сессий |
| `YANDEX_GPT_API_KEY` | — | API-ключ YandexGPT |
| `YANDEX_GPT_MODEL` | `yandexgpt-5.1/latest` | Модель YandexGPT |
| `OLLAMA_ENABLED` | `False` | Включить Ollama |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | SBERT-модель |
| `ROOT_PERSONALITY_BLEND_WEIGHT` | `0.3` | Вес корневого архетипа |
| `MBTI_NEURAL_BLEND_WEIGHT` | `0.7` | Вес нейросети в MBTI |

---

## API-эндпоинты

### Профиль и граф

| Маршрут | Метод | Описание |
|---------|-------|----------|
| `/viewProfile?user_id=N` | GET | Профиль пользователя |
| `/profile` | GET/POST | Свой профиль |
| `/api/graph/match/<node_id>` | GET | Поиск кандидатов по узлу |
| `/api/graph/report/<user_id>?node_id=N` | GET | AI-ревью совместимости |
| `/knowledge_graph_data` | GET | Данные для визуализации графа |
| `/knowledge_graph/node` | POST | Создать узел |
| `/knowledge_graph/node/<id>` | PUT | Обновить узел |
| `/knowledge_graph/node/<id>` | DELETE | Удалить узел |

### Чат

| Маршрут | Метод | Описание |
|---------|-------|----------|
| `/create_chat/<user_id>` | POST | Создать чат |
| `/chat` | GET | Страница чата |
| `/messages/<chat_id>` | GET | История сообщений |
| `/chat/<chat_id>/delete` | DELETE | Удалить чат |
| `/chat/<user_id>/block` | POST | Заблокировать пользователя |
| `/chat/<user_id>/unblock` | POST | Разблокировать пользователя |

### Прочее

| Маршрут | Метод | Описание |
|---------|-------|----------|
| `/register` | GET/POST | Регистрация |
| `/login` | GET/POST | Вход |
| `/upload_avatar` | POST | Загрузка аватара |
| `/favorite/<interest_id>` | POST | Избранное |
| `/terms` | GET | Страница правил |

---

## Разработка

### Структура проекта

```
Nexus/
├── App.py                     # Точка входа (dev)
├── wsgi.py                    # WSGI точка входа (prod — gunicorn)
├── config.py                  # Конфигурация (все env читаются здесь)
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Полный стек (app + db + redis + celery + nginx)
├── entrypoint.sh              # Точка входа контейнера
│
├── app/                       # Flask-приложение
│   ├── __init__.py            # create_app()
│   ├── profile.py             # Профиль, matching, граф
│   ├── chat.py                # Чаты
│   ├── ai/                    # Celery-задачи, LLM, match-отчёты
│   └── ai_profiler/           # AI-ядро (OCEAN, MBTI, граф интересов)
│
├── data/                      # SQLAlchemy ORM-модели
├── db/                        # SQL-скрипты БД (init.sql с pgvector)
├── tests/                     # Pytest-тесты (55+ тестов метрик)
├── tools/                     # Утилиты (seed_db, clean_seed, diagnose)
├── templates/                 # Jinja2-шаблоны (14 файлов)
├── static/                    # CSS, JS, картинки
├── nginx/                     # Nginx reverse proxy config
├── docs/                      # Документация
├── migrations/                # Alembic миграции
├── ml/                        # Обучение моделей
│   ├── train_ocean.py         # OCEAN PersonalityClassifier
│   ├── train_mbti.py          # MBTI классификатор
│   ├── train_interest_head.py # Interest Head
│   ├── finetune_sbert.py      # Fine-tuning SBERT
│   ├── preprocess.py          # Извлечение признаков
│   ├── prepare_dataset.py     # Подготовка датасета
│   ├── precompute_embedding.py
│   ├── test_model.py          # Тест инференса
│   ├── data/                  # Датасеты
│   └── artifacts/             # Обученные веса + графики
└── LICENSE
```

### CSS-переменные

Все цвета определяются в `static/src/styles/variables.css`:

```css
--color-bg: #2b2e31;              /* Фон */
--color-bg-deep: #1b1b20;         /* Граф */
--color-panel: #3C4044;           /* Панели */
--color-card: #242629;            /* Карточки */
--color-primary: #7f5af0;         /* Основной акцент */
--color-accent: #FD7B41;          /* Вторичный акцент */
--color-success: #2cb67d;         /* Теги */
--color-text: #DDDCDB;            /* Текст */
--color-text-heading: #fffffe;    /* Заголовки */
```

---

## Тестирование

```bash
# Все тесты метрик
pytest test_metrics.py -v

# 12 тестов проходят (один предустановленный сбой не связан с метриками)
```

Группы тестов: Graph Score, Schwartz Similarity, Root Personality, Jaccard, OCEAN, Mapping Integrity, Edge Cases.

---

## Лицензия

MIT License. Подробнее в файле LICENSE.

---

*Проект разработан для исследовательских и образовательных целей в области AI-профилирования личности и социальных рекомендательных систем.*