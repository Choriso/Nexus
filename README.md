# Nexus — Социальная платформа на основе психологической совместимости

**Nexus** — платформа для поиска единомышленников, использующая AI-профилирование личности (OCEAN Big Five, MBTI, ценности Шварц), граф знаний интересов и гибридный алгоритм совместимости.

---

## Содержание

- [Архитектура](#архитектура)
- [Технологический стек](#технологический-стек)
- [Основные возможности](#основные-возможности)
- [AI-конвейер](#ai-конвейер)
- [Схема базы данных](#схема-базы-данных)
- [Установка и запуск](#установка-и-запуск)
- [Конфигурация](#конфигурация)
- [API-эндпоинты](#api-эндпоинты)
- [Разработка](#разработка)
- [Тестирование](#тестирование)

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Клиент (Browser)                             │
│  HTML/Jinja2  │  D3.js (граф знаний)  │  Chart.js (OCEAN radar)    │
│  Socket.IO (real-time чат)                                          │
├─────────────────────────────────────────────────────────────────────┤
│                        Flask Application                            │
│  Blueprints: auth, profile, chat, interests, moderation, analytics  │
│  Flask-Login │ Flask-SocketIO │ Flask-CORS │ Flask-Migrate          │
├─────────────────────────────────────────────────────────────────────┤
│                        AI / ML Layer                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  AI Profiler (singleton)                                     │   │
│  │  ├── PersonalityClassifier (OCEAN)                           │   │
│  │  ├── MBTIClassifier (16 типов)                               │   │
│  │  └── ContextualAdapter (SBERT enrichment)                   │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  LLM Providers (failover cascade)                            │   │
│  │  ├── YandexGPT (primary)                                     │   │
│  │  └── Ollama (fallback)                                       │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  Search & Ranking                                            │   │
│  │  ├── Graph Score (overlap весов интересов)                   │   │
│  │  ├── OCEAN Similarity                                        │   │
│  │  ├── Jaccard Similarity                                      │   │
│  │  ├── Schwartz Similarity                                     │   │
│  │  └── Root Personality Score                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                        Background Tasks (Celery)                    │
│  Redis (broker)  ←→  analyze_user_profile  →  update_compatibility  │
├─────────────────────────────────────────────────────────────────────┤
│                        Data Layer                                   │
│  SQLAlchemy 2.x │ PostgreSQL 16 + pgvector │ Alembic migrations     │
│  Redis Cache (сессии, кэш)                                          │
└─────────────────────────────────────────────────────────────────────┘
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
| Инструмент | Назначение |
|-----------|-----------|
| Docker / docker-compose | PostgreSQL + pgvector |
| pip / venv | Управление зависимостями |
| pytest | Тестирование |

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

```
Сообщение → POST /messages → analyze_user_profile.delay(user_id)
                                      │
                                      ▼
                         AIProfiler.analyze_profile(text)
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                  PersonalityProfile      ExtractedInterests
                  OCEAN + MBTI + embed    hobbies, skills, goals
                          │
                          ▼
                  update_compatibility.delay(user_id)
                          │
                          ▼
                  pgvector cosine_distance
                  → UserCompatibility
```

### Фаза 2: Поиск (READ)

```
Клик узла → GET /api/graph/match/<node_id>
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  Graph Score       OCEAN Sim        Jaccard Sim
        │                │                │
        └────────────────┼────────────────┘
                         ▼
               Root Personality Score
                         │
                         ▼
               Комбинация с весами
               + micro_gradient_step
                         │
                         ▼
               Отсортированный список кандидатов
```

### Фаза 3: Match-отчёт

```
Нажатие «Анализ ИИ» → GET /api/graph/report/<user_id>?node_id=N
                               │
                    _build_prompt_payload()
                    ┌─────────────────────────┐
                    │ OCEAN + MBTI + Schwartz │
                    │ Поведение + цели        │
                    │ Общие теги              │
                    └─────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              YandexGPT              Ollama/Fallback
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    AI Toast (bottom-left)
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

---

## Установка и запуск

### Требования

- Python 3.12+
- PostgreSQL 16 (или Docker)
- Redis 7+
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

### 6. Миграции

```bash
flask db upgrade
```

### 7. Запуск

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

### Прочее

| Маршрут | Метод | Описание |
|---------|-------|----------|
| `/register` | GET/POST | Регистрация |
| `/login` | GET/POST | Вход |
| `/upload_avatar` | POST | Загрузка аватара |
| `/favorite/<interest_id>` | POST | Избранное |

---

## Разработка

### Структура проекта

```
Nexus/
├── App.py                     # Точка входа
├── config.py                  # Конфигурация
├── app/                       # Flask-приложение
│   ├── __init__.py            # create_app()
│   ├── profile.py             # Профиль, matching, граф
│   ├── chat.py                # Чаты
│   ├── ai/
│   │   ├── personality_analyzer.py  # Celery-задачи
│   │   └── match_report.py          # AI-ревью
│   └── ai_profiler/           # AI-ядро
├── data/                      # ORM-модели
├── templates/                 # Jinja2-шаблоны
├── static/                    # CSS, JS
├── tools/                     # Утилиты
│   ├── seed_db.py             # Сидинг БД
│   └── test_node_match.py     # Тестер matching
└── test_metrics.py            # 55 тестов метрик
```

### CSS-переменные

Все цвета определяются в `static/src/styles/variables.css`:

```css
--color-bg: #2b2e31;              /* Фон */
--color-bg-deep: #1b1b20;         /* Граф */
--color-panel: #3C4044;           /* Панели */
--color-card: #242629;            /* Карточки */
--color-accent-purple: #7f5af0;   /* Акцент графа */
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
