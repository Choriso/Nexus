# Архитектура системы AI-профилирования для Flask приложения

## 📋 Обзор системы

Интеграция чат-бота с AI-профилированием в существующий Flask проект с минимальными изменениями.

## 🏗️ Компоненты системы

### 1. Структура проекта (предлагаемая)

```
your_flask_app/
│
├── app/
│   ├── __init__.py
│   ├── models.py                    # Ваши существующие модели
│   ├── routes/
│   │   ├── chat.py                  # Существующие маршруты чата
│   │   └── profile.py               # Существующие маршруты профиля
│   │
│   ├── ai_profiler/                 # 🆕 НОВЫЙ МОДУЛЬ
│   │   ├── __init__.py
│   │   ├── core.py                  # Основная логика профилирования
│   │   ├── extractors.py            # Извлечение информации из чата
│   │   ├── personality.py           # Анализ личности (Big Five, MBTI)
│   │   ├── compatibility.py         # Расчет совместимости
│   │   ├── embeddings.py            # Семантические эмбеддинги (PyTorch)
│   │   ├── routes.py                # API endpoints для AI функций
│   │   └── tasks.py                 # Фоновые задачи (Celery/RQ)
│   │
│   ├── services/
│   │   ├── llm_service.py           # Интерфейс для работы с LLM
│   │   └── matching_service.py      # Улучшенный матчинг пользователей
│   │
│   └── utils/
│       ├── cache.py                 # Redis кэширование
│       └── nlp_utils.py             # NLP утилиты
│
├── models/                          # 🆕 ML модели
│   ├── sentence_transformer/        # Модель для эмбеддингов
│   └── personality_classifier/      # Опционально: локальная модель
│
├── migrations/                      # Alembic миграции
│   └── versions/
│       └── add_ai_profile_tables.py # Новые таблицы
│
├── config.py
├── requirements.txt
└── run.py
```

---

## 💾 Схема базы данных

### Новые таблицы для интеграции:

```sql
-- Таблица для хранения профилей личности
CREATE TABLE ai_user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    -- Big Five scores (0-1)
    openness FLOAT CHECK (openness >= 0 AND openness <= 1),
    conscientiousness FLOAT CHECK (conscientiousness >= 0 AND conscientiousness <= 1),
    extraversion FLOAT CHECK (extraversion >= 0 AND extraversion <= 1),
    agreeableness FLOAT CHECK (agreeableness >= 0 AND agreeableness <= 1),
    neuroticism FLOAT CHECK (neuroticism >= 0 AND neuroticism <= 1),
    
    -- MBTI тип
    mbti_type VARCHAR(4),
    
    -- Дополнительные характеристики
    communication_style JSONB,  -- {formality: 0.5, enthusiasm: 0.8, ...}
    traits TEXT[],              -- ['креативный', 'вдумчивый', ...]
    values TEXT[],              -- ['творчество', 'саморазвитие', ...]
    
    -- Совместимость
    compatible_mbti_types TEXT[],
    collaboration_style TEXT,
    
    -- Метаданные
    confidence_score FLOAT DEFAULT 0.5,
    last_analyzed TIMESTAMP DEFAULT NOW(),
    conversation_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Таблица для извлеченных интересов из чата
CREATE TABLE ai_extracted_interests (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    -- Интересы
    hobbies TEXT[],
    topics TEXT[],
    skills TEXT[],
    dislikes TEXT[],
    
    -- Работа и цели
    occupation VARCHAR(200),
    work_style TEXT,
    short_term_goals TEXT[],
    long_term_goals TEXT[],
    
    -- Предпочтения
    preferences JSONB,          -- {work_style: "remote", collaboration: "small teams"}
    
    -- Источник данных
    extracted_from_messages INTEGER DEFAULT 0,
    last_extraction TIMESTAMP DEFAULT NOW(),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Таблица для кэширования совместимости
CREATE TABLE user_compatibility_cache (
    id SERIAL PRIMARY KEY,
    user_id_1 INTEGER REFERENCES users(id) ON DELETE CASCADE,
    user_id_2 INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    -- Оценки совместимости
    overall_score FLOAT,
    romantic_score FLOAT,
    professional_score FLOAT,
    creative_score FLOAT,
    interest_overlap FLOAT,
    
    -- Рекомендации
    recommendations TEXT,
    
    calculated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(user_id_1, user_id_2)
);

-- Индексы для быстрого поиска
CREATE INDEX idx_ai_profiles_user_id ON ai_user_profiles(user_id);
CREATE INDEX idx_ai_profiles_mbti ON ai_user_profiles(mbti_type);
CREATE INDEX idx_extracted_interests_user_id ON ai_extracted_interests(user_id);
CREATE INDEX idx_compatibility_users ON user_compatibility_cache(user_id_1, user_id_2);

-- Индекс для поиска по чертам характера (PostgreSQL)
CREATE INDEX idx_traits_gin ON ai_user_profiles USING gin(traits);
```

---

## 🔄 Workflow интеграции

### Вариант 1: Гибридный подход (рекомендую)

**Используем комбинацию:**
- **Claude API** (Anthropic) - для анализа личности и извлечения информации
- **PyTorch + Sentence-BERT** - для семантического поиска и эмбеддингов
- **Scikit-learn** - для расчета совместимости

**Преимущества:**
- Высокая точность анализа (Claude)
- Быстрый локальный поиск (PyTorch)
- Низкая стоимость (API вызовы только для анализа)

### Вариант 2: Полностью локальное решение

**Используем только:**
- **PyTorch** с open-source моделями (LLaMA, Mistral)
- **Sentence Transformers** для эмбеддингов
- **Custom классификаторы** для Big Five

**Преимущества:**
- Полный контроль
- Нет зависимости от внешних API
- Приватность данных

**Недостатки:**
- Требует GPU
- Меньшая точность анализа
- Сложнее в поддержке

---

## 🔌 API Endpoints (новые)

```python
# Анализ и профилирование
POST   /api/ai/analyze-conversation/{user_id}  # Запустить анализ чата
GET    /api/ai/profile/{user_id}               # Получить AI-профиль
GET    /api/ai/profile/{user_id}/summary       # Краткая сводка

# Совместимость
POST   /api/ai/compatibility                   # Расчет совместимости двух пользователей
GET    /api/ai/matches/{user_id}               # Топ совместимых пользователей

# Умный поиск
POST   /api/ai/search/semantic                 # Семантический поиск по интересам
POST   /api/ai/search/personality              # Поиск по типу личности

# Рекомендации
GET    /api/ai/recommendations/{user_id}       # Персональные рекомендации
GET    /api/ai/connections/suggested           # Предлагаемые связи
```

---

## ⚡ Стратегия обработки

### Асинхронная обработка (рекомендую Celery):

```python
# Задачи выполняются в фоне
@celery.task
def analyze_user_conversation(user_id):
    """Анализ всей истории чата пользователя"""
    pass

@celery.task
def update_compatibility_matrix(user_id):
    """Обновление совместимости с другими пользователями"""
    pass

@celery.task
def generate_embeddings_batch(user_ids):
    """Генерация эмбеддингов для пакета пользователей"""
    pass
```

### Триггеры для анализа:

1. **Каждые N сообщений** (например, каждые 10 сообщений)
2. **По расписанию** (ночной анализ всех активных пользователей)
3. **По запросу** (пользователь нажимает "Обновить профиль")

---

## 🎯 Алгоритм работы (пошагово)

### Шаг 1: Сбор данных из чата

```python
# При каждом сообщении
1. Сохранить сообщение в БД (как сейчас)
2. Increment счетчик сообщений для пользователя
3. Если (счетчик % 10 == 0):
   - Запустить фоновую задачу анализа
```

### Шаг 2: Извлечение информации

```python
# В фоновой задаче
1. Получить последние N сообщений пользователя
2. Отправить в LLM (Claude или локальная модель)
3. Извлечь структурированные данные:
   - Интересы, хобби, навыки
   - Работа, цели
   - Предпочтения
4. Сохранить в ai_extracted_interests
```

### Шаг 3: Анализ личности

```python
1. Проанализировать:
   - Словарный запас
   - Стиль общения
   - Темы разговоров
   - Эмоциональную окраску
2. Рассчитать Big Five scores
3. Определить MBTI тип
4. Сохранить в ai_user_profiles
```

### Шаг 4: Генерация эмбеддингов

```python
1. Создать текстовое представление профиля
2. Использовать Sentence-BERT для эмбеддинга
3. Сохранить в векторной БД (pgvector или FAISS)
```

### Шаг 5: Расчет совместимости

```python
1. При изменении профиля:
   - Найти похожих пользователей (по эмбеддингам)
   - Рассчитать детальную совместимость (Big Five)
   - Кэшировать результаты
```

---

## 🚀 План внедрения (фазы)

### Фаза 1: Минимальная интеграция (1-2 недели)
- ✅ Создать новые таблицы в БД
- ✅ Добавить извлечение базовой информации из чата
- ✅ Простой анализ через Claude API
- ✅ Endpoint для получения профиля

### Фаза 2: Расширенная функциональность (2-3 недели)
- ✅ Полный анализ личности (Big Five, MBTI)
- ✅ Расчет совместимости между пользователями
- ✅ Интеграция с существующей системой поиска
- ✅ Фоновые задачи через Celery

### Фаза 3: Оптимизация (1-2 недели)
- ✅ Локальные PyTorch модели для эмбеддингов
- ✅ Кэширование через Redis
- ✅ Векторный поиск (pgvector/FAISS)
- ✅ Dashboard для мониторинга

### Фаза 4: Продвинутые фичи (опционально)
- ✅ Рекомендательная система
- ✅ Предиктивный матчинг
- ✅ Групповая совместимость (для команд)
- ✅ Динамическое обновление профилей

---

## 💰 Примерная стоимость (на 1000 пользователей/месяц)

### С Claude API:
- Анализ профиля: ~$0.01 на пользователя
- 10 анализов на пользователя/месяц = $0.10
- **Итого: ~$100/месяц** для 1000 активных пользователей

### Полностью локально:
- Стоимость сервера с GPU: ~$50-200/месяц
- Нет переменных затрат

---

## 📊 Метрики для мониторинга

```python
# Ключевые метрики
- Точность извлечения информации (precision/recall)
- Время обработки одного профиля
- Процент успешных матчей (пользователи начали общение)
- Удовлетворенность рекомендациями (обратная связь)
- Загрузка системы (задачи в очереди)
```

---

## 🔒 Безопасность и приватность

```python
# Важные аспекты
1. Шифрование чувствительных данных
2. Согласие пользователя на анализ (GDPR)
3. Возможность удалить AI-профиль
4. Анонимизация данных при обучении моделей
5. Rate limiting для API endpoints
```

---

Готов создать код для любой из фаз! Скажите:
1. Какая БД у вас сейчас?
2. С чего начнем - с API интеграции или локальных моделей?
3. Есть ли у вас возможность использовать Celery для фоновых задач?
