"""
Онтологический словарь Nexus для ContextualAdapter.

Каждая запись раскрывает сленг, аббревиатуры и англицизмы в семантически
богатый текст, понятный базовой SBERT-модели (paraphrase-multilingual-MiniLM).

Структура записи:
    aliases       — варианты написания (нижний регистр, для поиска подстроки)
    enriched_text — развёрнутое описание для encode()
    category      — глобальная категория таксономии (work | life | entertainment)
    subcategory   — подкатегория из INTEREST_TAXONOMY
    parent        — родительское направление (для контекста в промпте отчёта)
"""

from __future__ import annotations

SEMANTIC_ONTOLOGY: dict[str, dict] = {
    # ============================================================
    # КОРНЕВЫЕ УЗЛЫ
    # ============================================================
    "it_development": {
        "aliases": ["айти", "it", "разработка", "программирование", "software", "код"],
        "enriched_text": (
            "IT и разработка программного обеспечения, программирование, "
            "создание приложений и сервисов, технологический стек"
        ),
        "category": "work",
        "subcategory": "IT и Разработка",
        "parent": "Профессиональная деятельность",
    },
    "gaming": {
        "aliases": ["гейминг", "игры", "геймер", "видеоигры", "поиграть", "геймплей"],
        "enriched_text": (
            "компьютерные и видеоигры, гейминг, игровая культура, "
            "киберспорт, настольные игры, развлечения и досуг"
        ),
        "category": "entertainment",
        "subcategory": "Гейминг",
        "parent": "Развлечения и досуг",
    },
    "creativity_art": {
        "aliases": ["творчество", "искусство", "арт", "креатив", "творю", "рисую"],
        "enriched_text": (
            "творчество и искусство, рисование, дизайн, фотография, "
            "создание визуального контента, самовыражение"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Хобби и увлечения",
    },
    "self_development": {
        "aliases": ["саморазвитие", "рост", "развитие", "обучение", "навыки", "учусь"],
        "enriched_text": (
            "саморазвитие и личностный рост, обучение новым навыкам, "
            "тайм-менеджмент, продуктивность, карьерное развитие"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Личностный рост",
    },
    "sports_active_life": {
        "aliases": ["спорт", "активный отдых", "тренировки", "фитнес", "активность"],
        "enriched_text": (
            "спорт и активный образ жизни, тренировки, фитнес, "
            "бег, велоспорт, туризм, походы и приключения"
        ),
        "category": "life",
        "subcategory": "Спорт и Активный отдых",
        "parent": "Здоровье и активность",
    },
    "psychology_relations": {
        "aliases": ["психология", "отношения", "психотерапия", "общение", "эмоции"],
        "enriched_text": (
            "психология и отношения, психотерапия, медитация, "
            "осознанность, межличностная коммуникация, ментальное здоровье"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Ментальное здоровье",
    },
    "music_audio": {
        "aliases": ["музыка", "аудио", "звук", "треки", "концерт", "слушаю музыку"],
        "enriched_text": (
            "музыка и аудио, прослушивание и создание музыки, "
            "игра на инструментах, продюсирование, диджеинг"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Развлечения и творчество",
    },
    "cinema_video": {
        "aliases": ["кино", "видео", "фильмы", "сериалы", "кинематограф", "смотрю"],
        "enriched_text": (
            "кино и видео, просмотр фильмов и сериалов, "
            "кинематограф, видеомонтаж, анимация, документалистика"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Медиа и развлечения",
    },
    "literature_reading": {
        "aliases": ["литература", "чтение", "книги", "читаю", "книголюб", "библиотека"],
        "enriched_text": (
            "литература и чтение, книги, комиксы, манга, "
            "художественная и научная фантастика, поэзия"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Культура и досуг",
    },
    "home_lifestyle": {
        "aliases": ["дом", "быт", "образ жизни", "уют", "хозяйство", "жилье"],
        "enriched_text": (
            "дом и образ жизни, кулинария, DIY и сделай сам, "
            "домашние животные, умный дом, бытовые увлечения"
        ),
        "category": "life",
        "subcategory": "Дом и Образ жизни",
        "parent": "Быт и увлечения",
    },
    "science_education": {
        "aliases": ["наука", "образование", "исследования", "изучение", "учеба"],
        "enriched_text": (
            "наука и образование, космос и астрономия, физика, "
            "философия, научные исследования, академическое обучение"
        ),
        "category": "work",
        "subcategory": "Наука и Образование",
        "parent": "Интеллектуальная деятельность",
    },
    "finance_business": {
        "aliases": ["финансы", "бизнес", "инвестиции", "стартап", "деньги", "карьера"],
        "enriched_text": (
            "финансы и бизнес, инвестиции, фондовый рынок, "
            "криптовалюта, стартапы, предпринимательство"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Бизнес и карьера",
    },

    # ============================================================
    # 1. IT И РАЗРАБОТКА
    # ============================================================
    "backend_dev": {
        "aliases": ["бэкенд", "backend", "серверная разработка", "бэк", "апи", "сервер"],
        "enriched_text": (
            "бэкенд-разработка, создание серверной логики, API и баз данных, "
            "архитектура серверных приложений на Python, Go, Node.js, Java"
        ),
        "category": "work",
        "subcategory": "Бэкенд-разработка",
        "parent": "IT и Разработка",
    },
    "backend_python": {
        "aliases": ["питон", "python бэкенд", "пайтон", "джанго", "фастапи", "фласк"],
        "enriched_text": (
            "Python-экосистема для бэкенда, Django, FastAPI, Flask, "
            "асинхронное программирование, ORM и серверные фреймворки"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_flask": {
        "aliases": ["flask", "фласк", "фласка", "flask python"],
        "enriched_text": (
            "веб-фреймворк Flask на Python, легковесный бэкенд, "
            "создание REST API, микрофреймворк для веб-приложений"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_fastapi": {
        "aliases": ["fastapi", "фастапи", "фаст", "fast api", "fastapi python"],
        "enriched_text": (
            "высокопроизводительный веб-фреймворк FastAPI на Python, "
            "асинхронная разработка API, автоматическая документация Swagger"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_django": {
        "aliases": ["django", "джанго", "дянго", "джанга", "django python"],
        "enriched_text": (
            "полнофункциональный веб-фреймворк Django на Python, "
            "ORM, админка, создание сложных веб-приложений и сайтов"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_asyncio": {
        "aliases": ["asyncio", "асинхронный python", "async", "асинхронность", "async await"],
        "enriched_text": (
            "асинхронное программирование на Python с asyncio, "
            "конкурентность, event loop, async/await, высоконагруженные системы"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_aiohttp": {
        "aliases": ["aiohttp", "аиохттп", "aiohttp python"],
        "enriched_text": (
            "асинхронный HTTP-клиент и сервер Aiohttp на Python, "
            "веб-сокеты, высокая производительность"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_sqlalchemy": {
        "aliases": ["sqlalchemy", "алхимия", "салалхими", "orm python"],
        "enriched_text": (
            "SQLAlchemy ORM на Python, работа с базами данных через объекты, "
            "миграции, построение запросов на питоне"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_pydantic": {
        "aliases": ["pydantic", "пидантик", "валидация данных python"],
        "enriched_text": (
            "библиотека Pydantic для валидации данных и настроек на Python, "
            "типизированные модели данных, сериализация"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_celery": {
        "aliases": ["celery", "селери", "сельдерей", "celery python", "фоновые задачи"],
        "enriched_text": (
            "распределенная очередь задач Celery на Python, "
            "асинхронное выполнение фоновых задач, воркеры, брокеры сообщений"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_poetry": {
        "aliases": ["poetry", "поэтри", "менеджер пакетов python"],
        "enriched_text": (
            "инструмент Poetry для управления зависимостями и сборки пакетов Python, "
            "изоляция окружений, публикация библиотек"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "python_black": {
        "aliases": ["black", "блэк", "black formatter", "форматтер python"],
        "enriched_text": (
            "автоматический форматтер кода Black для Python, "
            "единый стиль кода, PEP8, чистый и читаемый код"
        ),
        "category": "work",
        "subcategory": "Python-экосистема",
        "parent": "Бэкенд-разработка",
    },

    "backend_nodejs": {
        "aliases": ["nodejs", "нода js", "нод", "node", "node.js", "нода"],
        "enriched_text": (
            "бэкенд-разработка на Node.js, серверный JavaScript, "
            "Express, NestJS, работа с потоками и событиями"
        ),
        "category": "work",
        "subcategory": "Node.js-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "nodejs_express": {
        "aliases": ["express", "экспресс", "express.js", "экспресс js"],
        "enriched_text": (
            "минималистичный веб-фреймворк Express.js на Node.js, "
            "создание REST API, middleware, маршрутизация запросов"
        ),
        "category": "work",
        "subcategory": "Node.js-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "nodejs_nestjs": {
        "aliases": ["nestjs", "нест", "nest", "нест js", "nest.js"],
        "enriched_text": (
            "прогрессивный фреймворк NestJS на Node.js с TypeScript, "
            "модульная архитектура, декораторы, внедрение зависимостей"
        ),
        "category": "work",
        "subcategory": "Node.js-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "nodejs_graphql_apollo": {
        "aliases": ["apollo", "аполло", "graphql apollo", "apollo server"],
        "enriched_text": (
            "сервер GraphQL Apollo на Node.js, гибкие запросы данных, "
            "схема GraphQL, резолверы, федерация данных"
        ),
        "category": "work",
        "subcategory": "Node.js-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "nodejs_prisma": {
        "aliases": ["prisma", "призма", "prisma orm", "призма орм"],
        "enriched_text": (
            "современная ORM Prisma для Node.js и TypeScript, "
            "миграции баз данных, типобезопасные запросы, генерация клиента"
        ),
        "category": "work",
        "subcategory": "Node.js-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "nodejs_typeorm": {
        "aliases": ["typeorm", "тайп орм", "type orm"],
        "enriched_text": (
            "TypeORM на Node.js и TypeScript, работа с реляционными БД, "
            "Active Record и Data Mapper паттерны"
        ),
        "category": "work",
        "subcategory": "Node.js-экосистема",
        "parent": "Бэкенд-разработка",
    },
    "nodejs_fastify": {
        "aliases": ["fastify", "фастифай", "быстрый nodejs"],
        "enriched_text": (
            "высокопроизводительный веб-фреймворк Fastify на Node.js, "
            "низкие накладные расходы, плагины, JSON Schema"
        ),
        "category": "work",
        "subcategory": "Node.js-экосистема",
        "parent": "Бэкенд-разработка",
    },

    "backend_go": {
        "aliases": ["go", "golang", "го", "голанг", "go бэкенд"],
        "enriched_text": (
            "бэкенд-разработка на Go (Golang), высокая производительность, "
            "конкурентность, микросервисы, строгая типизация"
        ),
        "category": "work",
        "subcategory": "Go-разработка",
        "parent": "Бэкенд-разработка",
    },
    "go_gin": {
        "aliases": ["gin", "гин", "gin framework", "gin go"],
        "enriched_text": (
            "веб-фреймворк Gin на Go, высокая скорость, "
            "REST API, middleware, рендеринг ответов"
        ),
        "category": "work",
        "subcategory": "Go-разработка",
        "parent": "Бэкенд-разработка",
    },
    "go_fiber": {
        "aliases": ["fiber", "файбер", "fiber go", "фастихттп"],
        "enriched_text": (
            "веб-фреймворк Fiber на Go, вдохновлен Express.js, "
            "высокая производительность, удобная маршрутизация"
        ),
        "category": "work",
        "subcategory": "Go-разработка",
        "parent": "Бэкенд-разработка",
    },
    "go_microservices": {
        "aliases": ["микросервисы go", "микросервисы на go", "go микросервисы"],
        "enriched_text": (
            "построение микросервисной архитектуры на Go, "
            "gRPC, protobuf, событийно-ориентированные системы"
        ),
        "category": "work",
        "subcategory": "Go-разработка",
        "parent": "Бэкенд-разработка",
    },
    "go_echo": {
        "aliases": ["echo", "эхо", "echo framework", "echo go"],
        "enriched_text": (
            "высокопроизводительный веб-фреймворк Echo на Go, "
            "минималистичный API, встроенный middleware"
        ),
        "category": "work",
        "subcategory": "Go-разработка",
        "parent": "Бэкенд-разработка",
    },
    "go_goroutines": {
        "aliases": ["goroutines", "горутины", "горутина", "конкурентность go"],
        "enriched_text": (
            "горутины и конкурентное программирование на Go, "
            "каналы, паттерны параллелизма, легковесные потоки"
        ),
        "category": "work",
        "subcategory": "Go-разработка",
        "parent": "Бэкенд-разработка",
    },

    "backend_rust": {
        "aliases": ["rust", "раст", "rust backend", "ржавчина", "раст бэкенд"],
        "enriched_text": (
            "бэкенд-разработка на Rust, безопасность памяти, "
            "высокая производительность, системное программирование"
        ),
        "category": "work",
        "subcategory": "Rust-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "rust_actix": {
        "aliases": ["actix", "актикс", "actix web", "actix rust"],
        "enriched_text": (
            "мощный веб-фреймворк Actix Web на Rust, "
            "акторная модель, асинхронность, высокая пропускная способность"
        ),
        "category": "work",
        "subcategory": "Rust-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "rust_rocket": {
        "aliases": ["rocket", "рокет", "rocket rust", "rocket framework"],
        "enriched_text": (
            "эргономичный веб-фреймворк Rocket на Rust, "
            "простота использования, макросы, безопасные типы"
        ),
        "category": "work",
        "subcategory": "Rust-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "rust_tokio": {
        "aliases": ["tokio", "токио", "tokio rust", "асинхронный rust"],
        "enriched_text": (
            "асинхронная среда выполнения Tokio на Rust, "
            "event loop, асинхронный ввод-вывод, сетевое программирование"
        ),
        "category": "work",
        "subcategory": "Rust-бэкенд",
        "parent": "Бэкенд-разработка",
    },

    "backend_java": {
        "aliases": ["java", "джава", "java backend", "ява", "джава бэкенд"],
        "enriched_text": (
            "бэкенд-разработка на Java, Spring Boot, Hibernate, "
            "enterprise разработка, JVM, корпоративные системы"
        ),
        "category": "work",
        "subcategory": "Java-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "java_spring": {
        "aliases": ["spring", "спринг", "spring boot", "спринг бут"],
        "enriched_text": (
            "фреймворк Spring Boot на Java, внедрение зависимостей, "
            "автоконфигурация, создание микросервисов и веб-приложений"
        ),
        "category": "work",
        "subcategory": "Java-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "java_hibernate": {
        "aliases": ["hibernate", "хибернейт", "гибернейт", "hibernate orm"],
        "enriched_text": (
            "ORM Hibernate на Java, отображение объектов на реляционные БД, "
            "JPQL, кеширование, ленивая загрузка"
        ),
        "category": "work",
        "subcategory": "Java-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "java_micronaut": {
        "aliases": ["micronaut", "микронавт", "micronaut java"],
        "enriched_text": (
            "фреймворк Micronaut на Java, compile-time внедрение зависимостей, "
            "микросервисы, serverless, низкое потребление памяти"
        ),
        "category": "work",
        "subcategory": "Java-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "java_quarkus": {
        "aliases": ["quarkus", "кваркус", "quarkus java", "суперсоник"],
        "enriched_text": (
            "Kubernetes-native фреймворк Quarkus на Java, "
            "быстрый запуск, низкое потребление памяти, GraalVM"
        ),
        "category": "work",
        "subcategory": "Java-бэкенд",
        "parent": "Бэкенд-разработка",
    },

    "backend_kotlin": {
        "aliases": ["kotlin", "котлин", "котлин бэкенд", "kotlin backend"],
        "enriched_text": (
            "бэкенд-разработка на Kotlin, современный язык под JVM, "
            "Ktor, Spring Boot Kotlin, корутины и асинхронность"
        ),
        "category": "work",
        "subcategory": "Kotlin-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "kotlin_ktor": {
        "aliases": ["ktor", "ктор", "ktor kotlin", "ктор фреймворк"],
        "enriched_text": (
            "легковесный веб-фреймворк Ktor на Kotlin, "
            "корутины, встроенный клиент и сервер, DSL маршрутизация"
        ),
        "category": "work",
        "subcategory": "Kotlin-бэкенд",
        "parent": "Бэкенд-разработка",
    },
    "kotlin_spring_boot": {
        "aliases": ["spring kotlin", "spring boot котлин", "спринг на котлине"],
        "enriched_text": (
            "использование Spring Boot на Kotlin, "
            "корутины, реактивные потоки, функциональные эндпоинты"
        ),
        "category": "work",
        "subcategory": "Kotlin-бэкенд",
        "parent": "Бэкенд-разработка",
    },

    "api_design": {
        "aliases": ["api дизайн", "проектирование api", "API", "апи", "эндпоинты"],
        "enriched_text": (
            "проектирование API, REST, GraphQL, WebSocket и gRPC, "
            "безопасность и аутентификация, документация и версионирование"
        ),
        "category": "work",
        "subcategory": "API Дизайн",
        "parent": "Бэкенд-разработка",
    },
    "rest_api": {
        "aliases": ["rest", "rest api", "рест", "рест апи", "RESTful"],
        "enriched_text": (
            "REST API архитектура, HTTP методы GET POST PUT DELETE, "
            "ресурсы, статус-коды, HATEOAS, пагинация"
        ),
        "category": "work",
        "subcategory": "API Дизайн",
        "parent": "Бэкенд-разработка",
    },
    "graphql_api": {
        "aliases": ["graphql", "графкюэль", "гкл", "граф ql"],
        "enriched_text": (
            "GraphQL API, язык запросов к данным, схема, резолверы, "
            "федерация, подписки в реальном времени"
        ),
        "category": "work",
        "subcategory": "API Дизайн",
        "parent": "Бэкенд-разработка",
    },
    "websocket_api": {
        "aliases": ["websocket", "вебсокет", "ws", "веб сокеты", "wss"],
        "enriched_text": (
            "протокол WebSocket, двусторонняя связь клиент-сервер, "
            "реалтайм приложения, чаты, уведомления, стриминг"
        ),
        "category": "work",
        "subcategory": "API Дизайн",
        "parent": "Бэкенд-разработка",
    },
    "grpc_api": {
        "aliases": ["grpc", "грпс", "grpc api", "protobuf", "протобаф"],
        "enriched_text": (
            "gRPC фреймворк для удаленного вызова процедур, "
            "protobuf, бинарный протокол, стриминг, микросервисы"
        ),
        "category": "work",
        "subcategory": "API Дизайн",
        "parent": "Бэкенд-разработка",
    },
    "api_security": {
        "aliases": ["безопасность api", "api security", "аутентификация", "авторизация"],
        "enriched_text": (
            "безопасность API, защита от атак, rate limiting, "
            "CORS, валидация входящих данных, OWASP API Security"
        ),
        "category": "work",
        "subcategory": "API Дизайн",
        "parent": "Бэкенд-разработка",
    },
    "oauth2_jwt": {
        "aliases": ["oauth2", "jwt", "джвт", "токены", "оаут", "openid"],
        "enriched_text": (
            "протокол OAuth2 и JWT токены, авторизация и аутентификация, "
            "refresh token, доступ к ресурсам через токены доступа"
        ),
        "category": "work",
        "subcategory": "API Дизайн",
        "parent": "Бэкенд-разработка",
    },

    "frontend_dev": {
        "aliases": ["фронтенд", "frontend", "фронт", "клиентская часть", "вёрстка", "UI"],
        "enriched_text": (
            "фронтенд-разработка, создание пользовательских интерфейсов, "
            "React, Vue, Angular, JavaScript, TypeScript, верстка и анимации"
        ),
        "category": "work",
        "subcategory": "Фронтенд-разработка",
        "parent": "IT и Разработка",
    },
    "frontend_react": {
        "aliases": ["react", "реакт", "react.js", "реакт js", "reactjs"],
        "enriched_text": (
            "React-экосистема, компонентный подход, виртуальный DOM, "
            "JSX, хуки, управление состоянием, одностраничные приложения"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_hooks": {
        "aliases": ["react hooks", "хуки", "useState", "useEffect", "кастомные хуки"],
        "enriched_text": (
            "React Hooks, функциональные компоненты, управление состоянием, "
            "побочные эффекты, переиспользование логики, контекст"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_nextjs": {
        "aliases": ["nextjs", "next.js", "некст", "некст js", "next", "ssr react"],
        "enriched_text": (
            "фреймворк Next.js на React, серверный рендеринг, "
            "статическая генерация, маршрутизация на основе файлов"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_redux": {
        "aliases": ["redux", "редакс", "redux toolkit", "редакс тулкит", "стейт менеджмент"],
        "enriched_text": (
            "библиотека управления состоянием Redux на React, "
            "стор, экшены, редюсеры, Redux Toolkit, иммутабельный стейт"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_zustand": {
        "aliases": ["zustand", "зустанд", "zustand state", "легкий стейт менеджер"],
        "enriched_text": (
            "легковесная библиотека управления состоянием Zustand для React, "
            "хуки, минималистичный API, без провайдеров и контекста"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_react_native": {
        "aliases": ["react native", "реакт нейтив", "мобильная разработка react", "rn"],
        "enriched_text": (
            "React Native, разработка мобильных приложений на React, "
            "кроссплатформенность iOS и Android, нативные компоненты"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_tailwind": {
        "aliases": ["tailwind", "тайлвинд", "tailwind css", "утилитарный css"],
        "enriched_text": (
            "CSS-фреймворк Tailwind, утилитарные классы, "
            "быстрая стилизация, кастомизация дизайна, адаптивная верстка"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_mui": {
        "aliases": ["material ui", "mui", "материал юай", "material design react"],
        "enriched_text": (
            "библиотека компонентов Material UI для React, "
            "дизайн-система Google Material, готовые UI элементы"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_threejs": {
        "aliases": ["threejs", "three.js", "3d react", "react fiber", "react three fiber"],
        "enriched_text": (
            "Three.js и React Three Fiber, 3D-графика в браузере, "
            "WebGL, визуализации, интерактивные 3D-сцены"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "react_remix": {
        "aliases": ["remix", "ремикс", "remix.run", "remix react"],
        "enriched_text": (
            "фреймворк Remix на React, прогрессивное улучшение, "
            "серверный рендеринг, вложенная маршрутизация, формы"
        ),
        "category": "work",
        "subcategory": "React-экосистема",
        "parent": "Фронтенд-разработка",
    },

    "frontend_vue": {
        "aliases": ["vue", "вью", "vue.js", "вью js", "vuejs"],
        "enriched_text": (
            "Vue.js экосистема, реактивный фреймворк, компоненты, "
            "директивы, Composition API, однофайловые компоненты"
        ),
        "category": "work",
        "subcategory": "Vue.js-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "vue_nuxt": {
        "aliases": ["nuxt", "нюкст", "nuxt.js", "nuxt 3", "ssr vue"],
        "enriched_text": (
            "фреймворк Nuxt.js на Vue, серверный рендеринг, "
            "статическая генерация, автоимпорты, модульная архитектура"
        ),
        "category": "work",
        "subcategory": "Vue.js-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "vue_pinia": {
        "aliases": ["pinia", "пиния", "pinia store", "стейт менеджер vue"],
        "enriched_text": (
            "официальная библиотека управления состоянием Pinia для Vue, "
            "модульные сторы, поддержка TypeScript, devtools"
        ),
        "category": "work",
        "subcategory": "Vue.js-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "vue_composition_api": {
        "aliases": ["composition api", "композишн апи", "setup", "ref", "reactive"],
        "enriched_text": (
            "Composition API во Vue, функциональное описание компонентов, "
            "переиспользование логики, реактивные ссылки и вычисляемые свойства"
        ),
        "category": "work",
        "subcategory": "Vue.js-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "vue_vuetify": {
        "aliases": ["vuetify", "вуетифай", "material design vue"],
        "enriched_text": (
            "библиотека компонентов Vuetify для Vue, "
            "Material Design, готовые UI-элементы, сетки, темы"
        ),
        "category": "work",
        "subcategory": "Vue.js-экосистема",
        "parent": "Фронтенд-разработка",
    },
    "vue_quasar": {
        "aliases": ["quasar", "квазар", "quasar framework", "quasar vue"],
        "enriched_text": (
            "фреймворк Quasar на Vue, создание SPA, PWA, SSR, "
            "мобильные и десктопные приложения из единой кодовой базы"
        ),
        "category": "work",
        "subcategory": "Vue.js-экосистема",
        "parent": "Фронтенд-разработка",
    },

    "frontend_angular": {
        "aliases": ["angular", "ангуляр", "angular 2+", "нгуляр"],
        "enriched_text": (
            "фронтенд-фреймворк Angular, TypeScript, модули, "
            "внедрение зависимостей, RxJS реактивное программирование"
        ),
        "category": "work",
        "subcategory": "Angular",
        "parent": "Фронтенд-разработка",
    },
    "angular_rxjs": {
        "aliases": ["rxjs", "реактивные потоки", "observable", "subject", "rxjs angular"],
        "enriched_text": (
            "библиотека RxJS для реактивного программирования, "
            "Observable, операторы, подписки, асинхронные потоки данных"
        ),
        "category": "work",
        "subcategory": "Angular",
        "parent": "Фронтенд-разработка",
    },
    "angular_material": {
        "aliases": ["angular material", "материал ангуляр", "material angular"],
        "enriched_text": (
            "библиотека компонентов Angular Material, "
            "Material Design для Angular, готовые UI-компоненты"
        ),
        "category": "work",
        "subcategory": "Angular",
        "parent": "Фронтенд-разработка",
    },

    "frontend_svelte": {
        "aliases": ["svelte", "свелте", "свелт", "svelte js"],
        "enriched_text": (
            "фреймворк Svelte, компиляция в чистый JavaScript, "
            "отсутствие виртуального DOM, реактивность на уровне языка"
        ),
        "category": "work",
        "subcategory": "Svelte",
        "parent": "Фронтенд-разработка",
    },
    "svelte_kit": {
        "aliases": ["sveltekit", "свелтекит", "svelte kit", "свелт кит"],
        "enriched_text": (
            "фреймворк SvelteKit, серверный рендеринг, маршрутизация, "
            "адаптеры для разных платформ, статическая и серверная генерация"
        ),
        "category": "work",
        "subcategory": "Svelte",
        "parent": "Фронтенд-разработка",
    },
    "svelte_stores": {
        "aliases": ["svelte stores", "сторы свелте", "writable", "derived"],
        "enriched_text": (
            "управление состоянием в Svelte через stores, "
            "реактивные хранилища данных, подписки и обновления"
        ),
        "category": "work",
        "subcategory": "Svelte",
        "parent": "Фронтенд-разработка",
    },
    "frontend_astro": {
        "aliases": ["astro", "астро", "астро js", "astro.build"],
        "enriched_text": (
            "статический сайт-генератор Astro, островная архитектура, "
            "отправка нуля JS по умолчанию, поддержка любых UI-фреймворков"
        ),
        "category": "work",
        "subcategory": "Фронтенд-разработка",
        "parent": "Фронтенд-разработка",
    },
    "frontend_solidjs": {
        "aliases": ["solidjs", "solid", "солид", "солид js"],
        "enriched_text": (
            "реактивный фреймворк SolidJS, мелкозернистая реактивность, "
            "компиляция в DOM, без виртуального DOM, высокая производительность"
        ),
        "category": "work",
        "subcategory": "Фронтенд-разработка",
        "parent": "Фронтенд-разработка",
    },

    "frontend_js_ts": {
        "aliases": ["javascript", "typescript", "js", "ts", "джаваскрипт", "тайпскрипт"],
        "enriched_text": (
            "языки JavaScript и TypeScript, основы веб-разработки, "
            "типизация, ES6+, сборка и транспиляция кода"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },
    "typescript_advanced": {
        "aliases": ["ts advanced", "advanced typescript", "продвинутый тайпскрипт"],
        "enriched_text": (
            "продвинутый TypeScript, utility types, conditional types, "
            "infer, mapped types, шаблонные литералы типов, дженерики"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },
    "js_esbuild": {
        "aliases": ["esbuild", "ебилд", "esbuild bundler", "быстрый сборщик"],
        "enriched_text": (
            "сверхбыстрый сборщик ESBuild на Go, транспиляция JS/TS, "
            "минификация, бандлинг для продакшена"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },
    "js_vite": {
        "aliases": ["vite", "вит", "vite js", "сборщик vite"],
        "enriched_text": (
            "сборщик Vite, мгновенный HMR, нативный ESM, "
            "оптимизированная сборка для продакшена через Rollup"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },
    "js_testing": {
        "aliases": ["тестирование фронтенд", "фронтенд тесты", "js tests", "unit tests"],
        "enriched_text": (
            "тестирование фронтенд-приложений, модульные и интеграционные тесты, "
            "Jest, Vitest, Cypress, Playwright, моки и стабы"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },
    "js_jest": {
        "aliases": ["jest", "джест", "jest js", "тесты джест"],
        "enriched_text": (
            "фреймворк тестирования Jest, снэпшоты, моки, "
            "покрытие кода, assertions, асинхронное тестирование"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },
    "js_vitest": {
        "aliases": ["vitest", "витест", "vitest js", "vite test"],
        "enriched_text": (
            "фреймворк тестирования Vitest, совместим с Vite, "
            "быстрые тесты, поддержка ESM, интеграция с Jest API"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },
    "js_cypress": {
        "aliases": ["cypress", "сайпресс", "cypress тесты", "e2e cypress"],
        "enriched_text": (
            "инструмент e2e тестирования Cypress, запуск в браузере, "
            "визуальный отладчик, перехват сетевых запросов"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },
    "js_playwright": {
        "aliases": ["playwright", "плейрайт", "playwright тесты", "кроссбраузерные тесты"],
        "enriched_text": (
            "инструмент e2e тестирования Playwright от Microsoft, "
            "автоматизация браузеров, параллельный запуск, мобильная эмуляция"
        ),
        "category": "work",
        "subcategory": "JavaScript / TypeScript",
        "parent": "Фронтенд-разработка",
    },

    "web_performance": {
        "aliases": ["производительность", "core web vitals", "lighthouse", "оптимизация"],
        "enriched_text": (
            "веб-производительность, оптимизация загрузки страниц, "
            "Core Web Vitals, аудит Lighthouse, lazy loading, бандлинг"
        ),
        "category": "work",
        "subcategory": "Веб-производительность",
        "parent": "Фронтенд-разработка",
    },
    "core_web_vitals": {
        "aliases": ["web vitals", "cws", "метрики производительности", "LCP", "FID", "CLS"],
        "enriched_text": (
            "метрики Core Web Vitals от Google, LCP скорость загрузки, "
            "FID задержка взаимодействия, CLS стабильность макета"
        ),
        "category": "work",
        "subcategory": "Веб-производительность",
        "parent": "Фронтенд-разработка",
    },
    "lighthouse_audit": {
        "aliases": ["lighthouse", "лайтхаус", "аудит сайта", "google lighthouse"],
        "enriched_text": (
            "инструмент аудита Lighthouse, проверка производительности, "
            "доступности, SEO и лучших практик для веб-страниц"
        ),
        "category": "work",
        "subcategory": "Веб-производительность",
        "parent": "Фронтенд-разработка",
    },

    "web_animations": {
        "aliases": ["веб анимации", "анимация", "микроанимации", "параллакс"],
        "enriched_text": (
            "веб-анимации, создание плавных переходов, GSAP и Framer Motion, "
            "анимация интерфейсов и интерактивных элементов"
        ),
        "category": "work",
        "subcategory": "Веб-анимации",
        "parent": "Фронтенд-разработка",
    },
    "gsap_animation": {
        "aliases": ["gsap", "гсап", "greensock", "анимация gsap"],
        "enriched_text": (
            "библиотека GSAP (GreenSock) для веб-анимаций, "
            "таймлайны, плавная анимация SVG и DOM элементов"
        ),
        "category": "work",
        "subcategory": "Веб-анимации",
        "parent": "Фронтенд-разработка",
    },
    "framer_motion": {
        "aliases": ["framer motion", "фреймер моушн", "анимация react", "motion"],
        "enriched_text": (
            "библиотека анимаций Framer Motion для React, "
            "декларативные анимации, жесты, layout-анимации"
        ),
        "category": "work",
        "subcategory": "Веб-анимации",
        "parent": "Фронтенд-разработка",
    },
    "css_art": {
        "aliases": ["css art", "css рисунки", "арт на css", "чистый css"],
        "enriched_text": (
            "создание иллюстраций и арта на чистом CSS, "
            "креативная верстка, градиенты, тени, трансформации"
        ),
        "category": "work",
        "subcategory": "Фронтенд-разработка",
        "parent": "Фронтенд-разработка",
    },
    "web_accessibility": {
        "aliases": ["a11y", "доступность", "web accessibility", "aria", "инклюзивность"],
        "enriched_text": (
            "веб-доступность (a11y), создание интерфейсов для всех, "
            "ARIA атрибуты, семантический HTML, скринридеры, WCAG"
        ),
        "category": "work",
        "subcategory": "Фронтенд-разработка",
        "parent": "Фронтенд-разработка",
    },

    "devops": {
        "aliases": ["devops", "девопс", "инфраструктура", "деплой", "sre", "инфра"],
        "enriched_text": (
            "DevOps и инфраструктура, автоматизация деплоя, CI/CD, "
            "облачные технологии, контейнеризация, мониторинг и наблюдаемость"
        ),
        "category": "work",
        "subcategory": "DevOps и Инфраструктура",
        "parent": "IT и Разработка",
    },
    "docker_kubernetes": {
        "aliases": ["docker", "kubernetes", "k8s", "докер", "кубернетес", "кубер"],
        "enriched_text": (
            "контейнеризация Docker и оркестрация Kubernetes, "
            "упаковка приложений в контейнеры, управление кластерами"
        ),
        "category": "work",
        "subcategory": "Docker и K8s",
        "parent": "DevOps и Инфраструктура",
    },
    "docker_compose": {
        "aliases": ["docker compose", "докер компоуз", "docker-compose", "компоуз"],
        "enriched_text": (
            "Docker Compose, описание многоконтейнерных приложений, "
            "локальная разработка с сервисами, сети и тома"
        ),
        "category": "work",
        "subcategory": "Docker и K8s",
        "parent": "DevOps и Инфраструктура",
    },
    "docker_security": {
        "aliases": ["безопасность докер", "docker security", "безопасность контейнеров"],
        "enriched_text": (
            "безопасность Docker-контейнеров, сканирование образов, "
            "привилегии, секреты, неподписанные образы"
        ),
        "category": "work",
        "subcategory": "Docker и K8s",
        "parent": "DevOps и Инфраструктура",
    },
    "kubernetes_cluster": {
        "aliases": ["kubernetes", "кубернетес", "k8s кластер", "кубер кластер"],
        "enriched_text": (
            "управление кластером Kubernetes, поды, сервисы, "
            "деплойменты, масштабирование, self-healing"
        ),
        "category": "work",
        "subcategory": "Docker и K8s",
        "parent": "DevOps и Инфраструктура",
    },
    "helm_charts": {
        "aliases": ["helm", "хелм", "helm charts", "чарты", "пакетный менеджер k8s"],
        "enriched_text": (
            "менеджер пакетов Helm для Kubernetes, шаблонизация манифестов, "
            "управление релизами, rollback, values"
        ),
        "category": "work",
        "subcategory": "Docker и K8s",
        "parent": "DevOps и Инфраструктура",
    },
    "istio_mesh": {
        "aliases": ["istio", "истио", "service mesh", "сервис меш", "трафик"],
        "enriched_text": (
            "Istio Service Mesh для Kubernetes, управление трафиком, "
            "балансировка, observability, mTLS, канареечные деплои"
        ),
        "category": "work",
        "subcategory": "Docker и K8s",
        "parent": "DevOps и Инфраструктура",
    },

    "ci_cd": {
        "aliases": ["ci/cd", "пайплайн", "pipeline", "автоматизация сборки", "деплоймент"],
        "enriched_text": (
            "непрерывная интеграция и доставка CI/CD, GitHub Actions, "
            "GitLab CI, Jenkins, автоматическое тестирование и деплой"
        ),
        "category": "work",
        "subcategory": "CI/CD Пайплайны",
        "parent": "DevOps и Инфраструктура",
    },
    "github_actions": {
        "aliases": ["github actions", "гх экшенс", "actions", "воркфлоу github"],
        "enriched_text": (
            "CI/CD платформа GitHub Actions, автоматизация рабочих процессов, "
            "воркфлоу, джобы, шаги, интеграция с репозиторием GitHub"
        ),
        "category": "work",
        "subcategory": "CI/CD Пайплайны",
        "parent": "DevOps и Инфраструктура",
    },
    "gitlab_ci": {
        "aliases": ["gitlab ci", "гитлаб сиай", "gitlab-ci", "ci gitlab"],
        "enriched_text": (
            "встроенный CI/CD в GitLab, .gitlab-ci.yml, раннеры, "
            "пайплайны, артефакты, окружения и деплой"
        ),
        "category": "work",
        "subcategory": "CI/CD Пайплайны",
        "parent": "DevOps и Инфраструктура",
    },
    "jenkins": {
        "aliases": ["jenkins", "дженкинс", "дженкинс пайплайн", "groovy ci"],
        "enriched_text": (
            "сервер автоматизации Jenkins, пайплайны на Groovy, "
            "расширяемая экосистема плагинов, распределенные сборки"
        ),
        "category": "work",
        "subcategory": "CI/CD Пайплайны",
        "parent": "DevOps и Инфраструктура",
    },
    "argocd": {
        "aliases": ["argocd", "арго сиди", "gitops", "гитопс", "argo cd"],
        "enriched_text": (
            "инструмент ArgoCD для GitOps в Kubernetes, "
            "синхронизация состояния кластера с Git-репозиторием"
        ),
        "category": "work",
        "subcategory": "CI/CD Пайплайны",
        "parent": "DevOps и Инфраструктура",
    },
    "circleci": {
        "aliases": ["circleci", "серкл сиай", "circle ci"],
        "enriched_text": (
            "облачная CI/CD платформа CircleCI, быстрые сборки, "
            "кеширование, параллелизм, Docker-окружения"
        ),
        "category": "work",
        "subcategory": "CI/CD Пайплайны",
        "parent": "DevOps и Инфраструктура",
    },

    "cloud_providers": {
        "aliases": ["облака", "облачные провайдеры", "aws", "gcp", "azure", "cloud"],
        "enriched_text": (
            "облачные платформы AWS, GCP, Azure, инфраструктура как сервис, "
            "серверлесс вычисления, хранение данных, масштабирование"
        ),
        "category": "work",
        "subcategory": "Облачные платформы",
        "parent": "DevOps и Инфраструктура",
    },
    "aws_cloud": {
        "aliases": ["aws", "амазон", "amazon web services", "авс"],
        "enriched_text": (
            "облачная платформа Amazon Web Services, EC2, S3, "
            "Lambda, IAM, широкий выбор облачных сервисов"
        ),
        "category": "work",
        "subcategory": "Облачные платформы",
        "parent": "DevOps и Инфраструктура",
    },
    "aws_lambda": {
        "aliases": ["aws lambda", "лямбда", "lambda aws", "серверлесс функции"],
        "enriched_text": (
            "бессерверные вычисления AWS Lambda, запуск кода без управления серверами, "
            "событийно-ориентированная архитектура, масштабирование по требованию"
        ),
        "category": "work",
        "subcategory": "Облачные платформы",
        "parent": "DevOps и Инфраструктура",
    },
    "aws_s3": {
        "aliases": ["s3", "aws s3", "эс3", "simple storage service", "объектное хранилище"],
        "enriched_text": (
            "объектное хранилище AWS S3, хранение статических файлов, "
            "бакеты, версионирование, хостинг статических сайтов"
        ),
        "category": "work",
        "subcategory": "Облачные платформы",
        "parent": "DevOps и Инфраструктура",
    },
    "gcp_cloud": {
        "aliases": ["gcp", "google cloud", "гугл облако", "google cloud platform"],
        "enriched_text": (
            "облачная платформа Google Cloud Platform, Compute Engine, "
            "BigQuery, Cloud Run, интеграция с экосистемой Google"
        ),
        "category": "work",
        "subcategory": "Облачные платформы",
        "parent": "DevOps и Инфраструктура",
    },
    "gcp_cloud_run": {
        "aliases": ["cloud run", "google cloud run", "клоуд ран", "серверлесс gcp"],
        "enriched_text": (
            "сервис Google Cloud Run, запуск контейнеров в бессерверном режиме, "
            "автомасштабирование, оплата за использование"
        ),
        "category": "work",
        "subcategory": "Облачные платформы",
        "parent": "DevOps и Инфраструктура",
    },
    "azure_cloud": {
        "aliases": ["azure", "азур", "майкрософт облако", "microsoft azure"],
        "enriched_text": (
            "облачная платформа Microsoft Azure, виртуальные машины, "
            "Azure DevOps, Functions, интеграция с продуктами Microsoft"
        ),
        "category": "work",
        "subcategory": "Облачные платформы",
        "parent": "DevOps и Инфраструктура",
    },
    "azure_functions": {
        "aliases": ["azure functions", "функции azure", "азур функции", "серверлесс майкрософт"],
        "enriched_text": (
            "бессерверные функции Azure Functions, "
            "событийно-ориентированное выполнение кода в облаке Microsoft"
        ),
        "category": "work",
        "subcategory": "Облачные платформы",
        "parent": "DevOps и Инфраструктура",
    },

    "terraform_iac": {
        "aliases": ["terraform", "терраформ", "iac", "инфраструктура как код", "tf"],
        "enriched_text": (
            "Terraform для управления инфраструктурой как кодом, "
            "HCL, провайдеры, модули, состояние, план и применение"
        ),
        "category": "work",
        "subcategory": "Terraform IaC",
        "parent": "DevOps и Инфраструктура",
    },
    "pulumi_iac": {
        "aliases": ["pulumi", "пулуми", "iac на языках", "пулуми iac"],
        "enriched_text": (
            "инструмент Pulumi для инфраструктуры как код, "
            "использование Python, TypeScript, Go для описания облачных ресурсов"
        ),
        "category": "work",
        "subcategory": "DevOps и Инфраструктура",
        "parent": "DevOps и Инфраструктура",
    },

    "observability": {
        "aliases": ["наблюдаемость", "мониторинг", "трейсинг", "логирование", "алертинг"],
        "enriched_text": (
            "наблюдаемость систем, мониторинг Prometheus, Grafana, "
            "логирование ELK, распределенный трейсинг Jaeger"
        ),
        "category": "work",
        "subcategory": "Наблюдаемость",
        "parent": "DevOps и Инфраструктура",
    },
    "prometheus_grafana": {
        "aliases": ["prometheus", "grafana", "прометеус", "графана", "дашборды"],
        "enriched_text": (
            "связка мониторинга Prometheus + Grafana, сбор метрик, "
            "алертинг, визуализация и дашборды для инфраструктуры"
        ),
        "category": "work",
        "subcategory": "Наблюдаемость",
        "parent": "DevOps и Инфраструктура",
    },
    "elk_stack": {
        "aliases": ["elk", "elasticsearch", "logstash", "kibana", "елк стэк", "эластик"],
        "enriched_text": (
            "стек ELK: Elasticsearch, Logstash, Kibana, "
            "сбор и анализ логов, поиск, визуализация данных"
        ),
        "category": "work",
        "subcategory": "Наблюдаемость",
        "parent": "DevOps и Инфраструктура",
    },
    "jaeger_tracing": {
        "aliases": ["jaeger", "егер", "трейсинг", "distributed tracing", "распределенный трейсинг"],
        "enriched_text": (
            "распределенный трейсинг Jaeger, отслеживание запросов в микросервисах, "
            "анализ задержек, OpenTelemetry"
        ),
        "category": "work",
        "subcategory": "Наблюдаемость",
        "parent": "DevOps и Инфраструктура",
    },
    "chaos_engineering": {
        "aliases": ["chaos engineering", "хаос инжиниринг", "chaos monkey", "отказоустойчивость"],
        "enriched_text": (
            "Chaos Engineering, тестирование устойчивости систем, "
            "имитация сбоев и отказов, проверка механизмов восстановления"
        ),
        "category": "work",
        "subcategory": "DevOps и Инфраструктура",
        "parent": "DevOps и Инфраструктура",
    },

    "data_science_ml": {
        "aliases": ["data science", "ml", "ии", "ai", "нейросети", "машинное обучение"],
        "enriched_text": (
            "Data Science и машинное обучение, анализ данных, "
            "нейронные сети, NLP, компьютерное зрение, MLOps"
        ),
        "category": "work",
        "subcategory": "Data Science и ML",
        "parent": "IT и Разработка",
    },
    "machine_learning": {
        "aliases": ["ml", "машинное обучение", "модели", "обучение", "алгоритмы"],
        "enriched_text": (
            "машинное обучение, PyTorch, TensorFlow, Scikit-learn, "
            "глубокое обучение, градиентный бустинг, ансамбли моделей"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "pytorch_ecosystem": {
        "aliases": ["pytorch", "пайторч", "торч", "torch", "pytorch lightning"],
        "enriched_text": (
            "фреймворк глубокого обучения PyTorch, динамические графы, "
            "автодифференцирование, обучение нейросетей на GPU"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "pytorch_lightning": {
        "aliases": ["pytorch lightning", "лайтнинг", "lightning", "pl"],
        "enriched_text": (
            "обертка PyTorch Lightning для структурирования кода обучения, "
            "Trainer, логирование, чекпоинты, уменьшение boilerplate"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "tensorflow_keras": {
        "aliases": ["tensorflow", "keras", "тензорфлоу", "керос", "tf"],
        "enriched_text": (
            "фреймворки TensorFlow и Keras для машинного обучения, "
            "статический граф, высокоуровневый API Keras"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "scikit_learn": {
        "aliases": ["scikit-learn", "скикит", "склерн", "sklearn", "скикит лерн"],
        "enriched_text": (
            "библиотека Scikit-learn для классического машинного обучения, "
            "классификация, регрессия, кластеризация, предобработка данных"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "xgboost_ensemble": {
        "aliases": ["xgboost", "хгбуст", "бустинг", "ансамбли", "градиентный бустинг"],
        "enriched_text": (
            "алгоритм XGBoost и ансамблевые методы, градиентный бустинг, "
            "соревновательное машинное обучение на табличных данных"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "catboost": {
        "aliases": ["catboost", "кэтбуст", "катбуст", "яндекс бустинг"],
        "enriched_text": (
            "библиотека градиентного бустинга CatBoost от Яндекса, "
            "автоматическая обработка категориальных признаков"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "ml_explainability": {
        "aliases": ["xai", "объяснимость", "интерпретация моделей", "shap", "lime"],
        "enriched_text": (
            "объяснимость моделей ML, SHAP и LIME, интерпретация предсказаний, "
            "feature importance, trustworthy AI"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "shap_lime": {
        "aliases": ["shap", "lime", "шап", "лайм", "объяснение предсказаний"],
        "enriched_text": (
            "инструменты SHAP и LIME для интерпретации моделей, "
            "вклад признаков в предсказание, визуализация влияния"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "deep_learning": {
        "aliases": ["глубокое обучение", "deep learning", "нейросети", "свертки", "рекуррентные"],
        "enriched_text": (
            "глубокое обучение, нейронные сети CNN, RNN, LSTM, GAN, "
            "архитектуры глубоких сетей, обучение представлений"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "cnn_networks": {
        "aliases": ["cnn", "сверточные сети", "convnets", "resnet", "vgg", "efficientnet"],
        "enriched_text": (
            "сверточные нейронные сети CNN, обработка изображений, "
            "ResNet, VGG, EfficientNet, свертки и пулинг"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "rnn_lstm": {
        "aliases": ["rnn", "lstm", "рекуррентные сети", "gru", "последовательности"],
        "enriched_text": (
            "рекуррентные нейронные сети RNN и LSTM, работа с последовательностями, "
            "обработка текста, временные ряды, управляемые блоки памяти"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },
    "gan_networks": {
        "aliases": ["gan", "ган", "генеративно-состязательные", "генеративные сети"],
        "enriched_text": (
            "генеративно-состязательные сети GAN, генерация изображений, "
            "стилизация, генератор и дискриминатор"
        ),
        "category": "work",
        "subcategory": "Машинное обучение",
        "parent": "Data Science и ML",
    },

    "nlp_natural_language": {
        "aliases": ["nlp", "обработка текста", "языковые модели", "nlp инжиниринг"],
        "enriched_text": (
            "Natural Language Processing, обработка естественного языка, "
            "трансформеры, BERT, LLM, анализ тональности, RAG"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "transformers_bert": {
        "aliases": ["transformers", "берт", "bert", "трансформеры", "hugging face"],
        "enriched_text": (
            "модели трансформеров BERT и библиотека Hugging Face, "
            "предобученные языковые модели, fine-tuning для NLP задач"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "huggingface_hub": {
        "aliases": ["hugging face", "хагинг фейс", "huggingface", "модели hf"],
        "enriched_text": (
            "платформа Hugging Face Hub, репозиторий моделей и датасетов, "
            "загрузка и публикация предобученных NLP-моделей"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "llm_large_models": {
        "aliases": ["llm", "большие языковые модели", "gpt", "llama", "mistral", "генеративный ии"],
        "enriched_text": (
            "большие языковые модели LLM, GPT, Llama, Mistral, "
            "генеративный AI, prompt engineering, фаин-тьюнинг"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "llm_finetuning": {
        "aliases": ["фаин-тьюнинг", "fine-tuning", "лора", "lora", "q-lora", "дотюнинг"],
        "enriched_text": (
            "фаин-тьюнинг языковых моделей, LoRA и QLoRA адаптеры, "
            "дообучение LLM на специфичных данных"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "prompt_engineering": {
        "aliases": ["prompt engineering", "промпт инжиниринг", "цепочки промптов", "few-shot"],
        "enriched_text": (
            "инженерия промптов для LLM, цепочки рассуждений, "
            "few-shot и zero-shot подсказки, управление поведением модели"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "spacy_nltk": {
        "aliases": ["spacy", "спейси", "nltk", "лингвистическая обработка"],
        "enriched_text": (
            "библиотеки spaCy и NLTK для NLP, токенизация, POS-тэггинг, "
            "NER извлечение сущностей, синтаксический разбор"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "sentiment_analysis": {
        "aliases": ["анализ тональности", "sentiment", "тональность текста", "эмоциональный окрас"],
        "enriched_text": (
            "анализ тональности текста, определение позитивного/негативного окраса, "
            "эмоциональная классификация отзывов и комментариев"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "rag_retrieval": {
        "aliases": ["rag", "retrieval augmented generation", "векторный поиск", "семантический поиск"],
        "enriched_text": (
            "RAG, поисково-дополненная генерация, LangChain, "
            "векторные базы данных, семантический поиск и ответы на вопросы"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "langchain_llamaindex": {
        "aliases": ["langchain", "лангчейн", "llamaindex", "ламаиндекс", "оркестрация llm"],
        "enriched_text": (
            "фреймворки LangChain и LlamaIndex для оркестрации LLM, "
            "цепочки вызовов, агенты, индексация документов"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },
    "vector_search": {
        "aliases": ["vector search", "векторный поиск", "эмбеддинги", "embeddings", "сходство"],
        "enriched_text": (
            "векторный поиск по эмбеддингам, косинусное сходство, "
            "семантический поиск, поиск похожих документов"
        ),
        "category": "work",
        "subcategory": "NLP и Текст",
        "parent": "Data Science и ML",
    },

    "computer_vision": {
        "aliases": ["computer vision", "компьютерное зрение", "cv", "обработка изображений"],
        "enriched_text": (
            "компьютерное зрение, OpenCV, распознавание объектов YOLO, "
            "генерация изображений, классификация и сегментация"
        ),
        "category": "work",
        "subcategory": "Computer Vision",
        "parent": "Data Science и ML",
    },
    "opencv_image": {
        "aliases": ["opencv", "оупенсив", "cv2", "обработка видео"],
        "enriched_text": (
            "библиотека OpenCV для компьютерного зрения, фильтрация изображений, "
            "детекция объектов, работа с видеопотоком"
        ),
        "category": "work",
        "subcategory": "Computer Vision",
        "parent": "Data Science и ML",
    },
    "image_generation": {
        "aliases": ["генерация изображений", "диффузионки", "stable diffusion", "нейроарт"],
        "enriched_text": (
            "генерация изображений нейросетями, Stable Diffusion, "
            "DALL-E, диффузионные модели, генеративный AI для картинок"
        ),
        "category": "work",
        "subcategory": "Computer Vision",
        "parent": "Data Science и ML",
    },
    "stable_diffusion_cv": {
        "aliases": ["stable diffusion", "стейбл диффьюжн", "sd", "automatic1111", "comfyui"],
        "enriched_text": (
            "модель Stable Diffusion для генерации изображений по тексту, "
            "img2img, inpainting, LoRA для стилей, ControlNet"
        ),
        "category": "work",
        "subcategory": "Computer Vision",
        "parent": "Data Science и ML",
    },
    "object_detection": {
        "aliases": ["object detection", "yolo", "детекция объектов", "йоло"],
        "enriched_text": (
            "детекция объектов на изображениях YOLO, bounding boxes, "
            "сегментация, real-time распознавание в видеопотоке"
        ),
        "category": "work",
        "subcategory": "Computer Vision",
        "parent": "Data Science и ML",
    },

    "graph_theory_ml": {
        "aliases": ["графы", "graph ml", "gnn", "графовые сети", "networkx"],
        "enriched_text": (
            "графовые методы в ML, GNN графовые нейросети, "
            "NetworkX, iGraph, графовые базы данных Neo4j"
        ),
        "category": "work",
        "subcategory": "Графовые методы",
        "parent": "Data Science и ML",
    },
    "graph_networks_gnn": {
        "aliases": ["gnn", "графовые нейросети", "graph neural networks", "graphsage", "gat"],
        "enriched_text": (
            "графовые нейронные сети GNN, GraphSAGE, GAT, "
            "обучение на графовых структурах, эмбеддинги узлов"
        ),
        "category": "work",
        "subcategory": "Графовые методы",
        "parent": "Data Science и ML",
    },
    "networkx_igraph": {
        "aliases": ["networkx", "igraph", "анализ графов", "визуализация графов"],
        "enriched_text": (
            "библиотеки NetworkX и iGraph для анализа графов, "
            "алгоритмы на графах, визуализация, метрики центральности"
        ),
        "category": "work",
        "subcategory": "Графовые методы",
        "parent": "Data Science и ML",
    },
    "graph_databases": {
        "aliases": ["графовые бд", "neo4j", "графовая база", "cypher", "сифер"],
        "enriched_text": (
            "графовые базы данных Neo4j, язык запросов Cypher, "
            "хранение связанных данных, графовые алгоритмы"
        ),
        "category": "work",
        "subcategory": "Графовые методы",
        "parent": "Data Science и ML",
    },
    "neo4j_cypher": {
        "aliases": ["neo4j", "нео4j", "cypher", "сифер", "запросы к графам"],
        "enriched_text": (
            "графовая СУБД Neo4j и язык запросов Cypher, "
            "узлы и связи, pathfinding, рекомендательные системы на графах"
        ),
        "category": "work",
        "subcategory": "Графовые методы",
        "parent": "Data Science и ML",
    },

    "mlops": {
        "aliases": ["mlops", "млопс", "пайплайны ml", "mlflow", "kubeflow"],
        "enriched_text": (
            "MLOps, управление жизненным циклом ML-моделей, "
            "эксперименты, версионирование, деплой в продакшен"
        ),
        "category": "work",
        "subcategory": "MLOps",
        "parent": "Data Science и ML",
    },
    "mlflow_tracking": {
        "aliases": ["mlflow", "млфлоу", "трекинг экспериментов", "mlflow tracking"],
        "enriched_text": (
            "платформа MLflow для управления ML-жизненным циклом, "
            "логирование параметров и метрик, реестр моделей"
        ),
        "category": "work",
        "subcategory": "MLOps",
        "parent": "Data Science и ML",
    },
    "kubeflow": {
        "aliases": ["kubeflow", "кубефлоу", "куб флоу", "ml на kubernetes"],
        "enriched_text": (
            "платформа Kubeflow для ML на Kubernetes, "
            "пайплайны обучения, distributed training, сервинг моделей"
        ),
        "category": "work",
        "subcategory": "MLOps",
        "parent": "Data Science и ML",
    },
    "feature_store": {
        "aliases": ["feature store", "фичер стор", "хранилище признаков", "features"],
        "enriched_text": (
            "Feature Store для ML, централизованное хранилище признаков, "
            "переиспользование фич между моделями, online/offline serving"
        ),
        "category": "work",
        "subcategory": "MLOps",
        "parent": "Data Science и ML",
    },

    "databases": {
        "aliases": ["базы данных", "бд", "sql", "nosql", "хранилище"],
        "enriched_text": (
            "базы данных, реляционные SQL и нереляционные NoSQL, "
            "векторные БД, временные ряды, проектирование схем"
        ),
        "category": "work",
        "subcategory": "Базы данных",
        "parent": "IT и Разработка",
    },
    "relational_sql": {
        "aliases": ["sql", "реляционные", "postgresql", "mysql", "sqlite", "таблицы"],
        "enriched_text": (
            "реляционные SQL базы данных, PostgreSQL, MySQL, SQLite, "
            "написание запросов, нормализация, индексы и транзакции"
        ),
        "category": "work",
        "subcategory": "Реляционные SQL",
        "parent": "Базы данных",
    },
    "postgresql_db": {
        "aliases": ["postgresql", "postgres", "постгрес", "пг", "pg"],
        "enriched_text": (
            "продвинутая реляционная СУБД PostgreSQL, расширения pgvector, "
            "PostGIS, JSONB, оконные функции, полнотекстовый поиск"
        ),
        "category": "work",
        "subcategory": "Реляционные SQL",
        "parent": "Базы данных",
    },
    "pgvector_extension": {
        "aliases": ["pgvector", "pg_vector", "векторный постгрес", "эмбеддинги в postgres"],
        "enriched_text": (
            "расширение pgvector для PostgreSQL, векторное хранение и поиск, "
            "косинусное сходство, семантический поиск в базе данных"
        ),
        "category": "work",
        "subcategory": "Реляционные SQL",
        "parent": "Базы данных",
    },
    "postgis_spatial": {
        "aliases": ["postgis", "геоданные", "гео запросы", "spatial sql"],
        "enriched_text": (
            "расширение PostGIS для PostgreSQL, работа с геоданными, "
            "пространственные индексы и запросы, картографические сервисы"
        ),
        "category": "work",
        "subcategory": "Реляционные SQL",
        "parent": "Базы данных",
    },
    "mysql_db": {
        "aliases": ["mysql", "майскюэль", "мускуль", "mysql database"],
        "enriched_text": (
            "реляционная СУБД MySQL, популярная в веб-разработке, "
            "InnoDB, репликация, простота настройки"
        ),
        "category": "work",
        "subcategory": "Реляционные SQL",
        "parent": "Базы данных",
    },
    "sqlite_db": {
        "aliases": ["sqlite", "скюлайт", "эмбеддед база", "легкая бд"],
        "enriched_text": (
            "встраиваемая СУБД SQLite, без сервера, для мобильных приложений, "
            "тестирования и прототипирования"
        ),
        "category": "work",
        "subcategory": "Реляционные SQL",
        "parent": "Базы данных",
    },

    "nosql_db": {
        "aliases": ["nosql", "нереляционные", "монгодб", "mongodb", "redis", "cassandra"],
        "enriched_text": (
            "NoSQL базы данных, MongoDB документная, Redis key-value, "
            "Cassandra колоночная, гибкие схемы данных"
        ),
        "category": "work",
        "subcategory": "NoSQL Базы",
        "parent": "Базы данных",
    },
    "mongodb_db": {
        "aliases": ["mongodb", "монгодб", "монго", "mongo", "документная бд"],
        "enriched_text": (
            "документная NoSQL СУБД MongoDB, JSON-подобные документы, "
            "гибкая схема, агрегации, горизонтальное масштабирование"
        ),
        "category": "work",
        "subcategory": "NoSQL Базы",
        "parent": "Базы данных",
    },
    "redis_db": {
        "aliases": ["redis", "редис", "кэш", "in-memory", "брокер сообщений"],
        "enriched_text": (
            "in-memory хранилище Redis, кэширование, pub/sub брокер, "
            "очереди, сессии, высокая производительность"
        ),
        "category": "work",
        "subcategory": "NoSQL Базы",
        "parent": "Базы данных",
    },
    "cassandra_db": {
        "aliases": ["cassandra", "кассандра", "apache cassandra", "wide-column"],
        "enriched_text": (
            "распределенная NoSQL СУБД Apache Cassandra, колоночное хранение, "
            "высокая доступность, горизонтальная масштабируемость"
        ),
        "category": "work",
        "subcategory": "NoSQL Базы",
        "parent": "Базы данных",
    },
    "couchdb": {
        "aliases": ["couchdb", "каучдб", "couch db", "apache couchdb"],
        "enriched_text": (
            "документная СУБД CouchDB с синхронизацией, "
            "RESTful HTTP API, мастер-мастер репликация"
        ),
        "category": "work",
        "subcategory": "NoSQL Базы",
        "parent": "Базы данных",
    },

    "vector_dbs": {
        "aliases": ["vector db", "векторная бд", "pinecone", "qdrant", "weaviate", "milvus", "chroma"],
        "enriched_text": (
            "векторные базы данных для семантического поиска, "
            "Pinecone, Qdrant, Weaviate, Milvus, Chroma, хранение эмбеддингов"
        ),
        "category": "work",
        "subcategory": "Векторные БД",
        "parent": "Базы данных",
    },
    "pinecone_qdrant": {
        "aliases": ["pinecone", "пайнкоун", "qdrant", "квант", "векторный поиск облачный"],
        "enriched_text": (
            "облачные векторные БД Pinecone и Qdrant, "
            "семантический поиск, рекомендации, RAG для LLM"
        ),
        "category": "work",
        "subcategory": "Векторные БД",
        "parent": "Базы данных",
    },
    "weaviate_milvus": {
        "aliases": ["weaviate", "милвус", "milvus", "вейвиэйт", "векторный поиск опенсорс"],
        "enriched_text": (
            "векторные БД Weaviate и Milvus, open-source решения, "
            "гибридный поиск, поиск по эмбеддингам изображений и текста"
        ),
        "category": "work",
        "subcategory": "Векторные БД",
        "parent": "Базы данных",
    },
    "chroma_db": {
        "aliases": ["chroma", "хрома", "chromadb", "векторная бд для llm"],
        "enriched_text": (
            "легковесная векторная БД Chroma для AI-приложений, "
            "интеграция с LangChain, хранение эмбеддингов и метаданных"
        ),
        "category": "work",
        "subcategory": "Векторные БД",
        "parent": "Базы данных",
    },

    "time_series_db": {
        "aliases": ["временные ряды", "time series", "influxdb", "timescaledb", "метрики"],
        "enriched_text": (
            "базы данных для временных рядов, хранение метрик и событий, "
            "InfluxDB, TimescaleDB, IoT и мониторинг"
        ),
        "category": "work",
        "subcategory": "Базы данных",
        "parent": "Базы данных",
    },
    "influxdb": {
        "aliases": ["influxdb", "инфлюкс", "influx", "tsdb"],
        "enriched_text": (
            "специализированная СУБД InfluxDB для временных рядов, "
            "метрики, IoT, мониторинг, запросы на Flux"
        ),
        "category": "work",
        "subcategory": "Базы данных",
        "parent": "Базы данных",
    },
    "timescaledb": {
        "aliases": ["timescaledb", "таймскейл", "timescale", "postgres временные ряды"],
        "enriched_text": (
            "расширение TimescaleDB для PostgreSQL, эффективное хранение временных рядов, "
            "гипертаблицы, непрерывные агрегации"
        ),
        "category": "work",
        "subcategory": "Базы данных",
        "parent": "Базы данных",
    },

    "gamedev": {
        "aliases": ["gamedev", "геймдев", "разработка игр", "игрострой", "делаю игру"],
        "enriched_text": (
            "разработка игр, игровые движки Unity и Unreal Engine, "
            "3D-моделирование в Blender, геймдизайн и левел-дизайн"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "game_engines": {
        "aliases": ["игровой движок", "game engine", "unity", "unreal", "godot", "движок"],
        "enriched_text": (
            "игровые движки, среда для создания видеоигр, "
            "Unity с C#, Unreal Engine с C++ и Blueprints, Godot с GDScript"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "unity_dev": {
        "aliases": ["unity", "юнити", "юнити3д", "unity3d", "c# игры"],
        "enriched_text": (
            "игровой движок Unity, разработка на C#, 2D и 3D игры, "
            "компонентная архитектура, Asset Store, мультиплатформенность"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "unity_csharp": {
        "aliases": ["unity c#", "юнити шарп", "скриптинг unity", "monobehaviour"],
        "enriched_text": (
            "написание скриптов на C# для Unity, MonoBehavior, "
            "игровая логика, coroutines, физика и управление объектами"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "unity_ecs": {
        "aliases": ["ecs", "dots", "unity dots", "entity component system"],
        "enriched_text": (
            "архитектура ECS и DOTS в Unity, data-oriented подход, "
            "высокая производительность, параллельная обработка сущностей"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "unity_urp": {
        "aliases": ["urp", "universal render pipeline", "рендер unity", "графика unity"],
        "enriched_text": (
            "Universal Render Pipeline в Unity, кроссплатформенный рендеринг, "
            "Shader Graph, пост-обработка, оптимизация графики"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "unreal_engine": {
        "aliases": ["unreal engine", "анриал", "ue4", "ue5", "unreal engine 5", "анрил"],
        "enriched_text": (
            "игровой движок Unreal Engine, высококачественная графика, "
            "Blueprint визуальное программирование, C++, Nanite и Lumen"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "unreal_blueprints": {
        "aliases": ["blueprints", "блюпринты", "визуальный скриптинг", "ноды unreal"],
        "enriched_text": (
            "система визуального скриптинга Blueprints в Unreal Engine, "
            "создание игровой логики без кода, быстрый прототип"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "unreal_cpp": {
        "aliases": ["unreal c++", "с++ анриал", "ue c++", "unreal cpp"],
        "enriched_text": (
            "разработка на C++ в Unreal Engine, UObject, акторы, "
            "делегаты, сборка мусора, высокая производительность"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "unreal_nanite": {
        "aliases": ["nanite", "нанайт", "lumen", "люмен", "ue5 графика"],
        "enriched_text": (
            "технологии Unreal Engine 5 Nanite и Lumen, виртуализированная геометрия, "
            "динамическое глобальное освещение, next-gen графика"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "godot_engine": {
        "aliases": ["godot", "годот", "годо", "godot engine", "опенсорс движок"],
        "enriched_text": (
            "свободный игровой движок Godot, GDScript, легковесный, "
            "собственный UI, 2D и 3D игра из коробки"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "godot_gdscript": {
        "aliases": ["gdscript", "гдскрипт", "godot script", "скриптинг godot"],
        "enriched_text": (
            "язык GDScript в Godot, похож на Python, "
            "интегрирован в движок, сигналы, анимации и игровая логика"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },

    "3d_modeling_game": {
        "aliases": ["3д", "3d modeling", "трехмерное моделирование", "blender", "maya", "zbrush"],
        "enriched_text": (
            "3D-моделирование для игр и кино, Blender, Maya, ZBrush, "
            "скульптинг, текстурирование, ретопология"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "blender_3d": {
        "aliases": ["blender", "блендер", "3d blender", "блендер 3д"],
        "enriched_text": (
            "свободный 3D-редактор Blender, моделирование, скульптинг, "
            "анимация, рендеринг, Geometry Nodes, композитинг"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "blender_geometry_nodes": {
        "aliases": ["geometry nodes", "геометри ноды", "процедурное моделирование", "ноды blender"],
        "enriched_text": (
            "система Geometry Nodes в Blender, процедурное моделирование, "
            "генерация геометрии, нодовый редактор для создания объектов"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "maya_3d": {
        "aliases": ["maya", "аутодеск майя", "майа", "autodesk maya"],
        "enriched_text": (
            "профессиональный пакет Autodesk Maya для 3D-моделирования и анимации, "
            "индустриальный стандарт, риггинг и персонажная анимация"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "zbrush_sculpting": {
        "aliases": ["zbrush", "збраш", "скульптинг", "скульпт", "цифровой скульптинг"],
        "enriched_text": (
            "программа ZBrush для цифрового скульптинга, "
            "высокодетализированные модели, DynaMesh, создание персонажей"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "substance_painter": {
        "aliases": ["substance painter", "сабстенс", "текстурирование", "pbr"],
        "enriched_text": (
            "инструмент Substance Painter для текстурирования 3D-моделей, "
            "PBR материалы, смарт-маски, процедурные текстуры"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },

    "game_design": {
        "aliases": ["геймдизайн", "game design", "дизайн игры", "игровая механика", "баланс"],
        "enriched_text": (
            "геймдизайн, проектирование игровой механики, "
            "нарративный дизайн, дизайн уровней, балансировка"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "level_design": {
        "aliases": ["level design", "левел дизайн", "дизайн уровней", "локации", "карты"],
        "enriched_text": (
            "дизайн уровней для игр, построение пространства и навигации, "
            "расстановка врагов и предметов, темп игры"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "game_narrative": {
        "aliases": ["нарративный дизайн", "сюжет игры", "narrative design", "игровой сценарий"],
        "enriched_text": (
            "нарративный дизайн в играх, создание сюжета и диалогов, "
            "ветвление истории, лор и миростроение"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "game_balancing": {
        "aliases": ["баланс игры", "балансировка", "game balance", "нерф", "бафф"],
        "enriched_text": (
            "балансировка игр, настройка параметров персонажей и оружия, "
            "экономика игры, матчинг и сложность"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "procedural_gen": {
        "aliases": ["процедурная генерация", "procedural generation", "прокжен", "случайные уровни"],
        "enriched_text": (
            "процедурная генерация контента в играх, создание уровней и миров, "
            "алгоритмы шума, генерация ландшафта"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "audio_design": {
        "aliases": ["звуковой дизайн", "аудио в играх", "саунд дизайн", "wwise", "fmod"],
        "enriched_text": (
            "звуковой дизайн для игр, аудио-движки Wwise и FMOD, "
            "создание звуковых эффектов и музыки, интерактивный звук"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },
    "wwise_fmod": {
        "aliases": ["wwise", "вайз", "fmod", "эфмод", "аудио движок"],
        "enriched_text": (
            "промежуточное ПО Wwise и FMOD для звука в играх, "
            "интерактивный звук, адаптивная музыка, профилирование аудио"
        ),
        "category": "work",
        "subcategory": "Разработка игр",
        "parent": "IT и Разработка",
    },

    # ============================================================
    # 2. ГЕЙМИНГ
    # ============================================================
    "cybersport": {
        "aliases": ["киберспорт", "esports", "про-игрок", "турнир", "чемпионат", "киберкотлета"],
        "enriched_text": (
            "киберспорт, соревновательные видеоигры, турниры и чемпионаты, "
            "CS2, Dota 2, LoL, Valorant, профессиональные игроки и команды"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "cs2_game": {
        "aliases": ["cs2", "кс2", "контра", "cs go", "кс го", "кс", "каэс"],
        "enriched_text": (
            "Counter-Strike 2, тактический шутер от Valve, киберспорт, "
            "стрельба, командная игра, скины оружия, соревновательный режим"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "cs2_skins": {
        "aliases": ["скины cs2", "скины кс", "крафт скинов", "трейд скинов", "skin"],
        "enriched_text": (
            "коллекционирование скинов в CS2, раскрытие кейсов, "
            "трейд скинами, оценка редкости и износа оружия"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "dota2_game": {
        "aliases": ["дота", "дота2", "dota", "dota 2", "дотка", "моба"],
        "enriched_text": (
            "Dota 2, MOBA от Valve, стратегические командные сражения 5x5, "
            "герои, предметы, линии, рейтинг MMR, The International"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "lol_game": {
        "aliases": ["lol", "league of legends", "лига легенд", "лол", "полная лолка"],
        "enriched_text": (
            "League of Legends, популярная MOBA от Riot Games, "
            "чемпионы, линии, ранги, Worlds Championship"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "valorant_game": {
        "aliases": ["valorant", "валорант", "валорка", "шаровая молния"],
        "enriched_text": (
            "тактический шутер Valorant от Riot Games, "
            "агенты со способностями, стрельба, ранги, эпизоды"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "overwatch_game": {
        "aliases": ["overwatch", "овервотч", "овер", "овер2", "ow2"],
        "enriched_text": (
            "командный шутер Overwatch 2 от Blizzard, герои, "
            "роли танк/дд/саппорт, карты, режимы"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "pubg_game": {
        "aliases": ["pubg", "пабг", "королевская битва", "батл рояль", "пабж"],
        "enriched_text": (
            "королевская битва PUBG, выживание на карте, "
            "лут, зона, реалистичная стрельба и тактика"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "apex_legends": {
        "aliases": ["apex", "апекс", "апекс леджендс", "королевская битва герои"],
        "enriched_text": (
            "королевская битва Apex Legends, отряды, легенды со способностями, "
            "пинг-система, динамичный геймплей"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "rocket_league": {
        "aliases": ["rocket league", "ракетка", "футбол на машинах", "ракет лига"],
        "enriched_text": (
            "Rocket League, футбол на реактивных машинах, "
            "акробатика, голы, соревновательный и казуальный режим"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "fighting_games": {
        "aliases": ["файтинги", "fighting games", "файтинг", "мордобой", "дуэли"],
        "enriched_text": (
            "файтинги, соревновательные бои один на один, "
            "Tekken, Street Fighter, Mortal Kombat, комбо и фаталити"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "tekken_street_fighter": {
        "aliases": ["tekken", "текин", "street fighter", "стрит файтер", "файтер"],
        "enriched_text": (
            "файтинги Tekken и Street Fighter, 3D и 2D бои, "
            "фреймдата, комбо, персонажи, соревновательная сцена EVO"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "mortal_kombat": {
        "aliases": ["mortal kombat", "мортал комбат", "мк", "фаталити", "саб-зиро"],
        "enriched_text": (
            "серия файтингов Mortal Kombat, жестокие добивания фаталити, "
            "спецприемы, кровавый экшен, сюжетная линия"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },
    "smash_bros": {
        "aliases": ["smash bros", "смеш", "super smash", "супер смеш", "платформенный файтинг"],
        "enriched_text": (
            "кроссовер-файтинг Super Smash Bros, персонажи Nintendo, "
            "выбивание с арены, предметы, хаотичные бои"
        ),
        "category": "entertainment",
        "subcategory": "Киберспорт",
        "parent": "Гейминг",
    },

    "simulation_games": {
        "aliases": ["симуляторы", "симы", "сим", "гонки", "симулятор", "симуляция"],
        "enriched_text": (
            "симуляторы, гоночные симрейсинг, авиа и космические симуляторы, "
            "градостроительные и симуляторы жизни"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "racing_sims": {
        "aliases": ["симрейсинг", "гонки", "racing", "assetto corsa", "iracing", "gran turismo"],
        "enriched_text": (
            "симрейсинг, реалистичные гоночные симуляторы, "
            "Assetto Corsa, iRacing, Gran Turismo, руль и педали"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "assetto_corsa": {
        "aliases": ["assetto corsa", "асетто корса", "ассето", "симрейсинг италия"],
        "enriched_text": (
            "автосимулятор Assetto Corsa, реалистичная физика, "
            "лазерное сканирование трасс, моддинг, дрифт и гонки"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "iracing_sim": {
        "aliases": ["iracing", "айресинг", "онлайн гонки", "симрейсинг соревнования"],
        "enriched_text": (
            "онлайн симрейсинг iRacing, лицензии и рейтинг, "
            "профессиональные гонщики, сезонные чемпионаты"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "gran_turismo": {
        "aliases": ["gran turismo", "гран туризмо", "gt7", "гт7", "полифони"],
        "enriched_text": (
            "гоночный симулятор Gran Turismo, эксклюзив PlayStation, "
            "коллекционирование автомобилей, реализм"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "flight_sims": {
        "aliases": ["авиасимулятор", "flight sim", "мсфс", "dcs", "самолеты"],
        "enriched_text": (
            "авиасимуляторы, Microsoft Flight Simulator, DCS World, "
            "реалистичная симуляция полета, изучение авиации"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "msfs_dcs": {
        "aliases": ["msfs", "flight simulator", "dcs world", "авиа симы", "dcs"],
        "enriched_text": (
            "авиасимуляторы MSFS и DCS World, исследование мира и боевые вылеты, "
            "реалистичная авионика, HOTAS, VR поддержка"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "space_sims": {
        "aliases": ["космические симуляторы", "космосим", "star citizen", "elite dangerous"],
        "enriched_text": (
            "космические симуляторы, исследование галактики, торговля, "
            "сражения в открытом космосе, Star Citizen, Elite Dangerous"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "star_citizen": {
        "aliases": ["star citizen", "стар ситизен", "стар ситизн", "sc", "пылесос"],
        "enriched_text": (
            "амбициозный космический симулятор Star Citizen, "
            "открытый мир, мультиплеер, корабли, исследование вселенной"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "elite_dangerous": {
        "aliases": ["elite dangerous", "элит денжерос", "элитка", "галактический симулятор"],
        "enriched_text": (
            "космический симулятор Elite Dangerous, масштабная галактика, "
            "исследование, торговля, битвы, корабли и модули"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "life_sims": {
        "aliases": ["симуляторы жизни", "симс", "stardew valley", "animal crossing", "sims"],
        "enriched_text": (
            "симуляторы жизни, The Sims, Stardew Valley, Animal Crossing, "
            "фермерство, строительство и обустройство"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "sims_game": {
        "aliases": ["the sims", "симс", "симс 4", "sims 4", "симсы"],
        "enriched_text": (
            "серия The Sims, симуляция жизни персонажей, "
            "строительство домов, создание семей, дополнения и моды"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "stardew_valley": {
        "aliases": ["stardew valley", "стардью", "стардью валлей", "ферма", "пиксельная ферма"],
        "enriched_text": (
            "симулятор фермерской жизни Stardew Valley, выращивание растений, "
            "животные, рыбалка, отношения с жителями, шахты"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "animal_crossing": {
        "aliases": ["animal crossing", "энимал кроссинг", "анкх", "acnh", "нью хорайзонс"],
        "enriched_text": (
            "симулятор жизни Animal Crossing от Nintendo, "
            "обустройство острова, жители-животные, сезонные события"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "city_builders": {
        "aliases": ["градострой", "city builder", "cities skylines", "стройка города"],
        "enriched_text": (
            "градостроительные симуляторы, Cities Skylines, "
            "зонирование, транспорт, экономика и управление городом"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "cities_skylines": {
        "aliases": ["cities skylines", "ситис скайлайнс", "города", "skyline"],
        "enriched_text": (
            "градостроительный симулятор Cities Skylines, "
            "прокладка дорог, управление трафиком, зонирование, моды"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "surviving_mars": {
        "aliases": ["surviving mars", "колонизация марса", "сурвайвинг марс"],
        "enriched_text": (
            "градостроительный симулятор Surviving Mars, "
            "колонизация Красной планеты, постройка куполов, добыча ресурсов"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },
    "factorio_automation": {
        "aliases": ["factorio", "факторио", "фабрика", "автоматизация", "конвейеры"],
        "enriched_text": (
            "симулятор автоматизации Factorio, строительство фабрик, "
            "конвейерные ленты, оптимизация производства, защита от врагов"
        ),
        "category": "entertainment",
        "subcategory": "Симуляторы",
        "parent": "Гейминг",
    },

    "rpg_games": {
        "aliases": ["рпг", "rpg", "ролевые игры", "скилл", "прокачка", "квесты"],
        "enriched_text": (
            "ролевые игры RPG, прокачка персонажа, сюжет и квесты, "
            "западные RPG и японские JRPG, открытый мир, лут и билды"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "western_rpg": {
        "aliases": ["западные rpg", "wrpg", "baldurs gate", "skyrim", "ведьмак", "cyberpunk"],
        "enriched_text": (
            "западные RPG, Baldur's Gate 3, Elder Scrolls, Witcher, Cyberpunk, "
            "свобода выбора, моральные дилеммы, открытый мир"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "baldurs_gate3": {
        "aliases": ["baldur's gate 3", "балдурс гейт", "bg3", "бг3", "ларіан"],
        "enriched_text": (
            "ролевая игра Baldur's Gate 3 от Larian Studios, "
            "правила D&D 5e, пошаговые бои, глубокий сюжет и спутники"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "elder_scrolls": {
        "aliases": ["elder scrolls", "skyrim", "скайрим", "обливион", "морровинд"],
        "enriched_text": (
            "серия The Elder Scrolls, Skyrim, открытый мир, "
            "драконы, гильдии, моды, исследование подземелий"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "witcher_game": {
        "aliases": ["witcher", "ведьмак", "геральт", "дикая охота", "the witcher 3"],
        "enriched_text": (
            "серия RPG The Witcher, приключения Геральта из Ривии, "
            "охота на монстров, сложный моральный выбор, гвинт"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "cyberpunk_game": {
        "aliases": ["cyberpunk 2077", "сайберпанк", "киберпанк", "cd projekt red"],
        "enriched_text": (
            "ролевой экшен Cyberpunk 2077, Найт-Сити, киберимпланты, "
            "неонуар, ветвистый сюжет, Ви и Джонни Сильверхенд"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "mass_effect": {
        "aliases": ["mass effect", "масс эффект", "шепард", "нормандия", "жалкий"],
        "enriched_text": (
            "космическая RPG серия Mass Effect, капитан Шепард, "
            "выбор и последствия, исследование галактики, репутация"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "diablo_like": {
        "aliases": ["диабло", "diablo", "arpg", "экшн rpg", "дъябла", "poe"],
        "enriched_text": (
            "жанр Action RPG в стиле Diablo, изометрический вид, "
            "зачистка подземелий, лут, билды и прокачка персонажа"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "path_of_exile": {
        "aliases": ["path of exile", "poe", "поэ", "пас оф экзайл", "дерево пассивок"],
        "enriched_text": (
            "Action RPG Path of Exile, огромное дерево пассивных умений, "
            "камни умений, лиги, глубокая кастомизация билдов"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "jrpg_games": {
        "aliases": ["jrpg", "японские рпг", "final fantasy", "persona", "dragon quest"],
        "enriched_text": (
            "японские RPG, Final Fantasy, Persona, Dragon Quest, "
            "пошаговые бои, аниме-стиль, эпический сюжет"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "final_fantasy": {
        "aliases": ["final fantasy", "файнал фэнтези", "фф", "финалка", "square enix"],
        "enriched_text": (
            "культовая серия JRPG Final Fantasy, кристаллы и магия, "
            "эпидемические сюжеты, хокобо и муглы, саундтрек Уемацу"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "persona_series": {
        "aliases": ["persona", "персона", "shin megami tensei", "персона 5", "атлус"],
        "enriched_text": (
            "серия JRPG Persona от Atlus, школьная жизнь и подземелья, "
            "социальные связи, персоны, пошаговые бои в стиле аниме"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "dragon_quest": {
        "aliases": ["dragon quest", "драгон квест", "dq", "слаймы", "тораяма"],
        "enriched_text": (
            "классическая JRPG серия Dragon Quest, дизайн Акиры Ториямы, "
            "слаймы, пошаговые бои, светлое фэнтези"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "fire_emblem": {
        "aliases": ["fire emblem", "фаер эмблем", "фаер", "тактическая рпг"],
        "enriched_text": (
            "тактическая JRPG серия Fire Emblem, пошаговые сражения на сетке, "
            "перманентная смерть персонажей, поддержка и романтика"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "soulslike_genre": {
        "aliases": ["souls-like", "соулслайк", "душнила", "хардкор", "элден ринг"],
        "enriched_text": (
            "жанр souls-like, сложные экшен-RPG, высокий порог входа, "
            "заучивание паттернов врагов, смерти и чекпоинты у костров"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "elden_ring": {
        "aliases": ["elden ring", "элден ринг", "элден", "эр", "междуземье"],
        "enriched_text": (
            "шедевр FromSoftware Elden Ring, открытый мир Междуземья, "
            "эпидемичные боссы, пепел войны, сборка уникального билда"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "dark_souls": {
        "aliases": ["dark souls", "дарк соулс", "темные души", "дс", "лордран"],
        "enriched_text": (
            "серия Dark Souls, мрачная атмосфера, хардкорные бои, "
            "костры, эстус, запутанный лор и философия"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "sekiro_shadows": {
        "aliases": ["sekiro", "секиро", "тені", "шиноби", "парирование"],
        "enriched_text": (
            "Sekiro: Shadows Die Twice, феодальная Япония, "
            "точная система парирования, протез шиноби, скрытность"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "roguelike_genre": {
        "aliases": ["roguelike", "roguelite", "рогалик", "роглайк", "забеги", "перманентная смерть"],
        "enriched_text": (
            "жанр roguelike и roguelite, случайная генерация уровней, "
            "каждый забег уникален, синергии предметов"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "hades_game": {
        "aliases": ["hades", "гадес", "аид", "супергигант", "загрес"],
        "enriched_text": (
            "роглайк Hades от Supergiant Games, побег из подземного мира, "
            "греческая мифология, дары богов, прокачка оружия"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "dead_cells": {
        "aliases": ["dead cells", "дед селс", "метроидвания", "роглайт экшен"],
        "enriched_text": (
            "экшен-роглайт Dead Cells, метроидвания, "
            "динамичный бой, оружие и мутации, пиксельный стиль"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },
    "binding_of_isaac": {
        "aliases": ["the binding of isaac", "исаак", "таисия", "рофл", "слезы"],
        "enriched_text": (
            "роглайк The Binding of Isaac, процедурные подземелья, "
            "слезы как оружие, сотни предметов и синергий, мрачная тематика"
        ),
        "category": "entertainment",
        "subcategory": "RPG и JRPG",
        "parent": "Гейминг",
    },

    "board_games": {
        "aliases": ["настолки", "настольные игры", "board games", "настольщик", "игротека"],
        "enriched_text": (
            "настольные игры, еврогеймы Catan, Каркассон, "
            "коллекционные карточные игры MTG, варгеймы Warhammer, D&D"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "eurogames": {
        "aliases": ["еврогеймы", "euro", "немецкие настолки", "колонизаторы", "каркассон"],
        "enriched_text": (
            "европейские настольные игры, стратегия без прямого конфликта, "
            "подсчет победных очков, мипл, развитие и строительство"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "catan_game": {
        "aliases": ["колонизаторы", "catan", "катан", "остров катан", "settlers of catan"],
        "enriched_text": (
            "настольная игра Колонизаторы (Catan), торговля ресурсами, "
            "постройка поселений и городов, генерация поля из гексов"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "carcassonne_game": {
        "aliases": ["каркассон", "carcassonne", "тайловая игра", "миплы"],
        "enriched_text": (
            "настольная игра Каркассон, выкладывание тайлов, "
            "строительство городов и дорог, захват полей миплами"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "wingspan_game": {
        "aliases": ["wingspan", "крылья", "птицы", "движкострой"],
        "enriched_text": (
            "настольная игра Wingspan (Крылья), коллекционирование птиц, "
            "активация свойств в средах обитания, красивые иллюстрации"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "mtg_tcg": {
        "aliases": ["mtg", "magic the gathering", "магия", "кки", "коллекционные карты"],
        "enriched_text": (
            "коллекционная карточная игра Magic: The Gathering, "
            "колоды, мана, цвета, бустеры, стандарт и Commander"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "mtg_commander": {
        "aliases": ["commander", "edh", "коммандер", "едх", "тимелер"],
        "enriched_text": (
            "формат Commander (EDH) в Magic: The Gathering, "
            "колода из 100 уникальных карт, генерал-легенда, мультиплеер"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "dnd_rpg_tabletop": {
        "aliases": ["d&d", "днд", "dungeons and dragons", "подземелья и драконы", "нри"],
        "enriched_text": (
            "настольная ролевая игра Dungeons & Dragons, создание персонажа, "
            "броски d20, мастер, фэнтези-приключения в воображении"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "dnd_5e_rule": {
        "aliases": ["d&d 5e", "5 редакция", "пятерка", "днд 5", "5th edition"],
        "enriched_text": (
            "правила D&D пятой редакции, классы и архетипы, "
            "преимущество/помеха, билды персонажей, боевая система"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "dnd_worldbuilding": {
        "aliases": ["миростроение", "worldbuilding", "кампания d&d", "создание мира"],
        "enriched_text": (
            "создание миров и кампаний для D&D, картография, "
            "пантеон богов, политика фракций, написание квестов"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "pathfinder_rpg": {
        "aliases": ["pathfinder", "пасфайндер", "патчфайндер", "pf2e", "нри пайзо"],
        "enriched_text": (
            "настольная RPG Pathfinder от Paizo, глубокая кастомизация, "
            "тактические бои, обширный бестиарий и мир Голарион"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "warhammer_wargame": {
        "aliases": ["warhammer", "вархаммер", "40k", "40к", "варгейм", "империум"],
        "enriched_text": (
            "варгейм Warhammer 40,000, миниатюры, покрас, "
            "тактические сражения, Империум против Хаоса, орки и эльдары"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "warhammer_painting": {
        "aliases": ["покрас", "миниатюры", "хайлайты", "грунтовка", "сухая кисть"],
        "enriched_text": (
            "покраска миниатюр Warhammer, техники хайлайтинга и сухой кисти, "
            "цветовые схемы легионов, конверсии и кастомные базы"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "board_game_design": {
        "aliases": ["дизайн настолок", "разработка игр", "прототип настолки", "плейтест"],
        "enriched_text": (
            "создание и дизайн настольных игр, баланс механик, "
            "разработка правил, прототипирование и плейтесты"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },
    "arkham_horror_lcg": {
        "aliases": ["arkham horror", "аркхем хоррор", "лкц", "живая карточная игра"],
        "enriched_text": (
            "кооперативная карточная игра Arkham Horror LCG, "
            "сбор колоды сыщика, кампании Лавкрафтовских ужасов"
        ),
        "category": "entertainment",
        "subcategory": "Настольные игры",
        "parent": "Гейминг",
    },

    "retro_gaming": {
        "aliases": ["ретро-игры", "ретро гейминг", "старые игры", "эмуляция", "пиксели"],
        "enriched_text": (
            "ретро-гейминг, эмуляция и старые консоли Sega, NES, PS1, "
            "коллекционирование картриджей, пиксель-арт игры"
        ),
        "category": "entertainment",
        "subcategory": "Ретро-гейминг",
        "parent": "Гейминг",
    },
    "emulation_roms": {
        "aliases": ["эмуляция", "roms", "ромы", "retroarch", "эмуляторы"],
        "enriched_text": (
            "эмуляция ретро-игр, RetroArch, библиотека ROM-файлов, "
            "сохранение игрового наследия, шейдеры CRT"
        ),
        "category": "entertainment",
        "subcategory": "Ретро-гейминг",
        "parent": "Гейминг",
    },
    "retroarch_core": {
        "aliases": ["retroarch", "ретроарч", "ядра эмуляции", "libretro"],
        "enriched_text": (
            "универсальный эмулятор RetroArch, система ядер Libretro, "
            "шейдеры, перемотка, сетевой мультиплеер для старых игр"
        ),
        "category": "entertainment",
        "subcategory": "Ретро-гейминг",
        "parent": "Гейминг",
    },
    "retro_consoles": {
        "aliases": ["ретро-консоли", "сега", "snes", "денди", "плойка", "консоли детства"],
        "enriched_text": (
            "коллекционирование ретро-консолей Sega Genesis, NES, SNES, "
            "PlayStation 1/2, моддинг и RGB-апгрейды"
        ),
        "category": "entertainment",
        "subcategory": "Ретро-гейминг",
        "parent": "Гейминг",
    },
    "sega_genesis": {
        "aliases": ["sega genesis", "сега", "мега драйв", "mega drive", "16 бит"],
        "enriched_text": (
            "16-битная консоль Sega Genesis (Mega Drive), Sonic, "
            "Streets of Rage, эпоха консольных войн"
        ),
        "category": "entertainment",
        "subcategory": "Ретро-гейминг",
        "parent": "Гейминг",
    },
    "snes_nes": {
        "aliases": ["snes", "nes", "супер нинтендо", "денди", "famicom"],
        "enriched_text": (
            "консоли Nintendo SNES и NES (Денди), Super Mario, "
            "Zelda, 8- и 16-битная классика, картриджи"
        ),
        "category": "entertainment",
        "subcategory": "Ретро-гейминг",
        "parent": "Гейминг",
    },
    "ps1_ps2_retro": {
        "aliases": ["ps1", "ps2", "пс1", "плойка", "playstation 1", "сонька"],
        "enriched_text": (
            "классические консоли PlayStation 1 и 2, диски и memory card, "
            "Final Fantasy, Metal Gear Solid, эпоха 3D-графики"
        ),
        "category": "entertainment",
        "subcategory": "Ретро-гейминг",
        "parent": "Гейминг",
    },
    "pixel_art_games": {
        "aliases": ["пиксель-арт", "pixel art", "пиксельные игры", "инди пиксели"],
        "enriched_text": (
            "современные игры в стиле пиксель-арта, ностальгия по 8- и 16-биту, "
            "инди-проекты с пиксельной графикой"
        ),
        "category": "entertainment",
        "subcategory": "Ретро-гейминг",
        "parent": "Гейминг",
    },
    "speedrunning": {
        "aliases": ["спидран", "speedrun", "спидраннинг", "рекорд", "пройти быстро"],
        "enriched_text": (
            "спидраннинг игр, прохождение на скорость, мировые рекорды, "
            "оптимизация маршрутов и использование глитчей"
        ),
        "category": "entertainment",
        "subcategory": "Гейминг",
        "parent": "Гейминг",
    },

    # ============================================================
    # 3. ТВОРЧЕСТВО И ИСКУССТВО
    # ============================================================
    "digital_art": {
        "aliases": ["digital art", "цифровой рисунок", "диджитал арт", "cg арт", "цифра"],
        "enriched_text": (
            "цифровое искусство, рисование на планшете в Procreate и Photoshop, "
            "AI-генерация артов, пиксель-арт и воксель-арт"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "procreate_drawing": {
        "aliases": ["procreate", "прокриэйт", "прокреат", "рисование на ipad"],
        "enriched_text": (
            "приложение Procreate для рисования на iPad, "
            "кисти, слои, таймлапс, цифровая живопись и иллюстрации"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "photoshop_art": {
        "aliases": ["photoshop", "фотошоп", "адоб фотошоп", "adobe photoshop"],
        "enriched_text": (
            "растровый редактор Adobe Photoshop, цифровая живопись, "
            "обработка фото, коллажи, концепт-арт"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "illustrator_design": {
        "aliases": ["illustrator", "адоб иллюстратор", "илюстратор", "adobe illustrator", "вектор"],
        "enriched_text": (
            "векторный редактор Adobe Illustrator, логотипы, "
            "типографика, иконки, плоский дизайн и масштабируемая графика"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "clip_studio_paint": {
        "aliases": ["clip studio paint", "клип студио", "csp", "сай", "манга студио"],
        "enriched_text": (
            "программа Clip Studio Paint для рисования комиксов и манги, "
            "перья, тон, 3D-манекены для позинга"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "fractal_art": {
        "aliases": ["фракталы", "фрактальное искусство", "мандельброт", "апофизис"],
        "enriched_text": (
            "создание фрактальной графики, множество Мандельброта, "
            "алгоритмическое искусство на основе математических формул"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "pixel_art_create": {
        "aliases": ["пиксель-арт", "піксель", "aseprite", "пиксельное искусство"],
        "enriched_text": (
            "создание пиксельной графики, спрайты для игр, "
            "редактор Aseprite, ограниченная палитра, дазер-эффект"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "voxel_art": {
        "aliases": ["воксель-арт", "voxel", "магикавоксель", "magicavoxel", "объемные пиксели"],
        "enriched_text": (
            "создание объемного пиксельного искусства вокселей, "
            "MagicaVoxel, изометрические сцены и персонажи"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "ai_art_generation": {
        "aliases": ["нейроарт", "айи арт", "ai art", "миджорни", "stable diffusion"],
        "enriched_text": (
            "генерация изображений нейросетями, Midjourney, Stable Diffusion, "
            "DALL-E, написание промптов, постобработка AI-артов"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "midjourney_stable_diffusion": {
        "aliases": ["midjourney", "стабл дифьюжен", "stable diffusion", "mj", "sd", "нейроиллюстрация"],
        "enriched_text": (
            "нейросети Midjourney и Stable Diffusion, создание артов по тексту, "
            "стилизация, инпейнтинг, outpainting, генеративный дизайн"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "controlnet_workflow": {
        "aliases": ["controlnet", "контролнет", "позинг нейросетью", "canny", "openpose"],
        "enriched_text": (
            "расширение ControlNet для Stable Diffusion, контроль поз и композиции, "
            "скелетное управление, детекция краев, сегментация"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "dalle3_ai": {
        "aliases": ["dall-e", "далли", "dalle3", "генерация openai", "картинки нейросетью"],
        "enriched_text": (
            "нейросеть DALL-E 3 от OpenAI, генерация детализированных изображений, "
            "интеграция с ChatGPT, точное следование промпту"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },

    "traditional_art": {
        "aliases": ["традиционное искусство", "традишка", "живопись", "масло", "акварель", "карандаш"],
        "enriched_text": (
            "традиционное искусство, масляная и акварельная живопись, "
            "графика карандашом и углем, скульптура и керамика"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "oil_painting": {
        "aliases": ["масло", "oil painting", "масляная живопись", "холст", "мастихин"],
        "enriched_text": (
            "масляная живопись, работа с холстом и красками на масляной основе, "
            "лессировка, пастозная техника, разбавители и лаки"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "watercolor_art": {
        "aliases": ["акварель", "watercolor", "акварельная живопись", "размывка"],
        "enriched_text": (
            "акварельная живопись, прозрачные краски на водной основе, "
            "техника мокрым по мокрому, лессировки, бумага и кисти"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "graphic_technique": {
        "aliases": ["графика", "карандаш", "уголь", "рисунок", "скетч", "набросок"],
        "enriched_text": (
            "графический рисунок карандашом, углем, пастелью, "
            "академический штрих, светотень, анатомические зарисовки"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "calligraphy": {
        "aliases": ["каллиграфия", "леттеринг", "перо", "тушь", "красивый почерк"],
        "enriched_text": (
            "искусство каллиграфии и леттеринга, работа пером и тушью, "
            "готические, славянские и азиатские стили письма"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },
    "sculpture_ceramics": {
        "aliases": ["скульптура", "керамика", "гончарный круг", "глина", "лепка"],
        "enriched_text": (
            "скульптура и работа с керамикой, лепка из глины, "
            "гончарный круг, обжиг и глазурование, создание форм"
        ),
        "category": "life",
        "subcategory": "Творчество и Искусство",
        "parent": "Творчество и Искусство",
    },

    "graphic_design": {
        "aliases": ["графический дизайн", "дизайн", "визуал", "айдентика", "лого"],
        "enriched_text": (
            "графический дизайн, UI/UX дизайн интерфейсов в Figma, "
            "фирменный стиль, типографика, моушн-дизайн и полиграфия"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "ui_ux_design": {
        "aliases": ["ui/ux", "юай юикс", "проектирование интерфейсов", "веб-дизайн", "app design"],
        "enriched_text": (
            "UI/UX дизайн, проектирование пользовательских интерфейсов, "
            "прототипирование в Figma, юзабилити-тесты, вайрфреймы"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "figma_tool": {
        "aliases": ["figma", "фигма", "фигма дизайн", "облачный дизайн"],
        "enriched_text": (
            "графический редактор Figma, совместная работа в реальном времени, "
            "компоненты, auto layout, прототипирование интерфейсов"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "figma_plugins": {
        "aliases": ["плагины фигма", "figma plugins", "автоматизация дизайна"],
        "enriched_text": (
            "плагины для Figma, автоматизация дизайн-процессов, "
            "генерация контента, иконок, accessibility проверки"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "sketch_app": {
        "aliases": ["sketch", "скетч", "sketch app", "макос дизайн"],
        "enriched_text": (
            "векторный редактор Sketch для macOS, интерфейсный дизайн, "
            "символы и библиотеки, экспорт ассетов"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "prototyping_wireframing": {
        "aliases": ["прототипирование", "вайрфреймы", "wireframe", "кликабельный прототип"],
        "enriched_text": (
            "создание прототипов и вайрфреймов, проверка гипотез, "
            "интерактивные макеты и тестирование с пользователями"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "ux_research": {
        "aliases": ["ux исследования", "исследование пользователей", "юзабилити", "интервью"],
        "enriched_text": (
            "UX-исследования, глубинные интервью, тестирование удобства, "
            "CustDev, персоны и сценарии использования"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "motion_design": {
        "aliases": ["моушн дизайн", "motion design", "анимация", "after effects", "афтер"],
        "enriched_text": (
            "моушн-дизайн и анимация, After Effects, создание рекламных роликов, "
            "анимация интерфейсов, кинетическая типографика"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "after_effects": {
        "aliases": ["after effects", "афтер эффектс", "ae", "ае", "композитинг"],
        "enriched_text": (
            "программа Adobe After Effects, моушн-дизайн и визуальные эффекты, "
            "анимация слоев, expressions, плагины"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "rive_animations": {
        "aliases": ["rive", "райв", "интерактивная анимация", "rive app"],
        "enriched_text": (
            "инструмент Rive для интерактивных анимаций, state machine, "
            "анимация в реальном времени для приложений и игр"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "branding_identity": {
        "aliases": ["брендинг", "айдентика", "фирменный стиль", "логотип", "ребрендинг"],
        "enriched_text": (
            "разработка брендинга и айдентики, логотипы, цветовые палитры, "
            "гайдлайны, стратегия позиционирования бренда"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "typography_fonts": {
        "aliases": ["типографика", "шрифты", "fonts", "керинг", "интерлиньяж"],
        "enriched_text": (
            "типографика и работа со шрифтами, подбор гарнитур, "
            "создание шрифтовых композиций, вариативные шрифты"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "variable_fonts": {
        "aliases": ["вариативные шрифты", "variable fonts", "ось шрифта"],
        "enriched_text": (
            "технология вариативных шрифтов OpenType, бесконечная настройка веса, "
            "ширины и наклона в одном файле шрифта"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },
    "print_design": {
        "aliases": ["полиграфия", "печатный дизайн", "cmyk", "верстка буклета"],
        "enriched_text": (
            "полиграфический дизайн, подготовка макетов к печати, "
            "визитки, буклеты, журналы, цветоделение CMYK"
        ),
        "category": "life",
        "subcategory": "Графический дизайн",
        "parent": "Творчество и Искусство",
    },

    "cinema_filmmaking": {
        "aliases": ["кино", "кинематограф", "фильммейкинг", "режиссура", "сценарий", "съемка"],
        "enriched_text": (
            "кинематограф и создание фильмов, режиссура и сценарное мастерство, "
            "видеомонтаж, цветокоррекция, VFX-композитинг"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "film_directing": {
        "aliases": ["режиссура", "режиссер", "постановка", "мизансцена", "раскадровка"],
        "enriched_text": (
            "режиссура кино, постановка кадра и работа с актерами, "
            "раскадровка сцен, визуальное повествование"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "screenwriting": {
        "aliases": ["сценарий", "сценарное мастерство", "script", "диалоги", "сюжет"],
        "enriched_text": (
            "написание сценариев для кино и сериалов, структура актов, "
            "разработка персонажей и диалогов, софт Final Draft"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "film_editing": {
        "aliases": ["видеомонтаж", "монтаж", "editing", "склейка", "таймлайн"],
        "enriched_text": (
            "видеомонтаж и пост-продакшн, DaVinci Resolve, Premiere Pro, "
            "склейка сцен, работа со звуком, темп повествования"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "davinci_resolve": {
        "aliases": ["davinci resolve", "давинчи", "резолв", "цветокор", "монтаж и цвет"],
        "enriched_text": (
            "программа DaVinci Resolve, профессиональный видеомонтаж и цветокоррекция, "
            "Fusion VFX, Fairlight аудио, нодовый подход"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "premiere_pro": {
        "aliases": ["premiere pro", "премьер", "адоб премьер", "премьерка"],
        "enriched_text": (
            "видеоредактор Adobe Premiere Pro, нелинейный монтаж, "
            "интеграция с After Effects, работа с прокси"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "final_cut_pro": {
        "aliases": ["final cut", "финал кат", "fcp", "файнал кат", "монтаж на маке"],
        "enriched_text": (
            "видеоредактор Final Cut Pro от Apple, магнитный таймлайн, "
            "оптимизация под macOS, работа с ProRes"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "color_grading": {
        "aliases": ["цветокоррекция", "грейдинг", "колор грейдинг", "колорист", "lut"],
        "enriched_text": (
            "цветокоррекция и колористика кино, создание настроения цветом, "
            "работа с LUT и кривыми, вторичная коррекция"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "indie_filmmaking": {
        "aliases": ["инди-кино", "независимое кино", "инди", "малобюджетка", "авторское"],
        "enriched_text": (
            "создание независимого малобюджетного кино, краудфандинг, "
            "фестивальная дистрибуция, DIY подход к съемкам"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "documentary_film": {
        "aliases": ["документальное кино", "документалистика", "док", "реальное кино"],
        "enriched_text": (
            "производство документальных фильмов, интервью и наблюдение, "
            "социальные и исторические темы, журналистские расследования"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "vfx_compositing": {
        "aliases": ["vfx", "визуальные эффекты", "спецэффекты", "композитинг", "nuke"],
        "enriched_text": (
            "создание визуальных эффектов и композитинг, Nuke, Houdini, "
            "зеленый экран, трекинг и 3D интеграция"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "nuke_compositing": {
        "aliases": ["nuke", "нюк", "нодовый композитинг", "the foundry", "нук"],
        "enriched_text": (
            "профессиональная программа Nuke для композитинга, нодовая система, "
            "работа с каналами, depth и 3D-пространство"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },
    "houdini_fx": {
        "aliases": ["houdini", "гудини", "houdini fx", "процедурные эффекты"],
        "enriched_text": (
            "программа Houdini для процедурных спецэффектов, симуляции огня и воды, "
            "разрушения, нодовый подход, VEX scripting"
        ),
        "category": "entertainment",
        "subcategory": "Кинематограф",
        "parent": "Творчество и Искусство",
    },

    "tv_series_binge": {
        "aliases": ["сериалы", "binge watching", "запой", "смотрю сериал", "нетфликс"],
        "enriched_text": (
            "просмотр сериалов, научная фантастика, драмы, аниме и комедии, "
            "обсуждение серий и теорий, поиск новых шоу"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Развлечения",
    },
    "sci_fi_series": {
        "aliases": ["научная фантастика", "sci-fi сериал", "star trek", "черное зеркало"],
        "enriched_text": (
            "научно-фантастические сериалы, космос и технологии будущего, "
            "антиутопии, путешествия во времени, инопланетяне"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Развлечения",
    },
    "drama_series": {
        "aliases": ["драма", "драматический сериал", "тяжелая драма", "emotional"],
        "enriched_text": (
            "драматические сериалы, глубокие персонажи и сюжеты, "
            "медицинские и юридические драмы, семейные саги"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Развлечения",
    },
    "anime_series": {
        "aliases": ["аниме", "анимешки", "anime", "японская анимация", "тайтилы"],
        "enriched_text": (
            "просмотр аниме, японская анимация, жанры сенен и slice of life, "
            "опенинги, вайфу и лучшие девочки, студии и мангаки"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Развлечения",
    },
    "slice_of_life_anime": {
        "aliases": ["повседневность", "slice of life", "соl", "милота", "бытовое аниме"],
        "enriched_text": (
            "аниме жанра slice of life, спокойная повседневность, "
            "школа и романтика, уютная атмосфера, комедия ситуаций"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Развлечения",
    },
    "shonen_anime": {
        "aliases": ["сенен", "shonen", "боевое аниме", "дружба и сила", "наруто"],
        "enriched_text": (
            "жанр аниме сенен для юношей, эпидемические битвы и сила дружбы, "
            "тренировки и прокачка, Наруто, Блич, Ван Пис"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Развлечения",
    },
    "comedy_series": {
        "aliases": ["комедийный сериал", "ситком", "sitcom", "смешное шоу"],
        "enriched_text": (
            "комедийные сериалы и ситкомы, юмор и смех, "
            "закадровый смех, ситуационные комедии, стендап-шоу"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Развлечения",
    },
    "true_crime_docs": {
        "aliases": ["true crime", "тру крайм", "криминальные доки", "документалки убийства"],
        "enriched_text": (
            "жанр документалистики true crime, реальные преступления, "
            "расследования, психология преступников, судебные процессы"
        ),
        "category": "entertainment",
        "subcategory": "Кино и Видео",
        "parent": "Развлечения",
    },

    "photography_art": {
        "aliases": ["фотография", "фото", "фоткаю", "съемка", "объектив", "зеркалка"],
        "enriched_text": (
            "искусство фотографии, стрит-фото и портретная съемка, "
            "пейзаж и пленочная аналоговая фотография, дроны"
        ),
        "category": "life",
        "subcategory": "Фотография",
        "parent": "Творчество и Искусство",
    },
    "street_photo": {
        "aliases": ["стрит-фотография", "уличное фото", "street photo", "жизнь улиц"],
        "enriched_text": (
            "уличная фотография, запечатление городской жизни, "
            "откровенные кадры, игра теней, решающий момент по Брессону"
        ),
        "category": "life",
        "subcategory": "Фотография",
        "parent": "Творчество и Искусство",
    },
    "portrait_photo": {
        "aliases": ["портрет", "портретная съемка", "portrait", "модель", "студийный свет"],
        "enriched_text": (
            "портретная фотография, работа с моделью в студии и на локации, "
            "световые схемы, ретушь портретов в Photoshop"
        ),
        "category": "life",
        "subcategory": "Фотография",
        "parent": "Творчество и Искусство",
    },
    "landscape_photo": {
        "aliases": ["пейзаж", "ландшафт", "природа", "закат", "горы", "восход"],
        "enriched_text": (
            "пейзажная фотография, съемка природы и ландшафтов, "
            "золотой час, длинные выдержки, фильтры и штативы"
        ),
        "category": "life",
        "subcategory": "Фотография",
        "parent": "Творчество и Искусство",
    },
    "film_analog_photo": {
        "aliases": ["пленка", "пленочное фото", "аналог", "зерно", "35мм", "средний формат"],
        "enriched_text": (
            "аналоговая пленочная фотография, проявка и печать в темной комнате, "
            "зернистость и цветопередача, винтажные камеры"
        ),
        "category": "life",
        "subcategory": "Фотография",
        "parent": "Творчество и Искусство",
    },
    "drone_photo": {
        "aliases": ["дрон", "квадрокоптер", "аэросъемка", "коптер", "вид сверху"],
        "enriched_text": (
            "аэрофотосъемка с дрона, DJI, полеты над городом и природой, "
            "панорамы с воздуха, legal высота и правила полетов"
        ),
        "category": "life",
        "subcategory": "Фотография",
        "parent": "Творчество и Искусство",
    },
    "macro_photography": {
        "aliases": ["макро", "macro photo", "макросъемка", "насекомые крупно", "капля воды"],
        "enriched_text": (
            "макрофотография, съемка мелких объектов крупным планом, "
            "насекомые и цветы, макро-линзы и кольца, стекинг фокуса"
        ),
        "category": "life",
        "subcategory": "Фотография",
        "parent": "Творчество и Искусство",
    },
    "mobile_photography": {
        "aliases": ["мобильное фото", "съемка на телефон", "айфонография", "смартфон"],
        "enriched_text": (
            "фотография на смартфон, мобильные приложения для обработки, "
            "вычислительная фотография, instagram-формат"
        ),
        "category": "life",
        "subcategory": "Фотография",
        "parent": "Творчество и Искусство",
    },

    # ============================================================
    # 4. МУЗЫКА И АУДИО
    # ============================================================
    "music_creation": {
        "aliases": ["написание музыки", "создание треков", "продакшн", "битмейкинг", "аранжировка"],
        "enriched_text": (
            "написание и создание музыки, продюсирование в Ableton и FL Studio, "
            "запись вокала, сведение и мастеринг треков, диджеинг"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "music_production_daw": {
        "aliases": ["daw", "секвенсор", "рабочая станция", "студия звукозаписи"],
        "enriched_text": (
            "музыкальное продюсирование в DAW, Ableton Live, FL Studio, Logic Pro, "
            "запись и редактирование MIDI и аудио, виртуальные инструменты"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "ableton_live": {
        "aliases": ["ableton", "аблетон", "абла", "ableton live", "live"],
        "enriched_text": (
            "программа Ableton Live для создания музыки, сессионный вид, "
            "варпинг аудио, Max for Live, живое исполнение электронной музыки"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "ableton_max4live": {
        "aliases": ["max for live", "max4live", "макс", "м4л", "патчи"],
        "enriched_text": (
            "платформа Max for Live внутри Ableton, создание своих устройств, "
            "визуальное программирование звука и MIDI эффектов"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "fl_studio": {
        "aliases": ["fl studio", "фл студио", "фруктовые петли", "fruity loops", "фл"],
        "enriched_text": (
            "секвенсор FL Studio, паттерновая система, piano roll, "
            "создание битов и электронной музыки, популярен у битмейкеров"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "logic_pro": {
        "aliases": ["logic pro", "лоджик", "logic", "гаражбенд про", "apple daw"],
        "enriched_text": (
            "профессиональный секвенсор Logic Pro от Apple, виртуальные инструменты, "
            "Drumer, Alchemy синтезатор, семплирование"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "cubase_nuendo": {
        "aliases": ["cubase", "нуендо", "кьюбейс", "steinberg", "штайнберг"],
        "enriched_text": (
            "DAW Cubase и Nuendo от Steinberg, MIDI редактирование, "
            "пост-продакшн звука для кино, продвинутая автоматизация"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "studio_one": {
        "aliases": ["studio one", "студио уан", "presonus", "пресонус"],
        "enriched_text": (
            "цифровая рабочая станция Studio One от PreSonus, drag and drop, "
            "интегрированный мастеринг, ARA поддержка"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "songwriting": {
        "aliases": ["сонграйтинг", "написание песен", "сочинение", "авторская песня"],
        "enriched_text": (
            "написание песен, создание текстов и мелодий, гармония, "
            "структура куплет-припев, рифмовка и смысл"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "lyrics_writing": {
        "aliases": ["тексты песен", "лирика", "рифмы", "стихи для песен", "барс"],
        "enriched_text": (
            "написание текстов песен, поэзия в музыке, поиск рифм, "
            "метафоры и образы, речитатив и рэп-тексты"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "beatmaking": {
        "aliases": ["битмейкинг", "биты", "делаю бит", "инструментал", "минус"],
        "enriched_text": (
            "создание битов и инструменталов, хип-хоп и трэп продакшн, "
            "дрэм-машина, сэмплирование, 808 бас"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "trap_beats": {
        "aliases": ["трэп биты", "trap", "хайхеты триплетами", "трэп продакшн"],
        "enriched_text": (
            "создание битов в стиле трэп, быстрые хай-хеты, "
            "тяжелый 808 бас, роллы, мрачная атмосфера"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "lofi_beats": {
        "aliases": ["lofi", "лоуфай", "спокойные биты", "чилл", "расслабляющая музыка"],
        "enriched_text": (
            "создание lo-fi битов, спокойная музыка для учебы и релакса, "
            "виниловый шум, теплый звук, джазовые аккорды"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "mixing_mastering": {
        "aliases": ["сведение", "мастеринг", "микс", "эквализация", "компрессия"],
        "enriched_text": (
            "сведение и мастеринг музыки, баланс громкости и панорамы, "
            "iZotope Ozone и FabFilter, подготовка трека к релизу"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "izotope_ozone": {
        "aliases": ["izotope ozone", "озон", "изотоп", "мастеринг плагин"],
        "enriched_text": (
            "плагин iZotope Ozone для мастеринга, AI-помощник, "
            "эквалайзер и максимизатор, подготовка трека к стримингу"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "fab_filter": {
        "aliases": ["fabfilter", "фабфильтер", "про-q", "pro-q", "про л"],
        "enriched_text": (
            "плагины FabFilter для сведения и мастеринга, Pro-Q эквалайзер, "
            "прозрачное звучание и удобный интерфейс"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "dj_mixing": {
        "aliases": ["диджеинг", "dj", "сведение треков", "микс", "пульты"],
        "enriched_text": (
            "диджеинг, сведение треков вживую, виниловый и цифровой диджеинг, "
            "битмэтчинг, скретч, работа с CDJ и контроллерами"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "vinyl_dj": {
        "aliases": ["винил", "вертушки", "пластинки", "техникс", "dvs"],
        "enriched_text": (
            "диджеинг на виниловых пластинках, Technics SL-1200, "
            "ручной битмэтчинг, DVS системы Serato и Traktor"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "digital_dj": {
        "aliases": ["cdj", "цифровой диджей", "пионер", "rekordbox", "флешка"],
        "enriched_text": (
            "цифровой диджеинг на CDJ и контроллерах Pioneer, "
            "анализ треков в Rekordbox, синхронизация, лупы и горячие точки"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },
    "scratch_dj": {
        "aliases": ["скретч", "scratch", "тернтаблизм", "кат", "чирик"],
        "enriched_text": (
            "скретч-техника на вертушках, тернтаблизм, джагглинг, "
            "использование кроссфейдера, баттлы диджеев"
        ),
        "category": "life",
        "subcategory": "Написание музыки",
        "parent": "Музыка и Аудио",
    },

    "music_instruments": {
        "aliases": ["музыкальные инструменты", "играю на", "репетиция", "домашняя студия"],
        "enriched_text": (
            "игра на музыкальных инструментах, гитара и фортепиано, "
            "барабаны и синтезаторы, создание музыки вживую"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "guitar_play": {
        "aliases": ["гитара", "guitar", "играю на гитаре", "гитарист", "перебор"],
        "enriched_text": (
            "игра на гитаре, электрогитара и акустика, бас-гитара, "
            "аккорды и соло, гитарные эффекты и педали"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "electric_guitar": {
        "aliases": ["электрогитара", "лес пол", "стратокастер", "звукосниматель", "перегруз"],
        "enriched_text": (
            "электрогитара, хамбакеры и синглы, гитарный усилитель и кабинет, "
            "дисторшн и овердрайв, тэппинг и бенды"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "acoustic_guitar": {
        "aliases": ["акустическая гитара", "акустика", "дредноут", "фламенко", "фингерстайл"],
        "enriched_text": (
            "акустическая гитара, фингерстайл и перебор, "
            "игра боем, дредноут и классическая форма корпуса"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "bass_guitar": {
        "aliases": ["бас", "бас-гитара", "басс", "слэп", "грув"],
        "enriched_text": (
            "бас-гитара, создание ритм-секции и грува, "
            "техника слэпа, глубокая частота и фундамент трека"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "piano_keys": {
        "aliases": ["пианино", "фортепиано", "рояль", "клавиши", "миди-клавиатура"],
        "enriched_text": (
            "игра на фортепиано и клавишных, акустический рояль и синтезатор, "
            "чтение нот, аккорды, арпеджио и импровизация"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "drums_percussion": {
        "aliases": ["барабаны", "ударные", "драмс", "ударная установка", "перкуссия"],
        "enriched_text": (
            "игра на барабанах и ударных инструментах, биты и заполнения, "
            "аккустическая и электронная ударная установка"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "electronic_drums": {
        "aliases": ["электронные барабаны", "роланд", "электронная установка", "меши"],
        "enriched_text": (
            "электронная ударная установка, mesh-пластики, "
            "модули Roland, VST для барабанов, тихая репетиция"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "synthesizers": {
        "aliases": ["синтезатор", "синт", "аналоговый синт", "волна", "осциллятор"],
        "enriched_text": (
            "игра на синтезаторах и создание звуков, модульный синтез, "
            "аналоговые и цифровые синты, звуковой дизайн"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "modular_synth": {
        "aliases": ["модульный синтезатор", "еврорак", "eurorack", "патч", "cv"],
        "enriched_text": (
            "модульный синтез и стандарт Eurorack, патчинг кабелями, "
            "генераторы и фильтры, управление напряжением CV/Gate"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "analog_synths": {
        "aliases": ["аналоговый синтезатор", "муг", "moog", "prophet", "juno"],
        "enriched_text": (
            "аналоговые синтезаторы, теплое звучание, "
            "осцилляторы и фильтры, легендарные модели Moog и Roland"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },
    "fm_synthesis": {
        "aliases": ["fm синтез", "yamaha dx7", "оператор", "частотная модуляция"],
        "enriched_text": (
            "FM-синтез звука на основе частотной модуляции, "
            "характерные колокольные и металлические тембры, Yamaha DX7"
        ),
        "category": "life",
        "subcategory": "Музыкальные инструменты",
        "parent": "Музыка и Аудио",
    },

    "music_genres": {
        "aliases": ["музыкальные стили", "жанры", "направления", "слушаю"],
        "enriched_text": (
            "музыкальные жанры, рок и метал, электронная музыка и техно, "
            "хип-хоп, джаз и классика, инди и альтернатива"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "rock_metal": {
        "aliases": ["рок", "метал", "тяжеляк", "прогрессив рок", "дэт метал", "пост-панк"],
        "enriched_text": (
            "рок и метал музыка, прогрессивный рок и дэт-метал, "
            "пост-панк и хард-рок, электрогитары и мощный вокал"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "progressive_rock": {
        "aliases": ["прогрессивный рок", "прог", "prog rock", "сложный рок", "арт-рок"],
        "enriched_text": (
            "прогрессивный рок, сложные композиции и нестандартные размеры, "
            "концептуальные альбомы, виртуозное исполнение"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "death_metal": {
        "aliases": ["дэт-метал", "death metal", "гроул", "тяжелая музыка", "бласт биты"],
        "enriched_text": (
            "экстремальный жанр дэт-метал, низкий гроулинг вокал, "
            "скоростные бласт-биты и техничные риффы"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "post_punk": {
        "aliases": ["пост-панк", "post punk", "советский рок", "кино", "мрачный рок"],
        "enriched_text": (
            "жанр пост-панк, холодное и мрачное звучание, "
            "группы Кино и Joy Division, бас-гитара на первом плане"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "electronic_music": {
        "aliases": ["электронная музыка", "электроника", "техно", "днб", "амбиент", "синтвейв"],
        "enriched_text": (
            "электронная музыка, техно и хаус, драм-н-бейс, эмбиент и IDM, "
            "синтвейв и ретровейв, клубная и танцевальная сцена"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "techno_house": {
        "aliases": ["техно", "хаус", "techno", "house", "клубняк", "четыре четверти"],
        "enriched_text": (
            "танцевальные жанры техно и хаус, прямой бит 4/4, "
            "рейвы и диджеи, андеграунд клубная культура"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "dnb_dubstep": {
        "aliases": ["днб", "dnb", "дабстеп", "dubstep", "бростеп", "драм энд бейс"],
        "enriched_text": (
            "жанры драм-н-бейс и дабстеп, ломаные ритмы и глубокий бас, "
            "энергичная и бас-ориентированная электронная музыка"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "ambient_idm": {
        "aliases": ["эмбиент", "ambient", "idm", "интеллиджент", "атмосферная музыка"],
        "enriched_text": (
            "эмбиент и IDM, атмосферная и экспериментальная электроника, "
            "медитативное звучание и сложные ритмические рисунки"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "synthwave_retro": {
        "aliases": ["синтвейв", "synthwave", "ретровейв", "ретро электроника", "outrun"],
        "enriched_text": (
            "жанр синтвейв и ретровейв, ностальгия по 80-м, "
            "неоновые арты и эстетика, музыка в стиле старых фильмов"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "hiphop_rap": {
        "aliases": ["хип-хоп", "рэп", "реп", "hip hop", "рэпчик", "фристайл"],
        "enriched_text": (
            "хип-хоп и рэп музыка, биты и речитатив, уличная культура, "
            "граффити и брейк-данс, тексты о жизни и социалке"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "jazz_blues": {
        "aliases": ["джаз", "jazz", "блюз", "blues", "саксофон", "импровизация"],
        "enriched_text": (
            "джаз и блюз музыка, импровизация и свинг, "
            "духовые инструменты, блюзовая гамма и квадрат"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "classical_music": {
        "aliases": ["классическая музыка", "классика", "симфония", "опера", "академическая"],
        "enriched_text": (
            "академическая классическая музыка, симфонический оркестр, "
            "опера и вокал, Бах и Моцарт, камерные концерты"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "opera_vocal": {
        "aliases": ["опера", "вокал", "оперное пение", "тенор", "сопрано"],
        "enriched_text": (
            "оперное искусство и академический вокал, арии, "
            "театральные постановки и певцы мировой величины"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "indie_alternative": {
        "aliases": ["инди", "indie", "альтернатива", "независимая музыка", "инди-рок"],
        "enriched_text": (
            "инди и альтернативная музыка, независимые лейблы, "
            "экспериментальное звучание, DIY культура"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "kpop_genre": {
        "aliases": ["k-pop", "кпоп", "кейпоп", "корейская попса", "айдолы"],
        "enriched_text": (
            "корейская поп-музыка K-pop, айдол-группы и хореография, "
            "фандомы и лайтстики, клипы и визуал высшего качества"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "funk_soul": {
        "aliases": ["фанк", "соул", "funk", "soul", "грув", "ритм-н-блюз"],
        "enriched_text": (
            "фанк и соул музыка, качающий грув и ритм-секция, "
            "душевный вокал и духовые аранжировки"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },
    "folk_world": {
        "aliases": ["фолк", "этно", "народная музыка", "world music", "кельтская"],
        "enriched_text": (
            "фолк и этническая музыка, народные инструменты и мотивы, "
            "культурные традиции, world music фестивали"
        ),
        "category": "entertainment",
        "subcategory": "Музыка и Аудио",
        "parent": "Музыка и Аудио",
    },

    # ============================================================
    # 5. САМОРАЗВИТИЕ
    # ============================================================
    "management_leadership": {
        "aliases": ["менеджмент", "лидерство", "управление", "тимлид", "руководитель"],
        "enriched_text": (
            "менеджмент и лидерство, управление командами и проектами, "
            "Agile и Scrum, OKR и стратегическое планирование"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },
    "project_management": {
        "aliases": ["управление проектами", "проджект менеджмент", "pm", "пм", "дедлайны"],
        "enriched_text": (
            "управление IT и бизнес-проектами, планирование спринтов, "
            "ресурсы и риски, Agile, Scrum и Kanban методологии"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },
    "agile_scrum": {
        "aliases": ["agile", "scrum", "эджайл", "скрам", "спринт", "дейли", "стендап"],
        "enriched_text": (
            "гибкие методологии Agile и Scrum, спринты и ежедневные стендапы, "
            "ретроспективы и планирование итераций"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },
    "kanban_method": {
        "aliases": ["канбан", "kanban", "доска задач", "wip", "поток"],
        "enriched_text": (
            "методология Kanban, визуализация рабочего потока, "
            "ограничение незавершенной работы, непрерывное улучшение"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },
    "safe_framework": {
        "aliases": ["safe", "сейф", "scaled agile", "масштабирование agile"],
        "enriched_text": (
            "фреймворк SAFe для масштабирования Agile на большие организации, "
            "Program Increment, Agile Release Train"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },
    "product_management": {
        "aliases": ["продакт менеджмент", "product manager", "управление продуктом", "роадмап"],
        "enriched_text": (
            "продакт-менеджмент, развитие цифровых продуктов, "
            "CustDev и проверка гипотез, приоритизация фич и метрики"
        ),
        "category": "work",
        "subcategory": "Продакт-менеджмент",
        "parent": "Саморазвитие",
    },
    "product_discovery": {
        "aliases": ["discovery", "кастдев", "custdev", "исследование продукта", "юзер ресерч"],
        "enriched_text": (
            "продуктовые исследования и CustDev, глубинные интервью, "
            "проверка гипотез и поиск проблем пользователей"
        ),
        "category": "work",
        "subcategory": "Продакт-менеджмент",
        "parent": "Саморазвитие",
    },
    "roadmapping": {
        "aliases": ["роадмап", "roadmap", "дорожная карта", "стратегия продукта"],
        "enriched_text": (
            "создание и управление дорожной картой продукта, "
            "стратегические цели, квартальное планирование, коммуникация видения"
        ),
        "category": "work",
        "subcategory": "Продакт-менеджмент",
        "parent": "Саморазвитие",
    },
    "team_leadership": {
        "aliases": ["лидерство", "управление людьми", "тимбилдинг", "one on one"],
        "enriched_text": (
            "лидерство в команде, мотивация сотрудников и обратная связь, "
            "one-on-one встречи, развитие и наставничество"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },
    "one_on_one": {
        "aliases": ["1:1", "one on one", "ван-он-ван", "индивидуальная встреча", "персональный митинг"],
        "enriched_text": (
            "регулярные встречи 1:1 с руководителем, обсуждение прогресса, "
            "карьерные цели и личное развитие, обратная связь"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },
    "negotiations": {
        "aliases": ["переговоры", "деловые переговоры", "win-win", "согласование"],
        "enriched_text": (
            "ведение деловых переговоров, стратегии win-win, "
            "управление конфликтами и поиск компромиссов"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },
    "okr_goals": {
        "aliases": ["okr", "цели и ключевые результаты", "постановка целей", "kpi"],
        "enriched_text": (
            "методология OKR, постановка амбициозных целей и измеримых результатов, "
            "каскадирование целей в организации"
        ),
        "category": "work",
        "subcategory": "Менеджмент и Лидерство",
        "parent": "Саморазвитие",
    },

    "time_productivity": {
        "aliases": ["тайм-менеджмент", "управление временем", "продуктивность", "gtd", "pomodoro"],
        "enriched_text": (
            "управление временем и личная продуктивность, метод GTD и Pomodoro, "
            "Notion и Obsidian, Second Brain и цифровой минимализм"
        ),
        "category": "life",
        "subcategory": "Тайм-менеджмент",
        "parent": "Саморазвитие",
    },
    "gtd_method": {
        "aliases": ["gtd", "getting things done", "доведение дел до конца", "инбокс зеро"],
        "enriched_text": (
            "методология Getting Things Done, сбор задач в инбокс, "
            "контексты и проекты, еженедельный обзор"
        ),
        "category": "life",
        "subcategory": "Тайм-менеджмент",
        "parent": "Саморазвитие",
    },
    "pomodoro_technique": {
        "aliases": ["pomodoro", "помидорная техника", "помидор", "таймер фокуса"],
        "enriched_text": (
            "техника Pomodoro для концентрации, работа интервалами по 25 минут, "
            "короткие перерывы и борьба с прокрастинацией"
        ),
        "category": "life",
        "subcategory": "Тайм-менеджмент",
        "parent": "Саморазвитие",
    },
    "digital_minimalism": {
        "aliases": ["цифровой минимализм", "уменьшение экранного времени", "детокс"],
        "enriched_text": (
            "цифровой минимализм, осознанное использование технологий, "
            "уменьшение времени в телефоне, информационная диета"
        ),
        "category": "life",
        "subcategory": "Тайм-менеджмент",
        "parent": "Саморазвитие",
    },
    "notion_obsidian": {
        "aliases": ["notion", "ноушен", "obsidian", "обсидиан", "база знаний", "заметки"],
        "enriched_text": (
            "инструменты для ведения заметок Notion и Obsidian, "
            "создание базы знаний, связывание идей, личная вики"
        ),
        "category": "life",
        "subcategory": "Тайм-менеджмент",
        "parent": "Саморазвитие",
    },
    "second_brain": {
        "aliases": ["second brain", "второй мозг", "para method", "система хранения знаний"],
        "enriched_text": (
            "концепция Second Brain, метод PARA для организации информации, "
            "управление личными знаниями и креативность"
        ),
        "category": "life",
        "subcategory": "Тайм-менеджмент",
        "parent": "Саморазвитие",
    },

    "soft_skills_empathy": {
        "aliases": ["софт скиллы", "мягкие навыки", "эмпатия", "общение", "выступление"],
        "enriched_text": (
            "развитие мягких навыков, эмоциональный интеллект и эмпатия, "
            "публичные выступления и сторителлинг, нетворкинг"
        ),
        "category": "life",
        "subcategory": "Soft Skills и Эмпатия",
        "parent": "Саморазвитие",
    },
    "emotional_intelligence": {
        "aliases": ["эмоциональный интеллект", "eq", "эмпатия", "понимание эмоций"],
        "enriched_text": (
            "развитие эмоционального интеллекта, распознавание своих и чужих эмоций, "
            "саморегуляция и эмпатическое общение"
        ),
        "category": "life",
        "subcategory": "Soft Skills и Эмпатия",
        "parent": "Саморазвитие",
    },
    "public_speaking": {
        "aliases": ["публичные выступления", "спикер", "ораторское искусство", "выступление"],
        "enriched_text": (
            "мастерство публичных выступлений, подготовка речи и слайдов, "
            "борьба со страхом сцены, работа с аудиторией"
        ),
        "category": "life",
        "subcategory": "Soft Skills и Эмпатия",
        "parent": "Саморазвитие",
    },
    "storytelling_skill": {
        "aliases": ["сторителлинг", "рассказывание историй", "нарратив", "питч"],
        "enriched_text": (
            "искусство сторителлинга, построение увлекательного нарратива, "
            "структура героя и конфликта, убеждение через историю"
        ),
        "category": "life",
        "subcategory": "Soft Skills и Эмпатия",
        "parent": "Саморазвитие",
    },
    "networking": {
        "aliases": ["нетворкинг", "связи", "знакомства", "деловое общение", "комьюнити"],
        "enriched_text": (
            "построение сети профессиональных контактов, митапы и конференции, "
            "поддержание отношений, социальный капитал"
        ),
        "category": "life",
        "subcategory": "Soft Skills и Эмпатия",
        "parent": "Саморазвитие",
    },
    "conflict_resolution": {
        "aliases": ["конфликтология", "разрешение конфликтов", "медиация", "ссора"],
        "enriched_text": (
            "навыки разрешения конфликтов, управление сложными разговорами, "
            "активное слушание и медиация, ненасильственное общение"
        ),
        "category": "life",
        "subcategory": "Soft Skills и Эмпатия",
        "parent": "Саморазвитие",
    },
    "mentoring_coaching": {
        "aliases": ["менторинг", "коучинг", "наставничество", "ментор", "коуч"],
        "enriched_text": (
            "менторинг и коучинг, развитие людей и помощь в карьере, "
            "постановка целей, развивающая обратная связь"
        ),
        "category": "work",
        "subcategory": "Soft Skills и Эмпатия",
        "parent": "Саморазвитие",
    },

    "learning_skills": {
        "aliases": ["навыки обучения", "самообразование", "учиться учиться", "метанавыки"],
        "enriched_text": (
            "метанавыки обучения, скорочтение и мнемотехники, "
            "метод Zettelkasten, изучение иностранных языков"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },
    "speed_reading": {
        "aliases": ["скорочтение", "быстрое чтение", "чтение по диагонали", "подавление субвокализации"],
        "enriched_text": (
            "техники скорочтения, расширение поля зрения, "
            "подавление мысленного проговаривания, увеличение скорости чтения"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },
    "memory_palace": {
        "aliases": ["мнемотехника", "дворец памяти", "memory palace", "запоминание"],
        "enriched_text": (
            "мнемотехники и метод дворца памяти, визуализация и ассоциации, "
            "запоминание больших объемов информации"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },
    "zettelkasten": {
        "aliases": ["zettelkasten", "цеттелькастен", "картотека", "умные заметки", "луман"],
        "enriched_text": (
            "метод ведения заметок Zettelkasten Никласа Лумана, "
            "атомарные заметки и перекрестные ссылки для генерации идей"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },
    "languages_learning": {
        "aliases": ["изучение языков", "иняз", "полиглот", "языковой барьер", "дуолинго"],
        "enriched_text": (
            "изучение иностранных языков, методики интервального повторения, "
            "погружение в среду и языковой обмен"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },
    "english_lang": {
        "aliases": ["английский", "english", "инглиш", "учу английский", "toefl", "ielts"],
        "enriched_text": (
            "изучение английского языка, грамматика и лексика, "
            "подготовка к международным экзаменам, разговорная практика"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },
    "german_lang": {
        "aliases": ["немецкий", "deutsch", "учу немецкий", "германия", "гете"],
        "enriched_text": (
            "изучение немецкого языка, сложная грамматика и падежи, "
            "подготовка к Goethe-Zertifikat и переезду"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },
    "japanese_lang": {
        "aliases": ["японский", "日本語", "учу японский", "хирагана", "кандзи"],
        "enriched_text": (
            "изучение японского языка, азбуки хирагана и катакана, "
            "иероглифы кандзи, уровни JLPT N5-N1"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },
    "spanish_lang": {
        "aliases": ["испанский", "español", "учу испанский", "сервантес", "латинос"],
        "enriched_text": (
            "изучение испанского языка, глагольные спряжения, "
            "культура испаноязычных стран, DELE экзамен"
        ),
        "category": "life",
        "subcategory": "Саморазвитие",
        "parent": "Саморазвитие",
    },

    # ============================================================
    # 6. ПСИХОЛОГИЯ И ОТНОШЕНИЯ
    # ============================================================
    "psychotherapy": {
        "aliases": ["психотерапия", "терапия", "психолог", "психоанализ", "кпт", "гештальт"],
        "enriched_text": (
            "психотерапия и психологическая помощь, КПТ и гештальт-терапия, "
            "психоанализ, схема-терапия и другие подходы"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },
    "cbt_therapy": {
        "aliases": ["кпт", "когнитивно-поведенческая", "cbt", "автоматические мысли", "поведение"],
        "enriched_text": (
            "когнитивно-поведенческая терапия, работа с автоматическими мыслями, "
            "изменение неадаптивных паттернов поведения и мышления"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },
    "dbt_skills": {
        "aliases": ["дпт", "диалектическая поведенческая", "dbt", "навыки", "эмоциональная регуляция"],
        "enriched_text": (
            "диалектическая поведенческая терапия DBT, навыки стрессоустойчивости, "
            "эмоциональная регуляция и межличностная эффективность"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },
    "gestalt_therapy": {
        "aliases": ["гештальт", "gestalt", "здесь и сейчас", "незавершенный гештальт"],
        "enriched_text": (
            "гештальт-терапия, осознание чувств здесь и сейчас, "
            "завершение гештальтов, работа с телом и эмоциями"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },
    "psychoanalysis": {
        "aliases": ["психоанализ", "фрейд", "юнг", "бессознательное", "либидо"],
        "enriched_text": (
            "классический психоанализ, учение Фрейда и Юнга, "
            "толкование сновидений и работа с бессознательным"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },
    "act_therapy": {
        "aliases": ["act", "терапия принятия", "принятие и ответственность", "act психотерапия"],
        "enriched_text": (
            "терапия принятия и ответственности ACT, психологическая гибкость, "
            "разделение с мыслями и движение к ценностям"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },
    "positive_psychology": {
        "aliases": ["позитивная психология", "счастье", "благополучие", "сильные стороны"],
        "enriched_text": (
            "позитивная психология, наука о счастье и благополучии, "
            "развитие сильных сторон характера и оптимизм"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },
    "neuropsychology": {
        "aliases": ["нейропсихология", "мозг", "нейроны", "когнитивные функции"],
        "enriched_text": (
            "нейропсихология, связь мозга и поведения, "
            "высшие психические функции и их нарушения"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },
    "schema_therapy": {
        "aliases": ["схема-терапия", "ранние дезадаптивные схемы", "режимы"],
        "enriched_text": (
            "схема-терапия Джеффри Янга, работа с глубинными убеждениями, "
            "детские режимы и копинговые стратегии"
        ),
        "category": "life",
        "subcategory": "Психотерапия и Наука",
        "parent": "Психология и Отношения",
    },

    "mindfulness_meditation": {
        "aliases": ["осознанность", "медитация", "mindfulness", "випассана", "дыхание"],
        "enriched_text": (
            "практики осознанности и медитации, випассана и трансцендентальная медитация, "
            "дыхательные упражнения, метод Вима Хофа"
        ),
        "category": "life",
        "subcategory": "Осознанность и Медитация",
        "parent": "Психология и Отношения",
    },
    "vipassana": {
        "aliases": ["випассана", "vipassana", "ретрит", "медитация прозрения", "10 дней"],
        "enriched_text": (
            "техника медитации випассана, наблюдение ощущений в теле, "
            "десятидневные ретриты молчания, очищение ума"
        ),
        "category": "life",
        "subcategory": "Осознанность и Медитация",
        "parent": "Психология и Отношения",
    },
    "transcendental_med": {
        "aliases": ["трансцендентальная медитация", "тм", "мантра", "махариши"],
        "enriched_text": (
            "трансцендентальная медитация, повторение личной мантры, "
            "глубокое расслабление и снятие стресса по 20 минут"
        ),
        "category": "life",
        "subcategory": "Осознанность и Медитация",
        "parent": "Психология и Отношения",
    },
    "yoga_nidra": {
        "aliases": ["йога-нидра", "yoga nidra", "йогический сон", "глубокая релаксация"],
        "enriched_text": (
            "практика йога-нидры, осознанный сон и глубокая релаксация, "
            "сканирование тела и санкальпа"
        ),
        "category": "life",
        "subcategory": "Осознанность и Медитация",
        "parent": "Психология и Отношения",
    },
    "breathwork": {
        "aliases": ["дыхательные практики", "breathwork", "пранаяма", "холотропное дыхание"],
        "enriched_text": (
            "осознанные дыхательные практики, пранаяма и холотропное дыхание, "
            "метод Вима Хофа, управление состоянием через дыхание"
        ),
        "category": "life",
        "subcategory": "Осознанность и Медитация",
        "parent": "Психология и Отношения",
    },
    "wim_hof_method": {
        "aliases": ["вим хоф", "wim hof", "ледяной человек", "холод", "дыхание вима хофа"],
        "enriched_text": (
            "метод Вима Хофа, сочетание дыхательных техник и холодовых тренировок, "
            "укрепление иммунитета и контроль вегетативной системы"
        ),
        "category": "life",
        "subcategory": "Осознанность и Медитация",
        "parent": "Психология и Отношения",
    },
    "gratitude_practice": {
        "aliases": ["благодарность", "gratitude", "дневник благодарности", "спасибо"],
        "enriched_text": (
            "практика благодарности, ежедневное отмечание позитивных моментов, "
            "повышение уровня счастья и снижение тревожности"
        ),
        "category": "life",
        "subcategory": "Осознанность и Медитация",
        "parent": "Психология и Отношения",
    },
    "mindful_walking": {
        "aliases": ["медитация при ходьбе", "осознанная ходьба", "прогулка внимательности"],
        "enriched_text": (
            "практика медитации при ходьбе, концентрация на шагах и дыхании, "
            "осознанное движение в повседневности"
        ),
        "category": "life",
        "subcategory": "Осознанность и Медитация",
        "parent": "Психология и Отношения",
    },

    "relationships_comm": {
        "aliases": ["отношения", "любовь", "пара", "коммуникация", "семья", "дружба"],
        "enriched_text": (
            "психология отношений, романтические и семейные отношения, "
            "теория привязанности, ненасильственное общение, воспитание детей"
        ),
        "category": "life",
        "subcategory": "Отношения",
        "parent": "Психология и Отношения",
    },
    "romantic_relations": {
        "aliases": ["романтические отношения", "любовь", "партнер", "свидания", "отношения в паре"],
        "enriched_text": (
            "романтические отношения, построение близости и доверия, "
            "языки любви и совместные ценности, решение конфликтов в паре"
        ),
        "category": "life",
        "subcategory": "Отношения",
        "parent": "Психология и Отношения",
    },
    "love_languages": {
        "aliases": ["языки любви", "5 языков", "love languages", "слова поощрения", "прикосновения"],
        "enriched_text": (
            "концепция пяти языков любви по Гэри Чепмену, "
            "слова поощрения, время, подарки, помощь и прикосновения"
        ),
        "category": "life",
        "subcategory": "Отношения",
        "parent": "Психология и Отношения",
    },
    "attachment_theory": {
        "aliases": ["теория привязанности", "типы привязанности", "надежный тип", "тревожный тип"],
        "enriched_text": (
            "теория привязанности Боулби, типы привязанности, "
            "влияние детского опыта на взрослые отношения"
        ),
        "category": "life",
        "subcategory": "Отношения",
        "parent": "Психология и Отношения",
    },
    "nonviolent_communication": {
        "aliases": ["ненасильственное общение", "нно", "nvc", "я-сообщения", "жираф"],
        "enriched_text": (
            "метод ненасильственного общения Маршалла Розенберга, "
            "язык чувств и потребностей, эмпатический диалог"
        ),
        "category": "life",
        "subcategory": "Отношения",
        "parent": "Психология и Отношения",
    },
    "parenting_psychology": {
        "aliases": ["воспитание", "дети", "родительство", "детская психология", "привязанность"],
        "enriched_text": (
            "психология воспитания детей, теории развития и привязанности, "
            "позитивное родительство и установление границ"
        ),
        "category": "life",
        "subcategory": "Отношения",
        "parent": "Психология и Отношения",
    },
    "friendship_psych": {
        "aliases": ["дружба", "друзья", "социальная поддержка", "токсичные друзья", "братан"],
        "enriched_text": (
            "психология дружбы, построение и поддержание дружеских связей, "
            "социальный круг и эмоциональная опора"
        ),
        "category": "life",
        "subcategory": "Отношения",
        "parent": "Психология и Отношения",
    },

    # ============================================================
    # 7. СПОРТ И АКТИВНЫЙ ОТДЫХ
    # ============================================================
    "cycling_bikes": {
        "aliases": ["велосипед", "велоспорт", "вел", "покатушки", "байк", "велосипедист"],
        "enriched_text": (
            "велосипеды и велоспорт, шоссе и MTB, грэвел-байки, "
            "велопутешествия и байкпакинг, ремонт и обслуживание"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },
    "road_bike": {
        "aliases": ["шоссе", "шоссейник", "road bike", "шоссейный велосипед", "разделка"],
        "enriched_text": (
            "шоссейный велоспорт, аэродинамика и скорость, "
            "групповая езда в пелотоне, карбоновые рамы и обвес"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },
    "aero_cycling": {
        "aliases": ["аэродинамика", "аэро посадка", "разделочный шлем", "велогонка на время"],
        "enriched_text": (
            "аэродинамика в велоспорте, оптимизация посадки и экипировки, "
            "разделочные велосипеды и дисковые колеса"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },
    "mtb_trail": {
        "aliases": ["mtb", "горный велосипед", "маунтинбайк", "трейлы", "эндуро"],
        "enriched_text": (
            "горный велосипед MTB, катание по трейлам и бездорожью, "
            "эндуро и кросс-кантри, амортизация и дропы"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },
    "downhill_mtb": {
        "aliases": ["даунхилл", "downhill", "дх", "скоростной спуск", "байк-парк"],
        "enriched_text": (
            "экстремальный даунхилл на MTB, скоростные спуски по крутым трассам, "
            "защита и шлем full-face, подъемники в байк-парках"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },
    "gravel_bike": {
        "aliases": ["грэвел", "gravel", "гравийный велосипед", "велосипед-приключение"],
        "enriched_text": (
            "грэвел-байки для смешанных покрытий, асфальт и грунтовки, "
            "дальние заезды с багажом, универсальность и выносливость"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },
    "bike_repair": {
        "aliases": ["ремонт велосипеда", "веломеханик", "настройка передач", "камера"],
        "enriched_text": (
            "обслуживание и ремонт велосипедов, настройка трансмиссии и тормозов, "
            "замена камер и покрышек, протяжка спиц"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },
    "bike_touring": {
        "aliases": ["велопутешествие", "велотуризм", "байкпакинг", "bikepacking"],
        "enriched_text": (
            "путешествия на велосипеде, байкпакинг с легким снаряжением, "
            "ночевки в палатке и автономные маршруты"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },
    "bikepacking": {
        "aliases": ["bikepacking", "байкпакинг", "вело с рюкзаком", "легкий велотуризм"],
        "enriched_text": (
            "байкпакинг, автономные велопутешествия с минимальным снаряжением, "
            "велосумки и бивуак, удаленные маршруты"
        ),
        "category": "life",
        "subcategory": "Велосипеды",
        "parent": "Спорт и Активный отдых",
    },

    "gym_fitness": {
        "aliases": ["зал", "качалка", "тренажерный зал", "железо", "гантели", "штанга"],
        "enriched_text": (
            "тренировки в тренажерном зале, бодибилдинг и пауэрлифтинг, "
            "кроссфит, калистеника, стретчинг и гиревой спорт"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },
    "bodybuilding": {
        "aliases": ["бодибилдинг", "культуризм", "качаюсь", "мышцы", "бицуха", "банки"],
        "enriched_text": (
            "бодибилдинг, наращивание мышечной массы и рельефа, "
            "тренировки на гипертрофию, спортивное питание и протеин"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },
    "natural_bodybuilding": {
        "aliases": ["натуральный бодибилдинг", "нат", "без химии", "естественный"],
        "enriched_text": (
            "натуральный бодибилдинг без применения допинга, "
            "максимальный результат через тренинг и питание"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },
    "powerlifting": {
        "aliases": ["пауэрлифтинг", "силовое троеборье", "присед", "жим", "тяга", "пауэр"],
        "enriched_text": (
            "пауэрлифтинг, соревнования по приседу, жиму лежа и становой тяге, "
            "развитие максимальной силы и работа с весами"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },
    "crossfit_training": {
        "aliases": ["кроссфит", "crossfit", "wod", "функциональный тренинг", "бокс"],
        "enriched_text": (
            "высокоинтенсивные тренировки кроссфит, WOD дня, "
            "тяжелая атлетика и гимнастика, развитие общей физподготовки"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },
    "calisthenics": {
        "aliases": ["калистеника", "воркаут", "брусья", "турник", "вес тела"],
        "enriched_text": (
            "калистеника и стрит воркаут, тренировки с весом собственного тела, "
            "подтягивания, отжимания на брусьях, выходы силой"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },
    "street_workout": {
        "aliases": ["стрит воркаут", "уличные тренировки", "площадка", "горизонт"],
        "enriched_text": (
            "уличный воркаут на турниках и брусьях, статические элементы, "
            "передний вис и флажок, дворовые соревнования"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },
    "stretching_flex": {
        "aliases": ["стретчинг", "растяжка", "гибкость", "шпагат", "флексибилити"],
        "enriched_text": (
            "стретчинг и развитие гибкости, динамическая и статическая растяжка, "
            "шпагаты и мостик, расслабление мышц после тренировки"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },
    "kettlebell_sport": {
        "aliases": ["гиревой спорт", "гиря", "рывок", "толчок", "kettlebell"],
        "enriched_text": (
            "гиревой спорт, соревнования в рывке и толчке гирь, "
            "развитие силовой выносливости и техники дыхания"
        ),
        "category": "life",
        "subcategory": "Тренажерный зал",
        "parent": "Спорт и Активный отдых",
    },

    "running_races": {
        "aliases": ["бег", "пробежка", "марафон", "трейлраннинг", "паркран", "кросс"],
        "enriched_text": (
            "бег и участие в гонках, марафон и ультрадистанции, "
            "трейлраннинг по пересеченной местности, parkrun 5k"
        ),
        "category": "life",
        "subcategory": "Бег и Гонки",
        "parent": "Спорт и Активный отдых",
    },
    "marathon_ultra": {
        "aliases": ["марафон", "ультра", "42 км", "ultramarathon", "бег на длинные дистанции"],
        "enriched_text": (
            "марафонский бег и ультра-гонки, подготовка к дистанции 42 и 100+ км, "
            "питание на трассе и восполнение электролитов"
        ),
        "category": "life",
        "subcategory": "Бег и Гонки",
        "parent": "Спорт и Активный отдых",
    },
    "trail_running": {
        "aliases": ["трейлраннинг", "бег по горам", "трейл", "пересеченка", "кросс"],
        "enriched_text": (
            "бег по трейлам и горной местности, техничный спуск и набор высоты, "
            "специальная обувь и трейловые палки"
        ),
        "category": "life",
        "subcategory": "Бег и Гонки",
        "parent": "Спорт и Активный отдых",
    },
    "parkrun_5k": {
        "aliases": ["паркран", "parkrun", "5к", "5 км", "бесплатный забег"],
        "enriched_text": (
            "еженедельные бесплатные забеги parkrun на 5 км, "
            "беговое сообщество, фиксация личных рекордов по времени"
        ),
        "category": "life",
        "subcategory": "Бег и Гонки",
        "parent": "Спорт и Активный отдых",
    },
    "minimalist_running": {
        "aliases": ["минималистичный бег", "босиком", "vibram five fingers", "естественный бег"],
        "enriched_text": (
            "бег в минималистичной обуви или босиком, техника приземления на переднюю часть стопы, "
            "укрепление свода стопы"
        ),
        "category": "life",
        "subcategory": "Бег и Гонки",
        "parent": "Спорт и Активный отдых",
    },
    "triathlon_event": {
        "aliases": ["триатлон", "ironman", "айронмен", "плавание вело бег"],
        "enriched_text": (
            "триатлон, три вида спорта подряд: плавание, велогонка, бег, "
            "Ironman и олимпийские дистанции, транзитная зона"
        ),
        "category": "life",
        "subcategory": "Бег и Гонки",
        "parent": "Спорт и Активный отдых",
    },

    "travel_adventure": {
        "aliases": ["путешествия", "трип", "отпуск", "поездка", "тревел", "другие страны"],
        "enriched_text": (
            "путешествия и приключения, бэкпекинг и вэнлайф, "
            "цифровой кочевник, урбанистика и тревел-фотография"
        ),
        "category": "life",
        "subcategory": "Путешествия",
        "parent": "Спорт и Активный отдых",
    },
    "backpacking_budget": {
        "aliases": ["бэкпекинг", "рюкзак", "бюджетные путешествия", "хостел", "автостоп"],
        "enriched_text": (
            "самостоятельные бюджетные путешествия с рюкзаком, "
            "ночевки в хостелах и каучсерфинг, поиск дешевых билетов"
        ),
        "category": "life",
        "subcategory": "Путешествия",
        "parent": "Спорт и Активный отдых",
    },
    "vanlife_nomad": {
        "aliases": ["вэнлайф", "дом на колесах", "автодом", "кемпер", "свобода"],
        "enriched_text": (
            "образ жизни vanlife, путешествия в автодоме и кемпере, "
            "автономность и минимализм, работа в дороге"
        ),
        "category": "life",
        "subcategory": "Путешествия",
        "parent": "Спорт и Активный отдых",
    },
    "urban_exploration": {
        "aliases": ["урбанистика", "заброшки", "руфинг", "городские исследования"],
        "enriched_text": (
            "исследование городских пространств, заброшенные здания и крыши, "
            "промышленная архитектура и сталкер-эстетика"
        ),
        "category": "life",
        "subcategory": "Путешествия",
        "parent": "Спорт и Активный отдых",
    },
    "travel_photography": {
        "aliases": ["трэвел-фотография", "travel photo", "фото в путешествии", "инстаграм"],
        "enriched_text": (
            "съемка фотографий в путешествиях, запечатление культур и пейзажей, "
            "ведение трэвел-блога, обработка в Lightroom"
        ),
        "category": "life",
        "subcategory": "Путешествия",
        "parent": "Спорт и Активный отдых",
    },
    "digital_nomad": {
        "aliases": ["цифровой кочевник", "digital nomad", "удаленка", "работа из любой точки"],
        "enriched_text": (
            "стиль жизни цифрового кочевника, удаленная работа из путешествий, "
            "баланс работы и свободы, коворкинги в Азии"
        ),
        "category": "work",
        "subcategory": "Путешествия",
        "parent": "Спорт и Активный отдых",
    },
    "eco_tourism": {
        "aliases": ["экотуризм", "зеленый туризм", "устойчивое путешествие", "волонтерство"],
        "enriched_text": (
            "экологический туризм, путешествия с заботой о природе, "
            "волонтерские программы и наблюдение за дикой природой"
        ),
        "category": "life",
        "subcategory": "Путешествия",
        "parent": "Спорт и Активный отдых",
    },

    "nature_outdoor": {
        "aliases": ["природа", "туризм", "походы", "горы", "лес", "река", "костер"],
        "enriched_text": (
            "отдых на природе и активный туризм, хайкинг и альпинизм, "
            "кемпинг и бушкрафт, скалолазание и водные виды спорта"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "hiking_trekking": {
        "aliases": ["хайкинг", "трекинг", "поход", "пеший туризм", "тропа"],
        "enriched_text": (
            "хайкинг и трекинг, пешие походы по размеченным маршрутам, "
            "дневные и многодневные переходы, трекинговые палки"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "mountaineering": {
        "aliases": ["альпинизм", "восхождение", "вершина", "кошки", "ледоруб"],
        "enriched_text": (
            "альпинизм и горные восхождения, работа в связке на леднике, "
            "технически сложные маршруты, акклиматизация на высоте"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "ice_climbing": {
        "aliases": ["ледолазание", "замерзший водопад", "айс-фифи", "кошки"],
        "enriched_text": (
            "экстремальное ледолазание, подъем по замерзшим водопадам и стенам, "
            "специальные кошки и ледовые инструменты"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "camping_bushcraft": {
        "aliases": ["кемпинг", "бушкрафт", "палатка", "спальник", "примус"],
        "enriched_text": (
            "кемпинг и навыки выживания бушкрафт, установка лагеря, "
            "разведение костра, приготовление пищи на огне"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "hammock_camping": {
        "aliases": ["гамачный кемпинг", "гамак", "hammock", "подвесная система"],
        "enriched_text": (
            "ночевка в гамаке в походе, легкость и компактность, "
            "тент и москитная сетка, подвесная система для леса"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "rock_climbing": {
        "aliases": ["скалолазание", "лазание", "скалы", "веревка", "оттяжка"],
        "enriched_text": (
            "спортивное скалолазание на естественном рельефе, "
            "пробивка трасс и работа с веревкой, мультипитчи"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "bouldering_indoor": {
        "aliases": ["боулдеринг", "bouldering", "скалодром", "трассы", "зацепки"],
        "enriched_text": (
            "боулдеринг на скалодроме и на камнях, короткие мощные трассы без веревки, "
            "обсуждение зацепов и техники движений"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "kayaking_rafting": {
        "aliases": ["каякинг", "рафтинг", "каяк", "сплав", "порог", "весло"],
        "enriched_text": (
            "водный туризм на каяках и рафтах, сплав по горным рекам, "
            "прохождение порогов и техника гребли"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "skiing_snowboard": {
        "aliases": ["лыжи", "сноуборд", "горные лыжи", "борд", "фрирайд"],
        "enriched_text": (
            "катание на горных лыжах и сноуборде, трассы и фрирайд, "
            "бэккантри вне подготовленных склонов"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "backcountry_ski": {
        "aliases": ["бэккантри", "скитур", "лыжи вне трасс", "лавинное снаряжение"],
        "enriched_text": (
            "катание на лыжах вне подготовленных трасс, подъем в гору пешком, "
            "лавинная безопасность и биперы"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },
    "surfing_sports": {
        "aliases": ["серфинг", "сапбординг", "sup", "доска", "волна", "океан"],
        "enriched_text": (
            "серфинг и сапбординг, катание на волнах и спокойной воде, "
            "выбор доски, воск и развороты на гребне"
        ),
        "category": "life",
        "subcategory": "Природа и Туризм",
        "parent": "Спорт и Активный отдых",
    },

    "yoga_pilates": {
        "aliases": ["йога", "пилатес", "асаны", "коврик", "практика"],
        "enriched_text": (
            "йога и пилатес, хатха и аштанга-виньяса, "
            "акройога и кундалини-йога, работа с телом и дыханием"
        ),
        "category": "life",
        "subcategory": "Йога и Пилатес",
        "parent": "Спорт и Активный отдых",
    },
    "hatha_yoga": {
        "aliases": ["хатха", "hatha", "классическая йога", "асаны"],
        "enriched_text": (
            "хатха-йога, статические позы и дыхание, "
            "баланс и гибкость, медленный темп практики"
        ),
        "category": "life",
        "subcategory": "Йога и Пилатес",
        "parent": "Спорт и Активный отдых",
    },
    "ashtanga_vinyasa": {
        "aliases": ["аштанга", "виньяса", "динамическая йога", "серии"],
        "enriched_text": (
            "динамические стили йоги аштанга и виньяса, связки асан и дыхание, "
            "силовая практика в потоке, построение тепла в теле"
        ),
        "category": "life",
        "subcategory": "Йога и Пилатес",
        "parent": "Спорт и Активный отдых",
    },
    "pilates_mat": {
        "aliases": ["пилатес", "pilates", "мат", "центр", "кор"],
        "enriched_text": (
            "система упражнений пилатес на коврике, укрепление мышц кора и спины, "
            "контроль движений и правильное дыхание"
        ),
        "category": "life",
        "subcategory": "Йога и Пилатес",
        "parent": "Спорт и Активный отдых",
    },
    "acroyoga": {
        "aliases": ["акройога", "парная йога", "акробатика", "база", "флайер"],
        "enriched_text": (
            "акройога, сочетание йоги и акробатики в паре, "
            "база и летящий, баланс и доверие в движении"
        ),
        "category": "life",
        "subcategory": "Йога и Пилатес",
        "parent": "Спорт и Активный отдых",
    },
    "kundalini_yoga": {
        "aliases": ["кундалини", "kundalini", "крии", "мантры", "пробуждение"],
        "enriched_text": (
            "кундалини-йога, динамические крийи и мантры, "
            "работа с энергией кундалини, белые одежды и тюрбаны"
        ),
        "category": "life",
        "subcategory": "Йога и Пилатес",
        "parent": "Спорт и Активный отдых",
    },

    # ============================================================
    # 8. ДОМ И ОБРАЗ ЖИЗНИ
    # ============================================================
    "cooking_culinary": {
        "aliases": ["кулинария", "готовка", "кухня", "рецепты", "повар", "вкусно"],
        "enriched_text": (
            "кулинария и приготовление пищи, выпечка хлеба и ферментация, "
            "specialty кофе и чайная культура, молекулярная кухня и BBQ"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "baking_bread": {
        "aliases": ["выпечка", "хлеб", "печь хлеб", "тесто", "булочки", "круассаны"],
        "enriched_text": (
            "выпечка домашнего хлеба и сдобы, дрожжевое и бездрожжевое тесто, "
            "хлеб на закваске, формовка и расстойка"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "sourdough_starters": {
        "aliases": ["закваска", "sourdough", "стартер", "живая закваска", "хлеб на опаре"],
        "enriched_text": (
            "выращивание и уход за хлебной закваской, sourdough starter, "
            "кормление мукой и водой, ароматный хлеб с хрустящей корочкой"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "fermentation": {
        "aliases": ["ферментация", "квашение", "кимчи", "соленья", "комбуча"],
        "enriched_text": (
            "ферментация продуктов, квашение капусты и овощей, "
            "приготовление кимчи и чайного гриба комбучи"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "kombucha_brewing": {
        "aliases": ["комбуча", "kombucha", "чайный гриб", "скоби", "вторая ферментация"],
        "enriched_text": (
            "домашнее приготовление комбучи, чайный гриб и SCOBY, "
            "вторая ферментация для карбонизации и вкуса"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "vegan_cuisine": {
        "aliases": ["веганская кухня", "веганство", "растительное питание", "тофу", "сейтан"],
        "enriched_text": (
            "веганская кулинария без продуктов животного происхождения, "
            "альтернативы мясу и молоку, этичное питание"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "specialty_coffee": {
        "aliases": ["specialty coffee", "кофе", "спешелти", "зерно", "эспрессо"],
        "enriched_text": (
            "культура спешелти кофе, методы заваривания воронка и аэропресс, "
            "обжарка зерна, вкусовые дескрипторы, латте-арт"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "coffee_brewing_methods": {
        "aliases": ["заваривание кофе", "v60", "аэропресс", "френч пресс", "кемекс"],
        "enriched_text": (
            "альтернативные методы заваривания кофе, воронка Hario V60, "
            "Aeropress, френч-пресс и кемекс, помол и температура воды"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "coffee_roasting": {
        "aliases": ["обжарка кофе", "рост кофе", "зеленое зерно", "ростер"],
        "enriched_text": (
            "домашняя обжарка зеленого кофейного зерна, "
            "степени обжарки от лайт до дарк, дегазация после роста"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "tea_culture": {
        "aliases": ["чай", "чайная культура", "tea", "чайная церемония", "чаепитие"],
        "enriched_text": (
            "чайная культура и дегустация, китайский и японский чай, "
            "матча и традиционная церемония, заваривание пуэра и улуна"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "matcha_ceremony": {
        "aliases": ["матча", "matcha", "чайная церемония", "тяною", "венчик"],
        "enriched_text": (
            "японская чайная церемония с порошковым чаем матча, "
            "приготовление венчиком часэн, дзен и медитативный процесс"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "molecular_gastronomy": {
        "aliases": ["молекулярная кухня", "креативная кулинария", "эспума", "азот"],
        "enriched_text": (
            "молекулярная гастрономия, научный подход к приготовлению еды, "
            "эспумы, сферификация и жидкий азот"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "bbq_grilling": {
        "aliases": ["барбекю", "гриль", "шашлык", "копчение", "стейк", "мангал"],
        "enriched_text": (
            "приготовление еды на гриле и барбекю, стейки и бургеры, "
            "угли и щепа для копчения, соусы и маринады"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },
    "sous_vide_cooking": {
        "aliases": ["sous vide", "су-вид", "вакуум", "точная температура", "погружной термостат"],
        "enriched_text": (
            "технология приготовления sous vide в вакууме, "
            "контроль точной температуры, сочное мясо и рыба"
        ),
        "category": "life",
        "subcategory": "Кулинария",
        "parent": "Дом и Образ жизни",
    },

    "diy_maker": {
        "aliases": ["diy", "сделай сам", "мастерю", "рукоделие", "хендмейд", "изобретаю"],
        "enriched_text": (
            "DIY и творчество своими руками, столярное дело и работа с деревом, "
            "электроника и Arduino, 3D-печать, умный дом"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "woodworking": {
        "aliases": ["столярное дело", "работа по дереву", "столярка", "мастерская", "дерево"],
        "enriched_text": (
            "столярное и плотницкое дело, работа с деревом и фанерой, "
            "изготовление мебели, фрезеровка и шлифовка"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "cnc_routing": {
        "aliases": ["чпу", "cnc", "фрезерный станок", "чпу фрезеровка", "числовое программное управление"],
        "enriched_text": (
            "работа на станках с ЧПУ, фрезеровка дерева и пластика по программе, "
            "CAD/CAM проектирование и изготовление деталей"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "electronics_diy": {
        "aliases": ["электроника", "arduino", "ардуино", "пайка", "микроконтроллер"],
        "enriched_text": (
            "любительская электроника и Arduino, создание устройств и гаджетов, "
            "Raspberry Pi проекты, ESP32 для IoT и автоматизации"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "raspberry_pi_projects": {
        "aliases": ["raspberry pi", "распберри пай", "малинка", "одноплатник"],
        "enriched_text": (
            "проекты на одноплатном компьютере Raspberry Pi, "
            "сервер, медиацентр, умный дом, GPIO и датчики"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "esp32_projects": {
        "aliases": ["esp32", "есп32", "iot", "интернет вещей", "wifi модуль"],
        "enriched_text": (
            "разработка IoT устройств на базе ESP32, WiFi и Bluetooth модуль, "
            "датчики температуры и влажности, MQTT протокол"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "3d_printing_hobby": {
        "aliases": ["3d печать", "3d принтер", "печатаю", "fdm", "sla", "abs", "pla"],
        "enriched_text": (
            "хобби 3D-печати, FDM и фотополимерная SLA печать, "
            "настройка принтера и слайсинг моделей, создание прототипов"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "fdm_printers": {
        "aliases": ["fdm", "печать пластиком", "pla", "petg", "abs"],
        "enriched_text": (
            "FDM 3D-печать расплавленной нитью, работа с пластиками PLA и PETG, "
            "калибровка стола и ретракты, постобработка моделей"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "resin_printers": {
        "aliases": ["sla", "фотополимер", "жидкая смола", "ультрафиолет"],
        "enriched_text": (
            "фотополимерная 3D-печать смолой SLA, высокая детализация, "
            "промывка и засветка моделей, безопасность при работе"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "smart_home": {
        "aliases": ["умный дом", "автоматизация дома", "smart home", "голосовой помощник"],
        "enriched_text": (
            "создание системы умного дома, Home Assistant, датчики движения и света, "
            "голосовое управление Алисой и Google Home"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "home_assistant": {
        "aliases": ["home assistant", "ха", "хоум ассистант", "автоматизация"],
        "enriched_text": (
            "платформа Home Assistant для управления умным домом, "
            "интеграции устройств, сценарии автоматизации и панели управления"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },
    "leathercraft": {
        "aliases": ["кожа", "кожевенное дело", "работа с кожей", "изделия из кожи"],
        "enriched_text": (
            "ремесло работы с натуральной кожей, раскрой и шитье, "
            "создание кошельков, сумок и чехлов ручной работы"
        ),
        "category": "life",
        "subcategory": "DIY и Сделай сам",
        "parent": "Дом и Образ жизни",
    },

    "pets_animals": {
        "aliases": ["животные", "питомцы", "собака", "кот", "аквариум", "террариум"],
        "enriched_text": (
            "домашние животные и уход за ними, собаки и кинология, "
            "кошки, аквариумные рыбки и террариумные животные"
        ),
        "category": "life",
        "subcategory": "Домашние животные",
        "parent": "Дом и Образ жизни",
    },
    "dog_training": {
        "aliases": ["собака", "кинология", "дрессировка", "пес", "щенок", "команды"],
        "enriched_text": (
            "воспитание и дрессировка собак, обучение базовым командам, "
            "социализация щенка, кинологический спорт и аджилити"
        ),
        "category": "life",
        "subcategory": "Домашние животные",
        "parent": "Дом и Образ жизни",
    },
    "dog_sports": {
        "aliases": ["кинологический спорт", "аджилити", "обидиенс", "флайбол"],
        "enriched_text": (
            "спортивные дисциплины с собаками, аджилити полоса препятствий, "
            "обидиенс послушание, флайбол и танцы с собаками"
        ),
        "category": "life",
        "subcategory": "Домашние животные",
        "parent": "Дом и Образ жизни",
    },
    "cat_lovers": {
        "aliases": ["кошка", "кот", "котейка", "мурлыка", "фелинология"],
        "enriched_text": (
            "любовь к кошкам, уход и воспитание, породы и здоровье, "
            "когтеточки и наполнители, психология кошачьих"
        ),
        "category": "life",
        "subcategory": "Домашние животные",
        "parent": "Дом и Образ жизни",
    },
    "aquarium_fish": {
        "aliases": ["аквариум", "рыбки", "аква", "травник", "креветки", "гуппи"],
        "enriched_text": (
            "аквариумистика, содержание пресноводных рыбок и растений, "
            "аквариум-травник с CO2, морской риф и кораллы"
        ),
        "category": "life",
        "subcategory": "Домашние животные",
        "parent": "Дом и Образ жизни",
    },
    "planted_aquarium": {
        "aliases": ["травник", "аквариум с растениями", "акваскейп", "ивагуми"],
        "enriched_text": (
            "создание аквариума-травника, голландский и природный акваскейп, "
            "система подачи CO2 и удобрения для водных растений"
        ),
        "category": "life",
        "subcategory": "Домашние животные",
        "parent": "Дом и Образ жизни",
    },
    "reef_aquarium": {
        "aliases": ["морской аквариум", "риф", "кораллы", "соль", "протеин скиммер"],
        "enriched_text": (
            "содержание морского рифового аквариума, кораллы и морские рыбы, "
            "контроль параметров воды, кальций и освещение"
        ),
        "category": "life",
        "subcategory": "Домашние животные",
        "parent": "Дом и Образ жизни",
    },
    "reptile_terrarium": {
        "aliases": ["террариум", "рептилии", "змеи", "ящерица", "геккон", "черепаха"],
        "enriched_text": (
            "террариумистика, содержание змей, ящериц и черепах, "
            "обогрев и ультрафиолет, оформление биотопов"
        ),
        "category": "life",
        "subcategory": "Домашние животные",
        "parent": "Дом и Образ жизни",
    },

    # ============================================================
    # 9. НАУКА, ОБРАЗОВАНИЕ, ФИНАНСЫ, КУЛЬТУРА
    # ============================================================
    "space_astronomy": {
        "aliases": ["космос", "астрономия", "телескоп", "звезды", "планеты", "галактика"],
        "enriched_text": (
            "космос и астрономия, астрофотография глубокого космоса, "
            "ракетостроение и SpaceX, изучение планет и звезд"
        ),
        "category": "life",
        "subcategory": "Космос и Астрономия",
        "parent": "Наука и Образование",
    },
    "astrophotography": {
        "aliases": ["астрофотография", "астрофото", "съемка неба", "млечный путь"],
        "enriched_text": (
            "фотографирование космических объектов, туманности и галактики, "
            "трекинг звезд, сложение кадров и обработка в PixInsight"
        ),
        "category": "life",
        "subcategory": "Космос и Астрономия",
        "parent": "Наука и Образование",
    },
    "deep_sky_objects": {
        "aliases": ["deep sky", "дипскай", "туманности", "галактики", "звездные скопления"],
        "enriched_text": (
            "наблюдение и съемка объектов глубокого космоса, "
            "туманность Ориона и галактика Андромеды, монтировка с автогидом"
        ),
        "category": "life",
        "subcategory": "Космос и Астрономия",
        "parent": "Наука и Образование",
    },
    "planetary_imaging": {
        "aliases": ["планетная съемка", "юпитер", "сатурн", "луна", "лаки-имаджинг"],
        "enriched_text": (
            "съемка планет Солнечной системы, lucky imaging метод, "
            "сложение тысяч кадров для четкой картинки Юпитера и Сатурна"
        ),
        "category": "life",
        "subcategory": "Космос и Астрономия",
        "parent": "Наука и Образование",
    },
    "rocket_science": {
        "aliases": ["ракетостроение", "ракеты", "spacex", "falcon", "starship", "нло"],
        "enriched_text": (
            "ракетная техника и космонавтика, SpaceX Falcon и Starship, "
            "NASA и Роскосмос, орбитальная механика и колонизация Марса"
        ),
        "category": "life",
        "subcategory": "Космос и Астрономия",
        "parent": "Наука и Образование",
    },
    "spacex_starship": {
        "aliases": ["spacex", "спейс икс", "starship", "илон маск", "марс"],
        "enriched_text": (
            "проекты SpaceX и корабль Starship, многоразовые ракеты, "
            "миссия на Марс, спутниковый интернет Starlink"
        ),
        "category": "life",
        "subcategory": "Космос и Астрономия",
        "parent": "Наука и Образование",
    },
    "physics_science": {
        "aliases": ["физика", "квантовая физика", "теория относительности", "эйнштейн"],
        "enriched_text": (
            "фундаментальная физика, квантовая механика и теория струн, "
            "общая теория относительности, большой адронный коллайдер"
        ),
        "category": "life",
        "subcategory": "Наука и Образование",
        "parent": "Наука и Образование",
    },
    "quantum_physics": {
        "aliases": ["квантовая физика", "квантмех", "запутанность", "шредингер", "кот"],
        "enriched_text": (
            "квантовая механика, суперпозиция и запутанность, "
            "квантовые вычисления и кубиты, интерпретации квантовой теории"
        ),
        "category": "life",
        "subcategory": "Наука и Образование",
        "parent": "Наука и Образование",
    },

    "philosophy_thinking": {
        "aliases": ["философия", "мышление", "смысл", "экзистенция", "стоицизм"],
        "enriched_text": (
            "философские размышления и школы, стоицизм и экзистенциализм, "
            "нигилизм и буддийская философия, поиск смысла"
        ),
        "category": "life",
        "subcategory": "Философия",
        "parent": "Наука и Образование",
    },
    "stoicism_phil": {
        "aliases": ["стоицизм", "стоик", "марк аврелий", "сенека", "дихотомия контроля"],
        "enriched_text": (
            "философия стоицизма, учение Марка Аврелия и Сенеки, "
            "принятие неконтролируемого, развитие добродетелей"
        ),
        "category": "life",
        "subcategory": "Философия",
        "parent": "Наука и Образование",
    },
    "existentialism": {
        "aliases": ["экзистенциализм", "сартр", "камю", "абсурд", "бытие"],
        "enriched_text": (
            "экзистенциальная философия, Сартр и Камю, "
            "свобода выбора и ответственность, абсурдность бытия и бунт"
        ),
        "category": "life",
        "subcategory": "Философия",
        "parent": "Наука и Образование",
    },
    "nihilism_absurdism": {
        "aliases": ["нигилизм", "абсурдизм", "ничто", "нет смысла", "ницше"],
        "enriched_text": (
            "философия нигилизма и абсурдизма, отрицание объективного смысла, "
            "идеи Ницше и критика морали"
        ),
        "category": "life",
        "subcategory": "Философия",
        "parent": "Наука и Образование",
    },
    "buddhist_philosophy": {
        "aliases": ["буддийская философия", "буддизм", "дхарма", "четыре благородные истины"],
        "enriched_text": (
            "философские основы буддизма, четыре благородные истины, "
            "восьмеричный путь и концепция пустоты"
        ),
        "category": "life",
        "subcategory": "Философия",
        "parent": "Наука и Образование",
    },

    "investing_finance": {
        "aliases": ["инвестиции", "финансы", "деньги", "акции", "крипта", "трейдинг"],
        "enriched_text": (
            "инвестирование и управление личными финансами, "
            "фондовый рынок и дивиденды, криптовалюты и недвижимость"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },
    "stock_market": {
        "aliases": ["фондовый рынок", "акции", "биржа", "облигации", "moex", "sp500"],
        "enriched_text": (
            "инвестиции в фондовый рынок, покупка акций и облигаций, "
            "технический и фундаментальный анализ, портфельные стратегии"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },
    "dividend_investing": {
        "aliases": ["дивиденды", "dividend investing", "пассивный доход", "дивидендные акции"],
        "enriched_text": (
            "дивидендная стратегия инвестирования, выбор компаний с выплатами, "
            "реинвестирование дивидендов и сложный процент"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },
    "technical_analysis": {
        "aliases": ["технический анализ", "теханализ", "уровни поддержки", "индикаторы", "рси"],
        "enriched_text": (
            "технический анализ рынка, уровни поддержки и сопротивления, "
            "японские свечи, RSI и MACD, графический анализ"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },
    "fundamental_analysis": {
        "aliases": ["фундаментальный анализ", "мультипликаторы", "отчетность", "p/e"],
        "enriched_text": (
            "фундаментальный анализ компаний, изучение финансовой отчетности, "
            "расчет справедливой стоимости, мультипликаторы P/E и P/B"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },
    "crypto_web3": {
        "aliases": ["криптовалюта", "крипта", "биткоин", "bitcoin", "web3", "defi"],
        "enriched_text": (
            "инвестиции в криптовалюты и Web3, Bitcoin и Ethereum, "
            "DeFi протоколы и смарт-контракты, NFT и токены"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },
    "defi_protocols": {
        "aliases": ["defi", "дефи", "децентрализованные финансы", "пулы ликвидности"],
        "enriched_text": (
            "сектор DeFi в криптовалютах, фарминг и стейкинг, "
            "децентрализованные биржи и кредитование под залог"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },
    "nft_art_collectibles": {
        "aliases": ["nft", "энэфти", "токен", "цифровой актив", "коллекционирование"],
        "enriched_text": (
            "рынок NFT и цифрового искусства, создание и торговля токенами, "
            "коллекционные предметы и метавселенные"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },
    "real_estate_invest": {
        "aliases": ["недвижимость", "квартира", "ипотека", "аренда", "коммерческая"],
        "enriched_text": (
            "инвестиции в недвижимость, жилая и коммерческая аренда, "
            "ипотечное кредитование и анализ окупаемости"
        ),
        "category": "work",
        "subcategory": "Инвестиции",
        "parent": "Финансы и Бизнес",
    },

    "startup_ecosystem": {
        "aliases": ["стартап", "предпринимательство", "бизнес", "фаундер", "венчур"],
        "enriched_text": (
            "стартап-экосистема и предпринимательство, венчурный капитал, "
            "запуск MVP и Lean Startup, питч-деки и привлечение инвестиций"
        ),
        "category": "work",
        "subcategory": "Стартап-экосистема",
        "parent": "Финансы и Бизнес",
    },
    "venture_capital": {
        "aliases": ["венчурный капитал", "venture capital", "vc", "инвесторы", "раунд"],
        "enriched_text": (
            "индустрия венчурного капитала, инвестиции в технологические стартапы, "
            "раунды seed и раунд A, оценка компаний и выход из инвестиций"
        ),
        "category": "work",
        "subcategory": "Стартап-экосистема",
        "parent": "Финансы и Бизнес",
    },
    "bootstrapping_biz": {
        "aliases": ["бутстрэппинг", "самофинансирование", "без инвестиций", "свои деньги"],
        "enriched_text": (
            "развитие бизнеса на собственные средства без внешних инвестиций, "
            "максимальная эффективность и контроль над компанией"
        ),
        "category": "work",
        "subcategory": "Стартап-экосистема",
        "parent": "Финансы и Бизнес",
    },
    "mvp_development": {
        "aliases": ["mvp", "минимальный продукт", "lean startup", "быстрый запуск"],
        "enriched_text": (
            "концепция минимально жизнеспособного продукта, "
            "быстрая проверка гипотез и итерации по обратной связи"
        ),
        "category": "work",
        "subcategory": "Стартап-экосистема",
        "parent": "Финансы и Бизнес",
    },
    "pitch_deck_creation": {
        "aliases": ["питч-дек", "pitch deck", "презентация стартапа", "инвест презентация"],
        "enriched_text": (
            "создание убедительной презентации для инвесторов, "
            "структура слайдов, проблема и решение, финансовая модель"
        ),
        "category": "work",
        "subcategory": "Стартап-экосистема",
        "parent": "Финансы и Бизнес",
    },

    "comics_geek": {
        "aliases": ["комиксы", "гик-культура", "марвел", "dc", "манга", "косплей"],
        "enriched_text": (
            "мир комиксов и гик-культуры, вселенные Marvel и DC, "
            "манга и вебтуны, графические романы и косплей"
        ),
        "category": "entertainment",
        "subcategory": "Комиксы и Гик-культура",
        "parent": "Литература и Чтение",
    },
    "marvel_universe": {
        "aliases": ["marvel", "марвел", "мстители", "человек-паук", "киновселенная"],
        "enriched_text": (
            "вселенная комиксов Marvel, супергерои и злодеи, "
            "MCU киновселенная, кроссовер-события и сольные серии"
        ),
        "category": "entertainment",
        "subcategory": "Комиксы и Гик-культура",
        "parent": "Литература и Чтение",
    },
    "dc_comics": {
        "aliases": ["dc", "детектив комикс", "бэтмен", "супермен", "джокер"],
        "enriched_text": (
            "вселенная DC Comics, Бэтмен и Супермен, Лига Справедливости, "
            "темный и gritty стиль, расширенная киновселенная"
        ),
        "category": "entertainment",
        "subcategory": "Комиксы и Гик-культура",
        "parent": "Литература и Чтение",
    },
    "manga_webtoon": {
        "aliases": ["манга", "вебтуны", "вебкомиксы", "манхва", "манхуа"],
        "enriched_text": (
            "японская манга и корейские вебтуны, чтение на смартфоне, "
            "разнообразие жанров и онлайн-платформы для публикации"
        ),
        "category": "entertainment",
        "subcategory": "Комиксы и Гик-культура",
        "parent": "Литература и Чтение",
    },
    "graphic_novels": {
        "aliases": ["графический роман", "графические новеллы", "взрослые комиксы"],
        "enriched_text": (
            "графические романы, серьезные и законченные истории в комикс-формате, "
            "Маус и Хранители, искусство и литература"
        ),
        "category": "entertainment",
        "subcategory": "Комиксы и Гик-культура",
        "parent": "Литература и Чтение",
    },

    "sci_fi_literature": {
        "aliases": ["научная фантастика", "sci-fi", "нф", "фантастика", "космоопера"],
        "enriched_text": (
            "литература научной фантастики, твердая НФ и киберпанк, "
            "космические саги и антиутопии будущего"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Литература и Чтение",
    },
    "hard_sci_fi": {
        "aliases": ["твердая нф", "hard sci-fi", "научно-обоснованная", "азимов"],
        "enriched_text": (
            "твердая научная фантастика с упором на научную достоверность, "
            "произведения Азимова и Кларка, технологии и физика"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Литература и Чтение",
    },
    "cyberpunk_lit": {
        "aliases": ["киберпанк литература", "гибсон", "неонуар", "высокие технологии"],
        "enriched_text": (
            "жанр киберпанк в литературе, Нейромант Уильяма Гибсона, "
            "высокие технологии и низкий уровень жизни, хакеры и корпорации"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Литература и Чтение",
    },
    "fantasy_literature": {
        "aliases": ["фэнтези", "фентези", "эльфы", "магия", "властелин колец"],
        "enriched_text": (
            "литература в жанре фэнтези, высокое и темное фэнтези, "
            "эпические саги и миры с магией и драконами"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Литература и Чтение",
    },
    "high_fantasy": {
        "aliases": ["высокое фэнтези", "эпическое фэнтези", "толкин", "колесо времени"],
        "enriched_text": (
            "эпическое высокое фэнтези, борьба добра и зла в вымышленных мирах, "
            "проработанный лор и языки, Властелин Колец"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Литература и Чтение",
    },
    "grimdark_fantasy": {
        "aliases": ["темное фэнтези", "grimdark", "моральная серость", "абекромби"],
        "enriched_text": (
            "мрачное и реалистичное темное фэнтези, отсутствие чистых героев, "
            "жестокий мир, серость морали и политические интриги"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Литература и Чтение",
    },
    "horror_literature": {
        "aliases": ["ужасы", "хоррор", "мистика", "king", "лавкрафт"],
        "enriched_text": (
            "литература ужасов и мистики, Стивен Кинг и Лавкрафт, "
            "сверхъестественный и психологический хоррор"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Литература и Чтение",
    },
    "lovecraftian_horror": {
        "aliases": ["лавкрафт", "ктулху", "мифы ктулху", "космический ужас"],
        "enriched_text": (
            "лавкрафтовские ужасы и мифы Ктулху, страх перед неизведанным, "
            "древние боги и безумие от непознаваемого"
        ),
        "category": "entertainment",
        "subcategory": "Литература и Чтение",
        "parent": "Литература и Чтение",
    },
}

# Пары для будущего fine-tuning SBERT (сленг ↔ каноническое описание).
# Используется ml/finetune_sbert.py как стартовый датасет.
FINETUNE_POSITIVE_PAIRS: list[tuple[str, str]] = [
    ("разработка на Flask", "написание бэкенда на Python, веб-фреймворк Flask"),
    ("Flask", "бэкенд-разработка на Python"),
    ("катнуть в кэсочку", "компьютерные игры Counter-Strike"),
    ("катнуть в кс", "играть в Counter-Strike, видеоигры"),
    ("бекенд", "серверная разработка, backend programming"),
    ("фронт", "фронтенд-разработка, клиентская часть приложения"),
    ("девопс", "DevOps, инфраструктура, CI/CD"),
    ("нейросети", "машинное обучение, искусственный интеллект"),
    ("figma", "дизайн интерфейсов UI/UX"),
    ("качалка", "спорт и фитнес, тренировки"),
    ("дота", "Dota 2, MOBA, киберспорт"),
    ("стартап", "предпринимательство, запуск IT-продукта"),
    ("фастапи", "веб-фреймворк FastAPI для Python"),
    ("питон", "программирование на языке Python"),
    ("работа с бд", "разработка и управление базами данных SQL"),
    ("кубер", "оркестрация контейнеров Kubernetes"),
    ("реакт", "фронтенд-фреймворк React и его экосистема"),
    ("геймдев", "разработка видеоигр на Unity или Unreal Engine"),
    ("рпг игры", "ролевые компьютерные игры, прокачка и сюжет"),
    ("днд", "настольная ролевая игра Dungeons & Dragons"),
    ("миджорни", "генерация изображений нейросетью Midjourney"),
    ("битмейкинг", "создание битов и музыкальное продюсирование"),
    ("гитара", "игра на музыкальном инструменте гитара"),
    ("кофе", "приготовление и культура потребления specialty кофе"),
    ("йога", "практика йоги для здоровья и гибкости"),
    ("астрология", "наблюдение за звездами и астрономия как хобби"),
    ("философия", "изучение философских концепций и размышления"),
    ("крипта", "инвестиции и торговля криптовалютами"),
]