import logging
from unittest.mock import MagicMock
from app.ai.match_report import generate_match_report, build_default_match_report
from data.ai import UserPersonalityProfile, AIExtractedInterests
import logging
from unittest.mock import MagicMock

# СНАЧАЛА ИМПОРТИРУЕМ ВСЕ МОДЕЛИ, ЧТОБЫ SQLALCHEMY ИХ ЗАРЕГИСТРИРОВАЛ
# (Замени 'data.__all_models' или добавь импорт core-моделей, если у тебя другой путь)
try:
    import data.__all_models
except ImportError:
    # Если __all_models не подтягивает всё автоматически в тесте,
    # импортируй напрямую модели User и Interest:
    from data.user import User  # Или там, где лежит User и Interest
    # Сюда же можно импортировать Interest, если он в другом файле

# Теперь твои импорты будут работать штатно
from app.ai.match_report import generate_match_report, build_default_match_report
from data.ai import UserPersonalityProfile, AIExtractedInterests
# Настройка логирования для наглядности
logging.basicConfig(level=logging.DEBUG)


class DummyConfig:
    OLLAMA_ENABLED = True
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "phi3:medium"
    OLLAMA_TIMEOUT = 10
    CONTEXTUAL_ADAPTER_ENABLED = True


# Подменяем глобальный конфиг для теста, если необходимо
import app.ai.match_report as mock_report

mock_report.config = DummyConfig()


def test_pipeline():
    print("=== Запуск теста генерации мини-докладов ===\n")

    # 1. Создаем моковые данные пользователей (Программист-геймер и Дизайнер)
    profile_a = UserPersonalityProfile(
        user_id=1,
        mbti_type="INTJ",
        openness=0.8,  # Должна определиться высокая открытость
        conscientiousness=0.7,
        extraversion=0.2,
        agreeableness=0.4
    )

    profile_b = UserPersonalityProfile(
        user_id=2,
        mbti_type="ENFP",
        openness=0.85,
        conscientiousness=0.4,
        extraversion=0.75,  # Высокая экстраверсия
        agreeableness=0.65
    )

    interests_a = AIExtractedInterests(
        user_id=1,
        hobbies=[{"subcategory": "Видеоигры", "score": 0.9}],
        skills=[{"subcategory": "Разработка", "score": 0.95}],
        topics=[{"subcategory": "Python", "score": 0.85}],
        occupation="Backend Developer"
    )

    interests_b = AIExtractedInterests(
        user_id=2,
        hobbies=[{"subcategory": "Видеоигры", "score": 0.85}],
        skills=[{"subcategory": "Дизайн", "score": 0.9}],
        topics=[{"subcategory": "UI/UX", "score": 0.8}],
        occupation="UI/UX Designer"
    )

    # 2. Изолированный тест шаблонного фолбека (помогает проверить сборку направлений)
    print("--- Проверка шаблонного вывода (Fallback) ---")
    interests_a_dict = {"hobbies": interests_a.hobbies, "skills": interests_a.skills, "topics": interests_a.topics,
                        "occupation": interests_a.occupation}
    interests_b_dict = {"hobbies": interests_b.hobbies, "skills": interests_b.skills, "topics": interests_b.topics,
                        "occupation": interests_b.occupation}

    fallback_report = build_default_match_report(
        profile_a, profile_b, interests_a_dict, interests_b_dict, matched_tags=["Видеоигры"]
    )
    print(f"Результат: {fallback_report}\n")

    # 3. Мокаем сессию базы данных SQLAlchemy для интеграционного теста функции
    mock_db = MagicMock()

    # Настраиваем query().filter_by().first() для выдачи наших моков
    mock_query = mock_db.query

    def side_effect(model):
        mock_filter = MagicMock()
        if model == UserPersonalityProfile:
            mock_filter.filter_by.side_effect = lambda user_id: MagicMock(
                first=lambda: profile_a if user_id == 1 else profile_b)
        elif model == AIExtractedInterests:
            mock_filter.filter_by.side_effect = lambda user_id: MagicMock(
                first=lambda: interests_a if user_id == 1 else interests_b)
        return mock_filter


    mock_query.side_effect = side_effect

    # 4. Проверка работы основного метода без использования LLM (use_llm=False)
    print("--- Проверка работы функции generate_match_report (Force Fallback) ---")
    report_no_llm = generate_match_report(1, 2, mock_db, matched_tags=["Видеоигры"], use_llm=False)
    print(f"Результат: {report_no_llm}\n")

    # 5. Интеграционная проверка с Ollama (если запущена локально)
    print("--- Проверка работы функции generate_match_report с Ollama ---")
    print("(Убедитесь, что Ollama запущена и модель загружена, иначе сработает тихий фолбек)")

    report_llm = generate_match_report(1, 2, mock_db, matched_tags=["Видеоигры"], use_llm=True)
    print(f"Итоговый результат: {report_llm}\n")

    # Валидация ограничений (не более 2 предложений)
    sentences = [s for s in report_llm.split('.') if s.strip()]
    print(f"Количество предложений: {len(sentences)} (Ожидается <= 2)")

if __name__ == "__main__":
    test_pipeline()