import numpy as np
from sqlalchemy import text
from data.user import User  # Убедитесь, что импорты правильные
from data.ai import UserPersonalityProfile, UserCompatibility
from data.session import create_session, global_init
from config import config

# Инициализируем БД
global_init(config.DATABASE_URL)


def run_test():
    session = create_session()
    print("🚀 Запуск теста pgvector...")

    try:
        # 1. Очистка старых тестовых данных (чтобы тест был независимым)
        # ВНИМАНИЕ: Делайте это только на тестовой базе! Если это прод, уберите этот блок.
        print("Очистка таблиц...")
        session.query(UserCompatibility).delete()
        session.query(UserPersonalityProfile).delete()
        session.query(User).filter(User.name.in_(["test_user_1", "test_user_2"])).delete()
        session.commit()

        # 2. Создание двух тестовых пользователей
        user1 = User(name="test_user_1", email="test1@test.com")
        user2 = User(name="test_user_2", email="test2@test.com")
        session.add_all([user1, user2])
        session.commit()
        print(f"✅ Пользователи созданы: ID {user1.id} и ID {user2.id}")

        # 3. Сохранение профилей с разными векторами OCEAN
        # У юзера 1 высокие O, C, E и низкие A, N
        ocean1 = [0.9, 0.8, 0.7, 0.3, 0.2]
        profile1 = UserPersonalityProfile(
            user_id=user1.id,
            openness=ocean1[0], conscientiousness=ocean1[1],
            extraversion=ocean1[2], agreeableness=ocean1[3],
            neuroticism=ocean1[4],
            embedding=ocean1,  # Прямая запись вектора
            mbti_type="ENTJ"
        )

        # У юзера 2 векторы почти такие же, чтобы расстояние было минимальным
        ocean2 = [0.85, 0.75, 0.65, 0.35, 0.25]
        profile2 = UserPersonalityProfile(
            user_id=user2.id,
            openness=ocean2[0], conscientiousness=ocean2[1],
            extraversion=ocean2[2], agreeableness=ocean2[3],
            neuroticism=ocean2[4],
            embedding=ocean2,
            mbti_type="INTJ"
        )

        session.add_all([profile1, profile2])
        session.commit()
        print("✅ Профили и векторы сохранены в БД.")

        # 4. Проверка косинусного расстояния через SQL
        # Метод .cosine_distance() рассчитывает расстояние от 0 до 2
        print("\n🔍 Запуск векторного поиска...")
        target_profile = session.query(UserPersonalityProfile).filter_by(user_id=user1.id).first()

        similar_profiles = session.query(
            UserPersonalityProfile.user_id,
            UserPersonalityProfile.mbti_type,
            UserPersonalityProfile.embedding.cosine_distance(target_profile.embedding).label("distance")
        ).filter(
            UserPersonalityProfile.user_id != user1.id
        ).all()

        for match in similar_profiles:
            # Преобразуем расстояние [0, 2] в процент совместимости [0, 100]
            similarity_pct = max(0.0, 100.0 * (1.0 - (float(match.distance) / 2.0)))
            print(f"👉 Найдено совпадение: Юзер ID {match.user_id} (MBTI: {match.mbti_type})")
            print(f"   Косинусное расстояние: {match.distance:.4f}")
            print(f"   Совместимость: {similarity_pct:.2f}%")

            # Вручную через numpy для проверки
            v1, v2 = np.array(ocean1), np.array(ocean2)
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            np_distance = 1.0 - cos_sim
            print(f"   [Проверка] Расстояние через numpy: {np_distance:.4f}")

    except Exception as e:
        session.rollback()
        print(f"❌ Ошибка во время теста: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    run_test()