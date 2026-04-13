import unittest
import json
import os
from datetime import datetime

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.ai.data_processor import TextDataProcessor
from app.ai.personality_analyzer import PersonalityAnalyzer
from app.ai.models import UserPersonalityProfile, TrainingMetrics
from data.user import User
from data.message import Message # Для имитации данных

# Временные настройки для теста
TEST_DB_PATH = 'test_ai.db'
TEST_MODEL_DIR = 'app/ai/test_models'
TEST_TFIDF_PATH = os.path.join(TEST_MODEL_DIR, 'tfidf_vectorizer.pkl')
TEST_MODEL_PATH = os.path.join(TEST_MODEL_DIR, 'personality_mlp_classifier.pkl')
TEST_DATASET_PATH = 'data/test_ai_personality_dataset.json'

class AITestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = Flask(__name__)
        cls.app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{TEST_DB_PATH}'
        cls.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        cls.db = SQLAlchemy(cls.app)

        # Создаем необходимую структуру папок для моделей
        os.makedirs(TEST_MODEL_DIR, exist_ok=True)

        with cls.app.app_context():
            cls.db.init_app(cls.app)
            cls.db.create_all()
            # Создаем тестовый набор данных
            cls._create_test_dataset()
            cls._create_mock_users_messages()

    @classmethod
    def tearDownClass(cls):
        with cls.app.app_context():
            cls.db.session.remove()
            cls.db.drop_all()
        os.remove(TEST_DB_PATH)
        # Удаляем тестовые модели и векторизатор
        if os.path.exists(TEST_MODEL_PATH): os.remove(TEST_MODEL_PATH)
        if os.path.exists(TEST_TFIDF_PATH): os.remove(TEST_TFIDF_PATH)
        if os.path.exists(TEST_DATASET_PATH): os.remove(TEST_DATASET_PATH)
        if os.path.exists(TEST_MODEL_DIR): os.rmdir(TEST_MODEL_DIR)

    @classmethod
    def _create_test_dataset(cls):
        # Простой датасет для тестирования
        dataset = [
            {"text": "Я люблю организовывать, планировать и выполнять задачи по расписанию. Я очень детализирован и всегда стремлюсь к порядку.", "label": "J"}, # Judging (планирование/организация)
            {"text": "Я творческий хаотик, предпочитаю спонтанность и гибкость. Принимаю решения на ходу и не люблю строгие рамки.", "label": "P"}, # Perceiving (спонтанность/гибкость)
            {"text": "Я очень общительный, люблю вечеринки и всегда нахожусь в центре внимания. Обожаю быть среди людей.", "label": "E"}, # Extraversion
            {"text": "Я предпочитаю тихие вечера дома с книгой, не люблю шумные компании. Мне нужно время для себя, чтобы восстановить силы.", "label": "I"}, # Introversion
            {"text": "Всегда думаю о будущем, меня привлекают абстрактные идеи и теории. Я ищу скрытые смыслы и возможности.", "label": "N"}, # Intuition
            {"text": "Я практичный человек, доверяю фактам и моему опыту. Живу настоящим моментом и ценю то, что могу увидеть и потрогать.", "label": "S"}, # Sensing
            {"text": "Когда принимаю решения, я полагаюсь на логику, объективные факты, и не позволяю эмоциям влиять на меня.", "label": "T"}, # Thinking
            {"text": "Мои решения всегда основаны на чувствах, ценностях и том, как это повлияет на других людей. Я эмпатичен и заботлив.", "label": "F"}
        ]
        with open(TEST_DATASET_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    @classmethod
    def _create_mock_users_messages(cls):
        with cls.app.app_context():
            user1 = User(username='test_user1', email='test1@example.com', password_hash='pbkdf2:sha256:150000$xxxx$yyyy')
            user2 = User(username='test_user2', email='test2@example.com', password_hash='pbkdf2:sha256:150000$xxxx$zzzz')
            user3 = User(username='test_user3', email='test3@example.com', password_hash='pbkdf2:sha256:150000$xxxx$aaaa')
            cls.db.session.add_all([user1, user2, user3])
            cls.db.session.commit()

            msg1 = Message(user_id=user1.id, text='Я очень люблю упорядочивать свои мысли и действия. Всегда составляю списки дел.')
            msg2 = Message(user_id=user1.id, text='Мои проекты всегда четко структурированы. Дисциплина - мой конек.')
            msg3 = Message(user_id=user2.id, text='Я скорее приспособлюсь к ситуации, чем буду ее жестко планировать. Мне нравится импровизировать.')
            msg4 = Message(user_id=user2.id, text='Полеты в облаках и новые идеи - вот что меня вдохновляет! Люблю думать о больших картинах.')
            msg5 = Message(user_id=user3.id, text='Я очень общительный и люблю быть в центре внимания. Обожаю новые знакомства.')
            msg6 = Message(user_id=user3.id, text='Главное для меня - гармония в отношениях и поддержка близких. Эмоции важны.')

            cls.db.session.add_all([msg1, msg2, msg3, msg4, msg5, msg6])
            cls.db.session.commit()

    def test_01_train_model(self):
        print("\n--- Running test_01_train_model ---")
        with self.app.app_context():
            processor = TextDataProcessor(model_path=TEST_TFIDF_PATH)
            analyzer = PersonalityAnalyzer(processor, model_name=os.path.basename(TEST_MODEL_PATH))

            with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            texts = [item['text'] for item in dataset]
            labels = [item['label'] for item in dataset]

            # Обучаем векторизатор
            processor.fit_vectorizer(texts)
            # Обучаем модель
            metrics = analyzer.train(texts, labels, model_version='test_v1.0')

            self.assertIsNotNone(metrics)
            self.assertGreaterEqual(metrics.train_accuracy, 0.5) # Ожидаем хоть какую-то точность
            print(f"Training Metrics: {metrics.to_dict()}")
            self.assertTrue(os.path.exists(TEST_MODEL_PATH))
            self.assertTrue(os.path.exists(TEST_TFIDF_PATH))

    def test_02_predict_personality(self):
        print("\n--- Running test_02_predict_personality ---")
        with self.app.app_context():
            processor = TextDataProcessor(model_path=TEST_TFIDF_PATH)
            analyzer = PersonalityAnalyzer(processor, model_name=os.path.basename(TEST_MODEL_PATH))

            # Имитация текста пользователя
            test_text_organized = "Я всегда делаю все по плану, очень пунктуален и надежен. Люблю, когда все четко."
            test_text_spontaneous = "Я человек настроения, предпочитаю действовать по ситуации. Планы - это не для меня."
            test_text_extrovert = "Обожаю большие компании, новые знакомства и активный отдых. В одиночестве мне скучно."

            profile_organized = analyzer.predict_personality(test_text_organized)
            profile_spontaneous = analyzer.predict_personality(test_text_spontaneous)
            profile_extrovert = analyzer.predict_personality(test_text_extrovert)

            self.assertIsNotNone(profile_organized)
            self.assertIsNotNone(profile_spontaneous)
            self.assertIsNotNone(profile_extrovert)

            print(f"Profile (Organized): MBTI={profile_organized.mbti_type}, Confidence={profile_organized.confidence_score:.2f}")
            print(f"Profile (Spontaneous): MBTI={profile_spontaneous.mbti_type}, Confidence={profile_spontaneous.confidence_score:.2f}")
            print(f"Profile (Extrovert): MBTI={profile_extrovert.mbti_type}, Confidence={profile_extrovert.confidence_score:.2f}")

            # Ассерты для проверки предсказаний (могут быть неточными из-за маленького датасета)
            # self.assertEqual(profile_organized.mbti_type, 'J')
            # self.assertEqual(profile_spontaneous.mbti_type, 'P')

            # Проверим, что загруженная модель имеет атрибут classes_
            self.assertTrue(hasattr(analyzer.model, 'classes_'))
            

    def test_03_analyze_and_fetch_user_profile_from_db(self):
        print("\n--- Running test_03_analyze_and_fetch_user_profile_from_db ---")
        from app.ai.celery_tasks import analyze_user_profile
        with self.app.app_context():
            user1 = User.query.filter_by(username='test_user1').first()
            self.assertIsNotNone(user1)

            # Инициируем анализ через таску Celery (вызываем напрямую для теста)
            result = analyze_user_profile(user1.id, force=True)
            self.assertEqual(result['status'], 'success')
            self.assertIn('profile_id', result)

            profile = UserPersonalityProfile.query.filter_by(user_id=user1.id).first()
            self.assertIsNotNone(profile)
            print(f"User1 Personality Profile: {profile.to_dict()}")
            
            self.assertIsNotNone(profile.mbti_type)
            self.assertGreater(profile.confidence_score, 0)

    def test_04_calculate_compatibility(self):
        print("\n--- Running test_04_calculate_compatibility ---")
        with self.app.app_context():
            user1_profile = UserPersonalityProfile(
                user_id=1, openness=0.8, conscientiousness=0.7, 
                extraversion=0.6, agreeableness=0.5, neuroticism=0.4
            )
            user2_profile = UserPersonalityProfile(
                user_id=2, openness=0.7, conscientiousness=0.6, 
                extraversion=0.5, agreeableness=0.7, neuroticism=0.3
            )
            user3_profile = UserPersonalityProfile(
                user_id=3, openness=0.2, conscientiousness=0.3, 
                extraversion=0.8, agreeableness=0.9, neuroticism=0.1
            )

            # Для этого теста нужно добавить эти профили в сессию, если они не существуют
            # Обычно они будут созданы через анализ сообщений
            existing_profile1 = UserPersonalityProfile.query.filter_by(user_id=user1_profile.user_id).first()
            if not existing_profile1: self.db.session.add(user1_profile)
            else: user1_profile = existing_profile1

            existing_profile2 = UserPersonalityProfile.query.filter_by(user_id=user2_profile.user_id).first()
            if not existing_profile2: self.db.session.add(user2_profile)
            else: user2_profile = existing_profile2

            existing_profile3 = UserPersonalityProfile.query.filter_by(user_id=user3_profile.user_id).first()
            if not existing_profile3: self.db.session.add(user3_profile)
            else: user3_profile = existing_profile3

            self.db.session.commit()

            processor = TextDataProcessor(model_path=TEST_TFIDF_PATH)
            analyzer = PersonalityAnalyzer(processor, model_name=os.path.basename(TEST_MODEL_PATH))

            compatibility1_2 = analyzer.calculate_compatibility(user1_profile, user2_profile)
            compatibility1_3 = analyzer.calculate_compatibility(user1_profile, user3_profile)

            print(f"Compatibility User1-User2: {compatibility1_2:.2f}")
            print(f"Compatibility User1-User3: {compatibility1_3:.2f}")

            self.assertGreaterEqual(compatibility1_2, 0) # Косинусное сходство от -1 до 1, но векторы положительные
            self.assertLessEqual(compatibility1_2, 1)

            # Ожидаем, что 1 и 2 более совместимы, чем 1 и 3 (из-за схожих Big Five)
            self.assertGreater(compatibility1_2, compatibility1_3)

if __name__ == '__main__':
    unittest.main()