import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import os
import re
import numpy as np


# 1. Гибридная архитектура (384 BERT + 4 Manual = 388)
class PersonalityClassifier(nn.Module):
    def __init__(self, input_size=388, num_traits=5):
        super(PersonalityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_traits)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Убираем BatchNorm для инференса по одному вектору, чтобы не "плыли" веса
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))


# 2. Основной класс-профайлер
class AIProfiler:
    def __init__(self, db=None, use_local_models=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)

        # Вход 388 (384 эмбеддинг + 4 фичи)
        self.model = PersonalityClassifier(input_size=388, num_traits=5).to(self.device)

        self.model_path = "ml/artifacts/personality_model_best.pth"
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def get_manual_features(self, text):
        """Извлекает цифровые маркеры с улучшенной нормализацией"""
        if not text: return [0.0, 0.0, 0.0, 0.0]

        # Длина: 500 символов - это мало для соцсети, сделаем 1000
        length = min(len(text) / 1000, 1.0)

        # Капс: считаем только для букв, чтобы знаки препинания не портили долю
        letters = [c for c in text if c.isalpha()]
        caps = sum(1 for c in letters if c.isupper()) / (len(letters) + 1) if letters else 0

        # Знаки: нормализуем чуть мягче
        excl = min(text.count('!') / 5, 1.0)
        ques = min(text.count('?') / 5, 1.0)

        # Важно: смещаем к среднему 0.5, чтобы "пустой" признак не тянул модель в нули
        return [length, caps, excl, ques]

    def _apply_keyword_boost(self, text, scores):
        """
        Психолингвистический фильтр: корректирует предсказания по ключевым словам.
        Индексы OCEAN: 0-O, 1-C, 2-E, 3-A, 4-N
        """
        low_text = text.lower()

        # Маркеры Невротизма (N) и низкой Доброжелательности (A)
        anger_words = ['бесит', 'ярость', 'ненавижу', 'ярости', 'несправедливо', 'раздражает']
        if any(word in low_text for word in anger_words):
            scores[4] = min(0.95, scores[4] + 0.25)  # Резко бустим Невротизм
            scores[3] = max(0.05, scores[3] - 0.20)  # Снижаем Дружелюбие

        # Маркеры Добросовестности (C)
        logic_words = ['планирую', 'список', 'заранее', 'график', 'дисциплина', 'цель']
        if any(word in low_text for word in logic_words):
            scores[1] = min(0.95, scores[1] + 0.20)

        # Маркеры Открытости (O)
        open_words = ['интересно', 'необычно', 'исследовать', 'космос', 'теория', 'философия']
        if any(word in low_text for word in open_words):
            scores[0] = min(0.95, scores[0] + 0.15)

        return scores

    def analyze_text(self, text):
        self.model.eval()

        # 1. Фичи берем из ОРИГИНАЛЬНОГО текста (капс, восклицания)
        m_feats = self.get_manual_features(text)

        # 2. Очищаем текст для BERT
        c_text = self.clean_text(text)
        # BERT лучше понимает смысл, когда регистр сохранен или приведен к нижнему системно
        emb = self.bert_model.encode([c_text.lower()])

        # 3. Склеиваем
        combined_input = np.concatenate([emb[0], m_feats])

        # Оптимизация создания тензора (убираем UserWarning)
        tensor_input = torch.from_numpy(combined_input).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(tensor_input).cpu().numpy()[0]

        # 4. Буст по ключевым словам (используем исходный текст)
        final_scores = self._apply_keyword_boost(text, prediction.tolist())

        return final_scores

    def clean_text(self, text):
        if not text: return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9?!.,\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def calculate_text_similarity(self, text1, text2):
        """Вычисляет семантическую близость между двумя названиями интересов/узлов"""
        if not text1 or not text2:
            return 0.0

        # Получаем эмбеддинги для обоих текстов
        emb1 = self.bert_model.encode([text1], convert_to_tensor=True)
        emb2 = self.bert_model.encode([text2], convert_to_tensor=True)

        # Считаем косинусное сходство
        cosine_scores = util.cos_sim(emb1, emb2)

        # Возвращаем число от 0 до 1
        return float(cosine_scores[0][0])

    def calculate_compatibility(self, vec1, vec2, weights=None):
        """
        Вычисляет процент совместимости (0-100%) между двумя векторами OCEAN.
        vec1, vec2: списки или тензоры из 5 значений.
        weights: список весов для каждой черты (например, [1, 1, 1, 1, 1]).
        """
        if not vec1 or not vec2:
            return 0

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        if weights is None:
            weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            weights = np.array(weights)

        # Считаем взвешенное евклидово расстояние
        # Чем меньше расстояние, тем выше совместимость
        distance = np.sqrt(np.sum(weights * (v1 - v2) ** 2))

        # Максимально возможное расстояние при весах 1.0 — это sqrt(5) ≈ 2.23
        max_dist = np.sqrt(np.sum(weights))

        compatibility = max(0, 100 * (1 - (distance / max_dist)))
        return round(compatibility, 2)
