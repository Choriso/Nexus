import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
import os
import re
import numpy as np
from collections import Counter

# Константы для всего проекта (синхронизировано с train_mbti.py)
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]


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
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))


class MBTIClassifier(nn.Module):
    def __init__(self, input_size=384, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x): return self.net(x)


class AIProfiler:
    def __init__(self, db=None, use_local_models=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        self.mbti_classes = MBTI_TYPES

        self.model = PersonalityClassifier(input_size=388, num_traits=5).to(self.device)
        self.mbti_model = MBTIClassifier(input_size=384, num_classes=len(self.mbti_classes)).to(self.device)

        # Динамические пути к артефактам
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = os.path.join(base_path, "ml/artifacts/personality_model_best.pth")
        self.mbti_model_path = os.path.join(base_path, "ml/artifacts/mbti_model.pth")

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.has_mbti_model = False
        if os.path.exists(self.mbti_model_path):
            self.mbti_model.load_state_dict(
                torch.load(self.mbti_model_path, map_location=self.device, weights_only=True))
            self.has_mbti_model = True
        self.mbti_model.eval()

    def _ocean_to_soft_probs(self, mbti_from_ocean, temperature=0.5):
        """Конвертирует OCEAN→MBTI в мягкое распределение вместо one-hot."""
        logits = np.zeros(len(self.mbti_classes), dtype=np.float32)
        idx = self.mbti_classes.index(mbti_from_ocean)
        logits[idx] = 1.0 / temperature

        for i, t in enumerate(self.mbti_classes):
            diff = sum(1 for a, b in zip(t, mbti_from_ocean) if a != b)
            if diff == 1:  # Соседние типы (разница в 1 букву)
                logits[i] = 0.3 / temperature

        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def get_manual_features(self, text):
        if not text: return [0.0, 0.0, 0.0, 0.0]
        length = min(len(text) / 1000, 1.0)
        letters = [c for c in text if c.isalpha()]
        caps = sum(1 for c in letters if c.isupper()) / (len(letters) + 1) if letters else 0
        excl = min(text.count('!') / 5, 1.0)
        ques = min(text.count('?') / 5, 1.0)
        return [length, caps, excl, ques]

    def analyze_profile(self, text):
        """Полный анализ профиля с использованием ансамбля моделей."""
        clean_text = self.clean_text(text)
        if not clean_text: return None

        # 1. OCEAN анализ
        m_feats = self.get_manual_features(text)
        emb = self.bert_model.encode([clean_text.lower()], convert_to_tensor=True)
        manual_tensor = torch.tensor(m_feats, dtype=torch.float32).to(self.device).unsqueeze(0)
        combined_input = torch.cat([emb, manual_tensor], dim=1)

        with torch.no_grad():
            ocean_prediction = self.model(combined_input).cpu().numpy()[0]
        ocean_scores = np.clip(ocean_prediction, 0.05, 0.95).tolist()

        # 2. MBTI Ансамбль
        # А) Предикт от модели
        if self.has_mbti_model:
            with torch.no_grad():
                mbti_logits = self.mbti_model(emb)
                mbti_probs = torch.softmax(mbti_logits, dim=1).cpu().numpy()[0]
        else:
            # Если модели нет, используем равномерное распределение
            mbti_probs = np.ones(len(self.mbti_classes)) / len(self.mbti_classes)

        # Б) Предикт от OCEAN (Prior)
        mbti_from_ocean = self.infer_mbti(ocean_scores)
        ocean_soft = self._ocean_to_soft_probs(mbti_from_ocean)

        # В) Смешивание (0.7 модель / 0.3 OCEAN)
        w_model = 0.7 if self.has_mbti_model else 0.3
        blended_probs = (w_model * mbti_probs) + ((1 - w_model) * ocean_soft)

        mbti_type = self.mbti_classes[int(np.argmax(blended_probs))]

        # 3. Дополнительные метрики
        communication = self.infer_communication_style(text, ocean_scores)
        interests = self.extract_interests(text)

        # Уверенность как среднее между разбросом OCEAN и уверенностью MBTI-модели
        ocean_conf = float(np.mean(np.abs(np.array(ocean_scores) - 0.5)) * 2)
        mbti_conf = float(np.max(blended_probs))

        return {
            "ocean": ocean_scores,
            "mbti_type": mbti_type,
            "communication": communication,
            "interests": interests,
            "traits": interests.get("traits", []),
            "values": interests.get("values", []),
            "compatible_mbti_types": self.get_compatible_mbti(mbti_type),
            "confidence_score": float(np.clip((ocean_conf + mbti_conf) / 2.0, 0.0, 1.0)),
        }

    def infer_mbti(self, scores):
        o, c, e, a, n = scores
        mbti = [
            "E" if e >= 0.5 else "I",
            "N" if o >= 0.5 else "S",
            "F" if a >= 0.5 else "T",
            "J" if c >= 0.5 else "P",
        ]
        if n > 0.75: mbti[0] = "I"  # Высокий невротизм склоняет к интроверсии
        return "".join(mbti)

    def infer_communication_style(self, text, scores):
        c_text = self.clean_text(text)
        if not c_text:
            return {
                "communication_style": "neutral",
                "formality": 0.5,
                "enthusiasm": 0.5,
                "detail_oriented": 0.5,
                "collaboration_style": "balanced",
            }

        letters = [ch for ch in c_text if ch.isalpha()]
        uppercase_ratio = (sum(1 for ch in letters if ch.isupper()) / len(letters)) if letters else 0.0
        exclamation_ratio = min(c_text.count("!") / 6.0, 1.0)
        avg_word_len = np.mean([len(w) for w in c_text.split()]) if c_text.split() else 0.0
        comma_density = min(c_text.count(",") / 10.0, 1.0)

        formality = float(np.clip(0.3 + (avg_word_len / 10.0) + comma_density * 0.2 - uppercase_ratio * 0.2, 0.0, 1.0))
        enthusiasm = float(np.clip(0.2 + exclamation_ratio * 0.6 + scores[2] * 0.2, 0.0, 1.0))
        detail_oriented = float(np.clip(0.3 + scores[1] * 0.5 + min(len(c_text) / 1200.0, 1.0) * 0.2, 0.0, 1.0))

        if formality > 0.7 and detail_oriented > 0.6:
            style = "professional"
        elif enthusiasm > 0.7:
            style = "friendly"
        elif formality < 0.35:
            style = "casual"
        else:
            style = "balanced"

        if scores[3] > 0.65:
            collaboration_style = "cooperative"
        elif scores[1] > 0.7:
            collaboration_style = "structured"
        elif scores[2] > 0.7:
            collaboration_style = "initiative"
        else:
            collaboration_style = "balanced"

        return {
            "communication_style": style,
            "formality": formality,
            "enthusiasm": enthusiasm,
            "detail_oriented": detail_oriented,
            "collaboration_style": collaboration_style,
        }

    def extract_interests(self, text):
        """Rule-based extraction of interests/topics/skills/preferences."""
        normalized = self.clean_text(text).lower()
        tokens = [t for t in re.findall(r"[а-яa-z0-9]+", normalized) if len(t) > 2]
        counts = Counter(tokens)

        def top_matching(words, limit=8):
            ranked = sorted(((w, counts[w]) for w in words if counts[w] > 0), key=lambda x: x[1], reverse=True)
            return [w for w, _ in ranked[:limit]]

        hobby_words = {
            "спорт", "футбол", "бег", "йога", "игры", "музыка", "фото", "чтение", "рисование",
            "coding", "music", "travel", "gaming", "books", "fitness",
        }
        topic_words = {
            "психология", "наука", "технологии", "бизнес", "финансы", "кино", "искусство", "образование",
            "science", "tech", "startup", "ai", "ml",
        }
        skill_words = {
            "python", "sql", "аналитика", "дизайн", "маркетинг", "лидерство", "коммуникация",
            "management", "backend", "frontend", "devops",
        }
        dislike_words = {"ненавижу", "бесит", "не люблю", "раздражает"}
        goal_markers = {"цель", "план", "хочу", "мечта", "задача"}
        values_markers = {"семья", "карьера", "свобода", "развитие", "стабильность", "творчество"}

        hobbies = top_matching(hobby_words)
        topics = top_matching(topic_words)
        skills = top_matching(skill_words)
        dislikes = [w for w in dislike_words if w in normalized]
        goals = [w for w in goal_markers if w in normalized]
        values = [w for w in values_markers if w in normalized]

        if "работа" in normalized or "проект" in normalized:
            occupation = "knowledge_worker"
            work_style = "structured" if "план" in normalized else "adaptive"
        else:
            occupation = None
            work_style = None

        preferences = {
            "language_mix": "ru+en" if re.search(r"[a-z]", normalized) and re.search(r"[а-я]", normalized) else "single",
            "message_length": "long" if len(normalized) > 500 else "medium" if len(normalized) > 180 else "short",
        }

        return {
            "hobbies": hobbies,
            "topics": topics,
            "skills": skills,
            "dislikes": dislikes,
            "occupation": occupation,
            "work_style": work_style,
            "short_term_goals": goals,
            "long_term_goals": goals[:2],
            "preferences": preferences,
            "traits": list(dict.fromkeys(skills[:2] + topics[:2])),
            "values": values,
        }

    def clean_text(self, text):
        if not text: return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9?!.,\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def calculate_compatibility(self, vec1, vec2):
        if not vec1 or not vec2: return 0
        v1, v2 = np.array(vec1), np.array(vec2)
        distance = np.sqrt(np.mean((v1 - v2) ** 2))
        return round(float(max(0, 100 * (1 - distance))), 2)

    def calculate_text_similarity(self, texts1, texts2):
        all_texts = texts1 + texts2
        embeddings = self.bert_model.encode(all_texts, batch_size=32, convert_to_tensor=True)
        emb1 = embeddings[:len(texts1)]
        emb2 = embeddings[len(texts1):]
        return util.cos_sim(emb1, emb2).diagonal().cpu().numpy()

    @staticmethod
    def get_compatible_mbti(mbti_type):
        compatibility_map = {
            "INTJ": ["ENFP", "ENTP"],
            "INTP": ["ENTJ", "ENFJ"],
            "ENTJ": ["INTP", "INFP"],
            "ENTP": ["INFJ", "INTJ"],
            "INFJ": ["ENFP", "ENTP"],
            "INFP": ["ENFJ", "ENTJ"],
            "ENFJ": ["INFP", "ISFP"],
            "ENFP": ["INFJ", "INTJ"],
            "ISTJ": ["ESFP", "ESTP"],
            "ISFJ": ["ESFP", "ESTP"],
            "ESTJ": ["ISFP", "ISTP"],
            "ESFJ": ["ISFP", "ISTP"],
            "ISTP": ["ESFJ", "ESTJ"],
            "ISFP": ["ENFJ", "ESFJ"],
            "ESTP": ["ISFJ", "ISTJ"],
            "ESFP": ["ISFJ", "ISTJ"],
        }
        return compatibility_map.get(mbti_type, [])
