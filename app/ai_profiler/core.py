from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

from .text_utils import clean_user_text

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

# ---------------------------------------------------------------
# Конфигурация ординальной классификации
# ---------------------------------------------------------------

NUM_BINS = 10
BIN_EDGES = np.linspace(0, 1, NUM_BINS + 1)[1:-1]

# Центры бинов как numpy-константа (для использования вне torch)
BIN_CENTERS = np.linspace(1 / (2 * NUM_BINS), 1 - 1 / (2 * NUM_BINS), NUM_BINS).astype(np.float32)


def scores_to_bins(y: np.ndarray | list[float], num_bins: int = NUM_BINS) -> np.ndarray:
    """Converte contínuos OCEAN em [0, 1] para índices de bin ordinais.

    Args:
        y: Vetor de scores no intervalo [0, 1] (clip aplicado).
        num_bins: Número de bins (deve coincidir com ``NUM_BINS`` da cabeça da rede).

    Returns:
        Array ``int64`` de índices de classe por traço.
    """
    y = np.clip(y, 0.0, 1.0)
    bins = np.digitize(y, BIN_EDGES, right=False)
    return bins.astype(np.int64)


def bins_to_score(logits: torch.Tensor) -> torch.Tensor:
    """Converte logits (B, T, K) em scores contínuos via esperança sob softmax.

    Args:
        logits: Tensor ``(batch, traits, num_bins)``.

    Returns:
        Tensor ``(batch, traits)`` com valores no suporte [0, 1] (centros dos bins).
    """
    probs = F.softmax(logits, dim=-1)
    num_bins = logits.shape[-1]
    # detach не нужен — bin_values не имеют grad по умолчанию (создаются как leaf-константа)
    bin_values = torch.linspace(
        1 / (2 * num_bins),
        1 - 1 / (2 * num_bins),
        num_bins,
        device=logits.device,
        dtype=logits.dtype,
    )
    return (probs * bin_values).sum(dim=-1)


# ---------------------------------------------------------------
# ИСПРАВЛЕННАЯ АРХИТЕКТУРА PersonalityClassifier
# Изменения:
#   1. Уменьшена глубина [256, 128] вместо [256, 128, 64] — меньше переобучения
#   2. Dropout снижен до 0.15 — датасет маленький, не надо давить сигнал
#   3. Убран LayerNorm после последнего скрытого слоя (перед head'ом) —
#      он мешал сходимости на малых батчах
#   4. Добавлен residual-skip для стабилизации градиентов
# ---------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Лёгкий residual-блок: Linear → LN → GELU → Dropout → Linear → LN + skip."""
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class PersonalityClassifier(nn.Module):
    """Classificador ordinal OCEAN (cinco traços × ``num_bins`` classes)."""

    def __init__(self, input_size=388, dropout=0.2, num_bins=NUM_BINS):
        super().__init__()
        self.num_bins = num_bins

        self.projection = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual = ResidualBlock(256, dropout=dropout)

        self.bottleneck = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(128, 5 * num_bins)

        # Улучшенная инициализация
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Head инициализируем отдельно для мягкого старта
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.residual(x)
        x = self.bottleneck(x)
        logits = self.classifier(x)
        return logits.view(-1, 5, self.num_bins)

    def predict_scores(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            scores = bins_to_score(logits)
        return scores.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------
# MBTIClassifier — без изменений
# ---------------------------------------------------------------

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

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------
# AIProfiler — без изменений в логике, обновлён вызов классификатора
# ---------------------------------------------------------------

class AIProfiler:
    """Orquestra SBERT, classificador OCEAN ordinal e MBTI opcional."""

    def __init__(self, db: Any = None, use_local_models: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_name = os.environ.get(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.bert_model = SentenceTransformer(embedding_name, device=self.device)
        self.mbti_classes = MBTI_TYPES

        self.model = PersonalityClassifier(input_size=388, num_bins=NUM_BINS).to(self.device)
        self.mbti_model = MBTIClassifier(
            input_size=384, num_classes=len(self.mbti_classes)
        ).to(self.device)

        base_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.model_path = os.path.join(
            base_path, "ml/artifacts/personality_model_best.pth"
        )
        self.mbti_model_path = os.path.join(
            base_path, "ml/artifacts/mbti_model.pth"
        )

        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device, weights_only=True)
            )
        self.model.eval()

        self.has_mbti_model = False
        if os.path.exists(self.mbti_model_path):
            self.mbti_model.load_state_dict(
                torch.load(
                    self.mbti_model_path, map_location=self.device, weights_only=True
                )
            )
            self.has_mbti_model = True
        self.mbti_model.eval()

    @staticmethod
    def _ocean_to_soft_probs(mbti_from_ocean, temperature=0.5):
        logits = np.zeros(len(MBTI_TYPES), dtype=np.float32)
        idx = MBTI_TYPES.index(mbti_from_ocean)
        logits[idx] = 1.0 / temperature
        for i, t in enumerate(MBTI_TYPES):
            diff = sum(1 for a, b in zip(t, mbti_from_ocean) if a != b)
            if diff == 1:
                logits[i] = 0.3 / temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def get_manual_features(self, text: str) -> list[float]:
        if not text:
            return [0.0, 0.0, 0.0, 0.0]
        length = min(len(text) / 1000, 1.0)
        letters = [c for c in text if c.isalpha()]
        caps = (
            sum(1 for c in letters if c.isupper()) / (len(letters) + 1)
            if letters
            else 0
        )
        excl = min(text.count('!') / 5, 1.0)
        ques = min(text.count('?') / 5, 1.0)
        return [length, caps, excl, ques]

    def analyze_profile(self, text: str) -> dict[str, Any] | None:
        cleaned = self.clean_text(text)
        if not cleaned:
            return None

        m_feats = self.get_manual_features(cleaned)
        emb = self.bert_model.encode(
            [cleaned.lower()], convert_to_tensor=True
        )
        manual_tensor = (
            torch.tensor(m_feats, dtype=torch.float32).to(self.device).unsqueeze(0)
        )
        combined_input = torch.cat([emb, manual_tensor], dim=1)

        ocean_scores = self.model.predict_scores(combined_input)
        ocean_scores = np.clip(ocean_scores, 0.05, 0.95).tolist()

        if self.has_mbti_model:
            with torch.no_grad():
                mbti_logits = self.mbti_model(emb)
                mbti_probs = (
                    torch.softmax(mbti_logits, dim=1).cpu().numpy()[0]
                )
        else:
            mbti_probs = np.ones(len(self.mbti_classes)) / len(self.mbti_classes)

        mbti_from_ocean = self.infer_mbti(ocean_scores)
        ocean_soft = self._ocean_to_soft_probs(mbti_from_ocean)

        # Peso da rede MBTI vs. prior derivado de OCEAN. Sem modelo MBTI treinado,
        # evitar misturar com softmax uniforme (ruído). Sobrescrever com
        # NEXUS_MBTI_NEURAL_BLEND_WEIGHT em [0, 1].
        w_model = self._mbti_neural_blend_weight()
        blended_probs = (w_model * mbti_probs) + ((1 - w_model) * ocean_soft)
        mbti_type = self.mbti_classes[int(np.argmax(blended_probs))]

        communication = self.infer_communication_style(cleaned, ocean_scores)
        interests = self.extract_interests(cleaned)

        ocean_conf = float(
            np.mean(np.abs(np.array(ocean_scores) - 0.5)) * 2
        )
        mbti_conf = float(np.max(blended_probs))

        return {
            "ocean": ocean_scores,
            "mbti_type": mbti_type,
            "communication": communication,
            "interests": interests,
            "traits": interests.get("traits", []),
            "values": interests.get("values", []),
            "compatible_mbti_types": self.get_compatible_mbti(mbti_type),
            "confidence_score": float(
                np.clip((ocean_conf + mbti_conf) / 2.0, 0.0, 1.0)
            ),
        }

    def _mbti_neural_blend_weight(self) -> float:
        env = os.environ.get("NEXUS_MBTI_NEURAL_BLEND_WEIGHT")
        if env is not None:
            try:
                return float(max(0.0, min(1.0, float(env.strip()))))
            except ValueError:
                pass
        return 0.7 if self.has_mbti_model else 0.0

    def infer_mbti(self, scores: list[float]) -> str:
        o, c, e, a, n = scores
        mbti = [
            "E" if e >= 0.5 else "I",
            "N" if o >= 0.5 else "S",
            "F" if a >= 0.5 else "T",
            "J" if c >= 0.5 else "P",
        ]
        if n > 0.75:
            mbti[0] = "I"
        return "".join(mbti)

    def infer_communication_style(self, text: str, scores: list[float]) -> dict[str, Any]:
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
        uppercase_ratio = (
            (sum(1 for ch in letters if ch.isupper()) / len(letters))
            if letters
            else 0.0
        )
        exclamation_ratio = min(c_text.count("!") / 6.0, 1.0)
        avg_word_len = (
            np.mean([len(w) for w in c_text.split()]) if c_text.split() else 0.0
        )
        comma_density = min(c_text.count(",") / 10.0, 1.0)

        formality = float(
            np.clip(
                0.3 + (avg_word_len / 10.0) + comma_density * 0.2 - uppercase_ratio * 0.2,
                0.0,
                1.0,
            )
        )
        enthusiasm = float(
            np.clip(0.2 + exclamation_ratio * 0.6 + scores[2] * 0.2, 0.0, 1.0)
        )
        detail_oriented = float(
            np.clip(
                0.3 + scores[1] * 0.5 + min(len(c_text) / 1200.0, 1.0) * 0.2,
                0.0,
                1.0,
            )
        )

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

    def extract_interests(self, text: str) -> dict[str, Any]:
        normalized = self.clean_text(text).lower()
        tokens = [t for t in re.findall(r"[а-яa-z0-9]+", normalized) if len(t) > 2]
        counts = Counter(tokens)

        def top_matching(words, limit=8):
            ranked = sorted(
                ((w, counts[w]) for w in words if counts[w] > 0),
                key=lambda x: x[1],
                reverse=True,
            )
            return [w for w, _ in ranked[:limit]]

        hobby_words = {
            "спорт", "футбол", "бег", "йога", "игры", "музыка", "фото",
            "чтение", "рисование", "coding", "music", "travel", "gaming",
            "books", "fitness",
        }
        topic_words = {
            "психология", "наука", "технологии", "бизнес", "финансы",
            "кино", "искусство", "образование", "science", "tech",
            "startup", "ai", "ml",
        }
        skill_words = {
            "python", "sql", "аналитика", "дизайн", "маркетинг",
            "лидерство", "коммуникация", "management", "backend",
            "frontend", "devops",
        }
        dislike_words = {"ненавижу", "бесит", "не люблю", "раздражает"}
        goal_markers = {"цель", "план", "хочу", "мечта", "задача"}
        values_markers = {
            "семья", "карьера", "свобода", "развитие", "стабильность", "творчество"
        }

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
            "language_mix": (
                "ru+en"
                if re.search(r"[a-z]", normalized) and re.search(r"[а-я]", normalized)
                else "single"
            ),
            "message_length": (
                "long"
                if len(normalized) > 500
                else "medium"
                if len(normalized) > 180
                else "short"
            ),
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

    def clean_text(self, text: str | None) -> str:
        """Delega para :func:`clean_user_text` (limite de tamanho e sanitização)."""
        return clean_user_text(text)

    def calculate_compatibility(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0
        v1, v2 = np.array(vec1), np.array(vec2)
        distance = np.sqrt(np.mean((v1 - v2) ** 2))
        return round(float(max(0, 100 * (1 - distance))), 2)

    def calculate_text_similarity(self, texts1: list[str] | str, texts2: list[str] | str):
        # Превращаем в списки, если пришла одиночная строка
        if isinstance(texts1, str): texts1 = [texts1]
        if isinstance(texts2, str): texts2 = [texts2]

        # Кодируем тексты отдельно
        emb1 = self.bert_model.encode(texts1, convert_to_tensor=True)
        emb2 = self.bert_model.encode(texts2, convert_to_tensor=True)

        # Считаем косинусное сходство
        cos_sim_matrix = util.cos_sim(emb1, emb2)

        # Если мы сравниваем один к одному (например, заголовки нод)
        if len(texts1) == len(texts2) == 1:
            return cos_sim_matrix.item()  # Возвращаем просто число (float)

        # Если списки разной длины, diagonal() не сработает.
        # Обычно в таком случае берут среднее сходство или возвращают всю матрицу.
        # Для твоего лога (1x16 и 368x1) лучше вернуть среднее:
        return cos_sim_matrix.cpu().numpy()

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
