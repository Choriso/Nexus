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

from config import config

from .interest_extractor import ZeroShotInterestExtractor, build_labels_from_taxonomy, load_neural_extractor
from .taxonomy import INTEREST_TAXONOMY
from .text_utils import clean_user_text
from .contextual_adapter import ContextualAdapter, get_contextual_adapter

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

NUM_BINS = 10
BIN_EDGES = np.linspace(0, 1, NUM_BINS + 1)[1:-1]
BIN_CENTERS = np.linspace(1 / (2 * NUM_BINS), 1 - 1 / (2 * NUM_BINS), NUM_BINS).astype(np.float32)


def scores_to_bins(y: np.ndarray | list[float], num_bins: int = NUM_BINS) -> np.ndarray:
    """
    Преобразует значения OCEAN (в диапазоне [0, 1]) в индексы ординальных бинов.

    Args:
        y (np.ndarray | list[float]): Массив или список значений в диапазоне [0, 1].
        num_bins (int): Количество бинов. По умолчанию равно NUM_BINS.

    Returns:
        np.ndarray: Индексы классов для каждого признака (int64).
    """
    y = np.clip(y, 0.0, 1.0)
    bins = np.digitize(y, BIN_EDGES, right=False)
    return bins.astype(np.int64)


def bins_to_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Преобразует логиты формы (batch, traits, num_bins) в непрерывные значения OCEAN через взвешенное ожидание по softmax.

    Args:
        logits (torch.Tensor): Тензор логитов (batch, traits, num_bins).

    Returns:
        torch.Tensor: Тензор с оценками (batch, traits) в диапазоне [0, 1].
    """
    probs = F.softmax(logits, dim=-1)
    num_bins = logits.shape[-1]
    bin_values = torch.linspace(
        1 / (2 * num_bins),
        1 - 1 / (2 * num_bins),
        num_bins,
        device=logits.device,
        dtype=logits.dtype,
    )
    return (probs * bin_values).sum(dim=-1)


class ResidualBlock(nn.Module):
    """
    Лёгкий residual-блок (Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm) с пропуском.

    Args:
        dim (int): Размерность слоя.
        dropout (float): Доля Dropout.
    """

    def __init__(self, dim: int, dropout: float = 0.15):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через residual-блок.

        Args:
            x (torch.Tensor): Входной тензор (batch, dim).

        Returns:
            torch.Tensor: Выходной тензор той же размерности.
        """
        return self.act(x + self.block(x))


class PersonalityClassifier(nn.Module):
    """
    Ординальный классификатор OCEAN (5 признаков × num_bins классов).

    Args:
        input_size (int): Размерность входного вектора.
        dropout (float): Доля Dropout.
        num_bins (int): Количество классов-бинов для одного признака.
    """

    def __init__(self, input_size: int = 388, dropout: float = 0.2, num_bins: int = NUM_BINS):
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через классификатор.

        Args:
            x (torch.Tensor): Входной тензор (batch, input_size).

        Returns:
            torch.Tensor: Логиты (batch, 5, num_bins).
        """
        x = self.projection(x)
        x = self.residual(x)
        x = self.bottleneck(x)
        logits = self.classifier(x)
        return logits.view(-1, 5, self.num_bins)

    def predict_scores(self, x: torch.Tensor) -> np.ndarray:
        """
        Прогнозирует OCEAN-признаки по входу.

        Args:
            x (torch.Tensor): Входной тензор (1, input_size).

        Returns:
            np.ndarray: Массив OCEAN-скоров (shape = [5]).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            scores = bins_to_score(logits)
        return scores.squeeze(0).cpu().numpy()


class MBTIClassifier(nn.Module):
    """
    Классификатор MBTI на эмбеддингах.

    Args:
        input_size (int): Размер входного вектора.
        num_classes (int): Количество MBTI-классов.
    """

    def __init__(self, input_size: int = 384, num_classes: int = 16):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.

        Args:
            x (torch.Tensor): Входной тензор.

        Returns:
            torch.Tensor: Логиты (batch, num_classes).
        """
        return self.net(x)


class AIProfiler:
    """
    Оркестрация моделей: SBERT, Ordinal OCEAN, MBTI-классификатор.

    Args:
        db (Any, optional): Внешний объект или БД.
        use_local_models (bool, optional): Использовать локальные веса.
        config_obj (Any, optional): Объект конфигурации (Config).
    """

    def __init__(
            self,
            db: Any = None,
            use_local_models: bool = True,
            config_obj: Any = None,
            adapter_enabled: bool = True
    ) -> None:
        """
        Инициализация — подгрузка моделей и ресурсов.

        Args:
            db (Any, optional): Дополнительные данные.
            use_local_models (bool): Использовать локальные веса.
            config_obj (Any, optional): Объект конфигурации (Config).
        """
        self.db = db
        self.use_local_models = use_local_models

        self.config = config_obj if config_obj is not None else config

        self.contextual_adapter = get_contextual_adapter(enabled=adapter_enabled)

        # Перенаправляем bert_model и sbert на общую модель синглтона
        self.bert_model = self.contextual_adapter.sbert_model
        self.sbert = self.contextual_adapter.sbert_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mbti_classes = MBTI_TYPES

        # Zero-shot экстрактор интересов (Tier 1.5, fallback если нейросеть недоступна):
        # переиспользует уже загруженный self.bert_model, эталонные эмбеддинги якорей
        # считаются один раз здесь (см. _build_anchor_embeddings). Порог поднят до 0.65 —
        # версия с точечным глобальным максимумом сходства (вместо mean по якорям) даёт
        # более резкое разделение сигнал/шум, поэтому старый threshold 0.40 занижен.
        semantic_threshold = getattr(self.config, "INTEREST_SEMANTIC_THRESHOLD", 0.65)
        self.semantic_extractor = ZeroShotInterestExtractor(
            bert_model=self.bert_model,
            taxonomy=INTEREST_TAXONOMY,
            threshold=semantic_threshold,
        )

        # Обучаемая PyTorch-голова интересов (Tier 1, основной путь если есть веса):
        # load_neural_extractor никогда не бросает исключение — если файла весов нет
        # (ещё не обучали) или он битый/несовместимый по архитектуре, self.neural_extractor
        # остаётся None, и extract_interests() тихо использует semantic_extractor вместо неё.
        neural_weights_path = getattr(
            self.config, "INTEREST_HEAD_WEIGHTS_PATH", os.path.join("ml", "artifacts", "interest_head.pth")
        )
        neural_threshold = getattr(self.config, "INTEREST_NEURAL_THRESHOLD", 0.55)
        self.neural_extractor = load_neural_extractor(
            bert_model=self.bert_model,
            weights_path=neural_weights_path,
            threshold=neural_threshold,
        )

        self.model = PersonalityClassifier(input_size=388, num_bins=NUM_BINS).to(self.device)
        self.mbti_model = MBTIClassifier(
            input_size=384, num_classes=len(self.mbti_classes)
        ).to(self.device)

        self.model_path = self.config.PERSONALITY_MODEL_PATH
        self.mbti_model_path = self.config.MBTI_MODEL_PATH

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

        adapter_enabled = getattr(self.config, "CONTEXTUAL_ADAPTER_ENABLED", True)
        self.contextual_adapter = get_contextual_adapter(enabled=adapter_enabled)

    @staticmethod
    def _ocean_to_soft_probs(mbti_from_ocean: str, temperature: float = 0.5) -> np.ndarray:
        """
        Формирует softmax-вероятности по 16 MBTI на основе типа, выведенного из OCEAN-оценок.

        Args:
            mbti_from_ocean (str): Прогноз MBTI по OCEAN (например, "INTJ").
            temperature (float): Коэффициент размытия распределения.

        Returns:
            np.ndarray: Вектор вероятностей для MBTI.
        """
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
        """
        Простейшие ручные признаки по тексту: длина, доля капса, "!", "?".

        Args:
            text (str): Текст пользователя.

        Returns:
            list[float]: Четыре признака (дробные значения).
        """
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
        """
        Основная функция анализа: вычисляет OCEAN, MBTI, стиль коммуникации и интересы.

        Args:
            text (str): Входной текст.

        Returns:
            dict[str, Any] | None: Результаты анализа либо None при пустом тексте.
        """
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
        """
        Чтение веса для смеси MBTI-сетевой и rule-based моделей из конфигурации.

        Returns:
            float: Вес в диапазоне [0, 1]. По дефолту 0.7 (если модель есть), иначе 0.0.
        """
        try:
            val = float(self.config.MBTI_NEURAL_BLEND_WEIGHT)
            return float(max(0.0, min(1.0, val)))
        except (TypeError, ValueError):
            return 0.7 if self.has_mbti_model else 0.0

    def infer_mbti(self, scores: list[float]) -> str:
        """
        Получает MBTI-тип по значениям OCEAN (грубая логика порогов).

        Args:
            scores (list[float]): Значения OCEAN в диапазоне [0, 1].

        Returns:
            str: MBTI-тип (например, "INTJ").
        """
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
        """
        Анализ стиля коммуникации пользователя.

        Args:
            text (str): Входной текст.
            scores (list[float]): Значения OCEAN.

        Returns:
            dict[str, Any]: Словарь характеристик стиля общения.
        """
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
        """
        Извлечение интересов, навыков и ценностей из текста — гибридный трёхуровневый pipeline.

        Сценарий А (self.neural_extractor доступен, веса обучены):
            Текст режется на предложения, каждое кодируется через self.bert_model.encode
            и пропускается через обученную CustomInterestClassifier (softmax + threshold).
            Даёт жёсткую классификацию по тегам платформы с наивысшей точностью.

        Сценарий Б (весов нет или классификатор не обучен / self.neural_extractor is None,
        либо сценарий А не нашёл ни одного совпадения выше threshold):
            Автоматически используется self.semantic_extractor — улучшенный zero-shot
            (точечный максимум сходства с якорями таксономии, без усреднения).

        Tier 0 (оба метода выше вернули пустоту для категорий):
            Полный откат на _rule_based_extract для hobbies/topics/skills/occupation.

        Поля dislikes / short_term_goals / long_term_goals / values / preferences /
        work_style ВСЕГДА считаются rule-based методом (_rule_based_extract) независимо
        от того, сработала ли нейросеть или zero-shot — ни эмбеддинги, ни голова над ними
        не заточены под детекцию негации ("не люблю спорт") и явных маркеров-триггеров,
        для них регулярки остаются точнее и дешевле.

        Args:
            text (str): Текст для анализа.

        Returns:
            dict[str, Any]: Словарь найденных интересов, скиллов, целей и т.д.
                Дополнительно содержит:
                - "semantic_categories" — сырой результат сработавшего Tier
                  (subcategory/score/evidence по каждой глобальной категории),
                  пригодный для прямой записи в граф знаний (KnowledgeNode.category);
                - "extraction_method" — какой именно Tier сработал: "neural" | "zero_shot" | "rule_based",
                  полезно для мониторинга (как часто обученная голова реально используется)
                  и для отладки без необходимости включать debug-логи.
        """
        if not text or not text.strip():
            result = self._rule_based_extract(text)
            result["extraction_method"] = "rule_based"
            return result

        cleaned = self.clean_text(text).lower()
        rule_based = self._rule_based_extract(text)

        if not cleaned:
            rule_based["extraction_method"] = "rule_based"
            return rule_based

        categorized: dict[str, list[dict[str, Any]]] = {}
        extraction_method = "rule_based"

        # Сценарий А: обученная голова, если веса были найдены и загружены в __init__
        if self.neural_extractor is not None:
            categorized = self.neural_extractor.extract(cleaned)
            if categorized:
                extraction_method = "neural"

        # Сценарий Б: голова недоступна ИЛИ ничего не нашла -> улучшенный zero-shot
        if not categorized:
            categorized = self.semantic_extractor.extract(cleaned)
            if categorized:
                extraction_method = "zero_shot"

        if not categorized:
            # ни нейросеть, ни zero-shot не дали сигнала -> полный откат на Tier 0
            rule_based["extraction_method"] = "rule_based"
            return rule_based

        hobbies = [item["subcategory"] for item in categorized.get("hobby", [])]
        skills = [item["subcategory"] for item in categorized.get("work", [])]
        topics = [item["subcategory"] for item in categorized.get("psychology", [])]

        # occupation по классификации — только если топ-совпадение в "work" заметно
        # увереннее базового threshold (иначе оставляем rule-based значение как консервативное)
        occupation = rule_based.get("occupation")
        work_matches = categorized.get("work")
        if work_matches:
            top_work_score = work_matches[0]["score"]
            active_threshold = (
                self.neural_extractor.threshold
                if extraction_method == "neural"
                else self.semantic_extractor.threshold
            )
            if top_work_score >= max(active_threshold + 0.1, 0.5):
                occupation = work_matches[0]["subcategory"]

        return {
            "hobbies": hobbies or rule_based.get("hobbies", []),
            "topics": topics or rule_based.get("topics", []),
            "skills": skills or rule_based.get("skills", []),
            "dislikes": rule_based.get("dislikes", []),
            "occupation": occupation,
            "work_style": rule_based.get("work_style"),
            "short_term_goals": rule_based.get("short_term_goals", []),
            "long_term_goals": rule_based.get("long_term_goals", []),
            "preferences": rule_based.get("preferences", {}),
            "traits": list(dict.fromkeys(skills[:2] + topics[:2])) or rule_based.get("traits", []),
            "values": rule_based.get("values", []),
            "semantic_categories": categorized,
            "extraction_method": extraction_method,
        }

    def _rule_based_extract(self, text: str) -> dict[str, Any]:
        """
        Tier 0 — rule-based извлечение интересов (словари/регулярки).

        Надёжный fallback: используется, если zero-shot экстрактор (Tier 1) не
        дал ни одного совпадения выше threshold, а также как единственный источник
        полей dislikes/short_term_goals/values/preferences — семантическая
        классификация по эмбеддингам не заточена под детекцию негации и маркеров-триггеров,
        для них регулярки остаются адекватным и дешёвым решением.

        Args:
            text (str): Текст для анализа.

        Returns:
            dict[str, Any]: Словарь найденных интересов, скиллов, целей и т.д.
        """
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
        """
        Очищает и нормализует текст (вызывает вспомогательную функцию).

        Args:
            text (str | None): Исходный текст.

        Returns:
            str: Очищенный текст.
        """
        return clean_user_text(text)

    def calculate_compatibility(
            self,
            vec1: list[float] | np.ndarray,
            vec2: list[float] | np.ndarray,
            weights: list[float] | None = None
    ) -> float:
        """
        Рассчитывает процент совместимости между двумя признаковыми векторами.

        Args:
            vec1 (list[float]): Вектор 1 (например, OCEAN).
            vec2 (list[float]): Вектор 2.

        Returns:
            float: Совместимость [0; 100].
        """
        if not vec1 or not vec2:
            return 0.0

        v1, v2 = np.array(vec1, dtype=np.float32), np.array(vec2, dtype=np.float32)

        if weights is not None:
            w = np.array(weights, dtype=np.float32)
            w_norm = w / np.sum(w)
            distance = np.sqrt(np.sum(w_norm * (v1 - v2) ** 2))
        else:
            distance = np.sqrt(np.mean((v1 - v2) ** 2))

        return round(float(max(0.0, 100.0 * (1.0 - distance))), 2)

    def calculate_text_similarity(
            self, texts1: list[str] | str, texts2: list[str] | str
    ) -> float | np.ndarray:
        """
        Косинусное сходство между двумя или более текстами по эмбеддингам SBERT.

        Перед encode() тексты проходят через ContextualAdapter — сленг и
        аббревиатуры раскрываются в семантически богатые описания.

        Args:
            texts1 (list[str] | str): Первый текст или список.
            texts2 (list[str] | str): Второй текст или список.

        Returns:
            float | np.ndarray: Косинусное сходство (скаляр или матрица).
        """
        if isinstance(texts1, str):
            texts1 = [texts1]
        if isinstance(texts2, str):
            texts2 = [texts2]

        texts1 = self.contextual_adapter.prepare_for_encoding(texts1)
        texts2 = self.contextual_adapter.prepare_for_encoding(texts2)

        emb1 = self.bert_model.encode(texts1, convert_to_tensor=True)
        emb2 = self.bert_model.encode(texts2, convert_to_tensor=True)

        cos_sim_matrix = util.cos_sim(emb1, emb2)

        if len(texts1) == len(texts2) == 1:
            return cos_sim_matrix.item()
        return cos_sim_matrix.cpu().numpy()

    # Веса компонентов гибридного скора матчинга (calculate_hybrid_matching_score).
    # Сумма = 1.0. OCEAN — намеренно наименьший вес: служит tie-breaker'ом
    # внутри группы людей с одинаковыми тегами, но не может перебить их отсутствие
    # (если tag_score=0, максимум, который даст OCEAN=1.0 — это 0.2 итогового скора,
    # что ниже практически любого совпадения по тегам+семантике).
    _TAG_SCORE_WEIGHT: float = 0.5
    _SBERT_SCORE_WEIGHT: float = 0.3
    _OCEAN_SCORE_WEIGHT: float = 0.2

    @staticmethod
    def _extract_tag_set(interests: dict[str, Any] | None) -> set[str]:
        """Превращает JSON извлеченных интересов в плоское множество lowercase тегов."""
        if not interests:
            return set()
        tags = set()
        for field_name in ("hobbies", "topics", "skills"):
            items = interests.get(field_name) or []
            for item in items:
                if isinstance(item, dict):
                    val = item.get("subcategory") or item.get("name") or ""
                else:
                    val = str(item)
                val_clean = val.lower().strip()
                if val_clean:
                    tags.add(val_clean)

        occ = interests.get("occupation")
        if occ:
            if isinstance(occ, dict):
                occ_val = occ.get("name") or ""
            else:
                occ_val = str(occ)
            occ_clean = occ_val.lower().strip()
            if occ_clean:
                tags.add(occ_clean)

        return tags

    def _flatten_interests_to_text(self, interests: dict[str, Any] | None) -> str:
        """
        Собирает интересы пользователя в одну строку для SBERT-сравнения,
        когда сырой исходный текст (сообщения) недоступен вызывающей стороне.

        Args:
            interests: Результат extract_interests() / AIExtractedInterests.

        Returns:
            str: Строка вида "Python backend Разработка ...". Пустая строка, если
                интересов нет вовсе (тогда sbert_score в calculate_hybrid_matching_score
                будет 0.0, а не упадёт на пустом encode()).
        """
        if not interests:
            return ""
        parts: list[str] = []
        for field in ("hobbies", "skills", "topics"):
            parts.extend(interests.get(field) or [])
        if interests.get("occupation"):
            parts.append(interests["occupation"])
        raw = " ".join(str(p) for p in parts)
        return self.contextual_adapter.enrich_text(raw).enriched if raw else ""

    def calculate_hybrid_matching_score(
        self,
        search_query: str,
        other_user_extracted_interests: dict[str, Any] | None,
        other_user_raw_text: str = "",
        ocean_compatibility: float | None = None,
        current_user_ocean: list[float] | None = None,
        other_user_ocean: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Гибридный скор матчинга поискового запроса (или узла графа) с интересами
        другого пользователя. Заменяет сырой SBERT-мэтч строк, из-за которого
        запрос "написание кода" смешивался с геймерскими тегами (CS2 и т.п.).

        Formula:
            final = 0.5 * tag_score + 0.3 * sbert_score + 0.2 * ocean_score

        tag_score — жёсткий (0.0/1.0): пересеклись ли подкатегории таксономии
        между запросом и интересами другого пользователя (см. _extract_tag_set).
        Это доминирующий компонент — именно он не даёт геймеру с CS2 попасть
        в топ по запросу "написание кода", даже если общий текст профиля
        случайно похож по SBERT на уровне сырых строк.

        sbert_score — косинусное сходство запроса с текстом интересов другого
        пользователя (calculate_text_similarity), отрицательные значения
        клэмпаются в 0, чтобы не утягивать итоговый скор ниже 0.

        ocean_score — совместимость по OCEAN, ожидается уже готовым значением
        из таблицы `ai_user_compatibility` (overall_score, шкала [0, 100],
        см. UserCompatibility в data/ai.py) — эта функция не пересчитывает
        overall_score заново, только нормализует его в [0, 1]. Если готового
        значения нет, но переданы сырые OCEAN-векторы обоих пользователей,
        считается через self.calculate_compatibility(...) как запасной путь.
        Вес OCEAN всего 0.2, поэтому даже идеальная совместимость (1.0) не
        может перебить отсутствие тегов (tag_score=0.0 -> максимум итогового
        скора без тегов = 0.3 + 0.2 = 0.5 против, например, 0.5 + 0.3 + 0.0 = 0.8
        у человека с совпадающим тегом, но нулевой OCEAN-совместимостью).

        Args:
            search_query: Текст поискового запроса или заголовок/описание узла графа.
            other_user_extracted_interests: Результат extract_interests() другого
                пользователя (или его сериализация из AIExtractedInterests в БД,
                включая новое поле extraction_method — используется только для
                информационных целей, на формулу не влияет напрямую).
                None, если у пользователя ещё нет извлечённых интересов.
            other_user_raw_text: Сырой текст интересов другого пользователя
                (например, конкатенация его последних сообщений) для sbert_score.
                Если не передан — собирается эвристически из hobbies/skills/topics
                через _flatten_interests_to_text (менее точно, чем реальный сырой текст).
            ocean_compatibility: Готовое значение совместимости из таблицы БД,
                шкала [0, 100] (как overall_score в UserCompatibility). None, если
                не рассчитывалось / недоступно.
            current_user_ocean: OCEAN-вектор текущего пользователя — запасной
                путь, если ocean_compatibility не передан.
            other_user_ocean: OCEAN-вектор другого пользователя — используется
                вместе с current_user_ocean как запасной путь.

        Returns:
            dict[str, Any]: {
                "final_score": float,     # итоговый скор для сортировки, [0, 1]
                "tag_score": float,       # 0.0 или 1.0
                "sbert_score": float,     # [0, 1] после клэмпа
                "ocean_score": float,     # [0, 1]
                "matched_tags": list[str],   # какие именно подкатегории совпали
                "query_tags": list[str],     # теги, извлечённые из search_query
                "other_user_tags": list[str],
            }
        """
        if not search_query or not search_query.strip():
            return {
                "final_score": 0.0,
                "tag_score": 0.0,
                "sbert_score": 0.0,
                "ocean_score": 0.0,
                "matched_tags": [],
                "query_tags": [],
                "other_user_tags": [],
            }

        enriched_query = self.contextual_adapter.enrich_text(search_query).enriched

        # Шаг 1-2: структурированные категории из обогащённого поискового запроса
        query_interests = self.extract_interests(enriched_query)
        query_tags = self._extract_tag_set(query_interests)

        # Шаг 3-4: жёсткий тег-скор против интересов другого пользователя
        other_tags = self._extract_tag_set(other_user_extracted_interests)
        matched_tags = query_tags & other_tags
        tag_score = 1.0 if matched_tags else 0.0

        # Шаг 5: семантический скор по сырым строкам
        other_text = other_user_raw_text.strip() if other_user_raw_text else ""
        if not other_text:
            other_text = self._flatten_interests_to_text(other_user_extracted_interests)

        if other_text.strip():
            raw_similarity = self.calculate_text_similarity(enriched_query, other_text)
            sbert_score = max(0.0, float(raw_similarity))
        else:
            sbert_score = 0.0

        # OCEAN — tie-breaker, готовое значение из БД в приоритете над пересчётом
        if ocean_compatibility is not None:
            ocean_score = max(0.0, min(1.0, float(ocean_compatibility) / 100.0))
        elif current_user_ocean and other_user_ocean:
            ocean_score = max(0.0, self.calculate_compatibility(current_user_ocean, other_user_ocean) / 100.0)
        else:
            ocean_score = 0.0

        final_score = (
            self._TAG_SCORE_WEIGHT * tag_score
            + self._SBERT_SCORE_WEIGHT * sbert_score
            + self._OCEAN_SCORE_WEIGHT * ocean_score
        )

        return {
            "final_score": round(float(final_score), 4),
            "tag_score": tag_score,
            "sbert_score": round(sbert_score, 4),
            "ocean_score": round(ocean_score, 4),
            "matched_tags": sorted(matched_tags),
            "query_tags": sorted(query_tags),
            "other_user_tags": sorted(other_tags),
        }

    @staticmethod
    def get_compatible_mbti(mbti_type: str) -> list[str]:
        """
        Список наиболее совместимых MBTI-типов для заданного.

        Args:
            mbti_type (str): MBTI-тип (например, "INTJ").

        Returns:
            list[str]: Список совместимых типов.
        """
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
