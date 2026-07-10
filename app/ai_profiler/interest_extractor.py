"""
Гибридный экстрактор интересов: обучаемая PyTorch-голова (Tier 1) поверх
эмбеддингов SBERT + zero-shot классификация (Tier 1.5, fallback) + rule-based
(Tier 0, финальный fallback, реализован в core.py как _rule_based_extract).

Иерархия вызова (см. AIProfiler.extract_interests в core.py):
    1. CustomInterestClassifier — если веса обучены и лежат в ml/artifacts/interest_head.pth,
       даёт жёсткую классификацию по тегам платформы с высокой точностью.
    2. ZeroShotInterestExtractor — если весов нет или классификатор ничего не нашёл,
       используется точечное сравнение эмбеддингов чанков с якорями таксономии.
    3. Rule-based (_rule_based_extract в core.py) — если оба метода выше вернули пустоту.

Оба Tier 1/1.5 метода отдают результат в ОДНОМ формате:
    {"hobby": [{"subcategory": str, "score": float, "evidence": str}, ...], "work": [...], ...}
что позволяет core.py работать с ними взаимозаменяемо, не зная, какой именно
метод сработал.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import util

logger = logging.getLogger(__name__)

# Разбиваем на предложения, а не на отдельные токены — интерес привязан к
# контексту фразы. Общий хелпер для zero-shot и нейросетевого экстрактора.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_MIN_CHUNK_LEN = 8


def split_into_chunks(text: str) -> list[str]:
    """
    Разбивает текст на смысловые чанки (предложения) для батч-инференса.

    Используется и ZeroShotInterestExtractor, и NeuralInterestExtractor —
    единая логика чанкования гарантирует, что оба метода видят один и тот же
    "evidence"-контекст при сравнении результатов.

    Args:
        text: Очищенный текст пользователя.

    Returns:
        list[str]: Непустые чанки длиной от _MIN_CHUNK_LEN символов.
    """
    if not text:
        return []
    raw_chunks = _SENTENCE_SPLIT_RE.split(text)
    return [c.strip() for c in raw_chunks if len(c.strip()) >= _MIN_CHUNK_LEN]


def build_labels_from_taxonomy(taxonomy: dict[str, dict[str, list[str]]]) -> list[tuple[str, str]]:
    """
    Строит упорядоченный список меток (global_category, subcategory) из таксономии.

    Порядок этого списка = порядок классов классификатора (индекс в списке —
    это class_id, который передаётся в train_on_data как labels). Именно
    поэтому таксономию нельзя тихо менять между обучением головы и инференсом
    без переобучения — состав/порядок классов зашит в веса через checkpoint.

    Args:
        taxonomy: Словарь {global_category: {subcategory: [anchors]}}.

    Returns:
        list[tuple[str, str]]: [(global_category, subcategory), ...] в порядке обхода.
    """
    labels: list[tuple[str, str]] = []
    for global_category, subcategories in taxonomy.items():
        for subcategory in subcategories:
            labels.append((global_category, subcategory))
    return labels


class CustomInterestClassifier(nn.Module):
    """
    Лёгкая PyTorch-голова классификации интересов поверх фиксированных
    эмбеддингов SBERT (SBERT не дообучается, дообучается только эта надстройка).

    Архитектура: Linear -> ReLU -> Dropout -> Linear(num_classes).
    Обучается на CPU за секунды на небольшом размеченном наборе фраз.

    Args:
        embedding_dim: Размерность эмбеддинга SBERT (384 для all-MiniLM-L6-v2,
            768 для многоязычных mpnet-моделей). Должна совпадать с
            bert_model.get_sentence_embedding_dimension().
        labels: Упорядоченный список (global_category, subcategory) — см.
            build_labels_from_taxonomy(). Индекс в списке = class_id.
        hidden_dim: Размер скрытого слоя.
        dropout: Вероятность dropout между слоями.
        bert_model: Опциональная ссылка на SentenceTransformer — используется
            ТОЛЬКО для удобства в train_on_data() (кодирование сырых текстов).
            Не сериализуется в чекпоинт и не участвует в forward().
    """

    def __init__(
        self,
        embedding_dim: int,
        labels: list[tuple[str, str]],
        hidden_dim: int = 128,
        dropout: float = 0.3,
        bert_model: Any | None = None,
    ) -> None:
        super().__init__()
        if not labels:
            raise ValueError("labels не может быть пустым — нечего классифицировать")

        self.embedding_dim = embedding_dim
        self.labels = list(labels)
        self.hidden_dim = hidden_dim
        self.bert_model = bert_model  # не в state_dict, см. save()/load()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.labels)),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Тензор эмбеддингов SBERT формы (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Логиты формы (batch_size, num_classes).
        """
        return self.net(embeddings)

    def train_on_data(
        self,
        texts: list[str],
        labels: list[int],
        epochs: int = 10,
        lr: float = 0.001,
    ) -> "CustomInterestClassifier":
        """
        Дообучает голову на размеченном наборе фраз. Прогоняется на CPU,
        типичный размер выборки (десятки-сотни фраз) укладывается в единицы секунд.

        Args:
            texts: Список фраз-примеров (например, реальные сообщения пользователей
                с проставленной вручную категорией).
            labels: Список class_id (индекс в self.labels) для каждой фразы,
                той же длины, что texts.
            epochs: Число эпох полного батч-градиентного спуска.
            lr: Learning rate для Adam.

        Returns:
            CustomInterestClassifier: self, для чейнинга (classifier.train_on_data(...).save(...)).

        Raises:
            RuntimeError: если bert_model не передан в конструктор — кодировать
                тексты нечем.
            ValueError: если длины texts и labels не совпадают.
        """
        if self.bert_model is None:
            raise RuntimeError(
                "CustomInterestClassifier создан без bert_model — train_on_data() "
                "не может закодировать тексты. Передайте bert_model в конструктор."
            )
        if len(texts) != len(labels):
            raise ValueError(f"len(texts)={len(texts)} != len(labels)={len(labels)}")

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # SBERT не дообучается — эмбеддинги считаются один раз, градиент через
        # них не пробрасывается (no_grad), обучается только self.net поверх них.
        with torch.inference_mode(False):  # явно отключаем inference mode
            embeddings = self.bert_model.encode(texts, convert_to_tensor=True)
        # клонируем, чтобы гарантированно работать с обычным тензором
        embeddings = embeddings.clone()
        label_tensor = torch.tensor(labels, dtype=torch.long)

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.forward(embeddings)
            loss = criterion(logits, label_tensor)
            loss.backward()
            optimizer.step()
            logger.debug("interest_head epoch=%d loss=%.4f", epoch, float(loss))

        self.eval()
        return self

    def save(self, path: str) -> None:
        """
        Сохраняет веса + метаданные (labels, embedding_dim, hidden_dim) одним
        чекпоинтом — без labels веса бессмысленны (не знаем, какой class_id
        какой подкатегории соответствует).

        Args:
            path: Путь к .pth файлу (например, из config.INTEREST_HEAD_WEIGHTS_PATH).
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "labels": self.labels,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, bert_model: Any | None = None) -> "CustomInterestClassifier":
        """
        Восстанавливает классификатор из чекпоинта, включая архитектуру
        (embedding_dim/hidden_dim/labels берутся из самого файла, а не из
        текущей таксономии в коде — это защищает от рассинхрона, если
        таксономию поменяли, а голову не переобучили).

        Args:
            path: Путь к .pth файлу.
            bert_model: Опционально — для последующего train_on_data() (дообучение).

        Returns:
            CustomInterestClassifier: Модель в режиме eval(), готовая к инференсу.

        Raises:
            Любое исключение torch.load/несовпадения архитектуры пробрасывается
            наверх — обработка "тихого" fallback лежит на load_neural_extractor().
        """
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            embedding_dim=checkpoint["embedding_dim"],
            labels=checkpoint["labels"],
            hidden_dim=checkpoint.get("hidden_dim", 128),
            bert_model=bert_model,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class NeuralInterestExtractor:
    """
    Обёртка инференса над CustomInterestClassifier: чанкование текста,
    softmax по классам, отсечение по threshold, сборка результата в формате,
    идентичном ZeroShotInterestExtractor — для бесшовной подмены в оркестраторе.

    Args:
        classifier: Обученный CustomInterestClassifier (уже в режиме eval()).
        bert_model: SentenceTransformer для кодирования чанков текста.
        threshold: Порог уверенности softmax [0, 1]. Так как это вероятность,
            а не косинусное сходство — семантика другая, чем у ZeroShotInterestExtractor.threshold,
            дефолт ниже (0.55) осмысленно отличается от 0.65 у zero-shot.
        max_per_group: Максимум подкатегорий на глобальную категорию в результате.
    """

    def __init__(
        self,
        classifier: CustomInterestClassifier,
        bert_model: Any,
        threshold: float = 0.55,
        max_per_group: int = 5,
    ) -> None:
        self.classifier = classifier
        self.classifier.eval()
        self.bert_model = bert_model
        self.threshold = threshold
        self.max_per_group = max_per_group

    def extract(self, cleaned_text: str) -> dict[str, list[dict[str, Any]]]:
        """
        Классифицирует текст обученной головой, по одному предсказанию на чанк.

        Args:
            cleaned_text: Текст после clean_user_text().

        Returns:
            dict[str, list[dict[str, Any]]]: Тот же формат, что и
            ZeroShotInterestExtractor.extract() — {"hobby": [...], "work": [...], ...}.
            {} если чанков нет или ни один не преодолел threshold.
        """
        chunks = split_into_chunks(cleaned_text)
        if not chunks:
            return {}

        embeddings = self.bert_model.encode(chunks, convert_to_tensor=True)

        with torch.no_grad():
            logits = self.classifier(embeddings)
            probs = F.softmax(logits, dim=-1)

        best_probs, best_idx = torch.max(probs, dim=-1)

        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk_idx, (prob, class_id) in enumerate(zip(best_probs.tolist(), best_idx.tolist())):
            if prob < self.threshold:
                continue
            global_category, subcategory = self.classifier.labels[int(class_id)]
            result[global_category].append(
                {
                    "subcategory": subcategory,
                    "score": round(float(prob), 3),
                    "evidence": chunks[chunk_idx],
                }
            )

        if not result:
            return {}

        for global_category, matches in result.items():
            matches.sort(key=lambda item: item["score"], reverse=True)
            result[global_category] = matches[: self.max_per_group]

        return dict(result)


def load_neural_extractor(
    bert_model: Any,
    weights_path: str,
    threshold: float = 0.55,
    max_per_group: int = 5,
) -> NeuralInterestExtractor | None:
    """
    Пытается загрузить обученную голову с диска. Никогда не бросает исключение
    наружу — при отсутствии файла или любой ошибке загрузки (битый чекпоинт,
    несовпадение архитектуры после смены таксономии без переобучения и т.п.)
    тихо возвращает None, и оркестратор в core.py переключается на Zero-shot.

    Args:
        bert_model: SentenceTransformer для кодирования текста при инференсе.
        weights_path: Путь к .pth файлу (например, ml/artifacts/interest_head.pth).
        threshold: Порог уверенности softmax для NeuralInterestExtractor.
        max_per_group: Максимум подкатегорий на группу.

    Returns:
        NeuralInterestExtractor | None: None, если весов нет или они не загрузились.
    """
    if not weights_path or not os.path.isfile(weights_path):
        logger.info(
            "Веса CustomInterestClassifier не найдены (%s) — используется zero-shot режим.",
            weights_path,
        )
        return None

    try:
        classifier = CustomInterestClassifier.load(weights_path, bert_model=bert_model)
    except Exception:
        logger.warning(
            "Не удалось загрузить веса CustomInterestClassifier из %s — откат на zero-shot режим.",
            weights_path,
            exc_info=True,
        )
        return None

    return NeuralInterestExtractor(
        classifier, bert_model, threshold=threshold, max_per_group=max_per_group
    )


class ZeroShotInterestExtractor:
    """
    Zero-shot классификатор интересов по эмбеддингам SBERT — Tier 1.5
    (fallback для CustomInterestClassifier, основной метод если голова не обучена).

    В отличие от первой версии, НЕ усредняет якоря подкатегории в один вектор
    (mean(dim=0)): на лёгких моделях вроде all-MiniLM-L6-v2 усреднение "схлопывает"
    эмбеддинги и поднимает фоновый шум нерелевантных категорий до 0.55-0.65,
    размывая результат. Вместо этого хранится полная матрица эмбеддингов
    якорей подкатегории, и берётся точечный глобальный максимум сходства
    (конкретный чанк x конкретный якорь), что подняло точность целевых
    категорий до 0.75+ в тестах на реальной модели.

    Args:
        bert_model: Уже инициализированный экземпляр SentenceTransformer.
        taxonomy: Словарь {global_category: {subcategory: [anchor_phrases]}}.
        threshold: Порог косинусного сходства [0, 1]. Оптимально: 0.65-0.68
            (поднят с 0.40 в первой версии — точечный максимум даёт более
            резкое и надёжное разделение сигнал/шум, чем сравнение со средним).
        max_per_group: Максимум подкатегорий на одну глобальную категорию.
    """

    def __init__(
        self,
        bert_model: Any,
        taxonomy: dict[str, dict[str, list[str]]],
        threshold: float = 0.65,
        max_per_group: int = 5,
    ) -> None:
        self.bert_model = bert_model
        self.taxonomy = taxonomy
        self.threshold = threshold
        self.max_per_group = max_per_group

        # (global_category, subcategory) -> полная матрица эмбеддингов якорей (n_anchors, dim)
        self._anchor_embeddings: dict[tuple[str, str], torch.Tensor] = {}
        self._build_anchor_embeddings()

    def _build_anchor_embeddings(self) -> None:
        """
        Кодирует и кеширует эмбеддинги КАЖДОГО якоря индивидуально (без усреднения).
        Вызывается один раз при инициализации экстрактора в Celery-воркере.
        """
        self._anchor_embeddings = {}
        for global_category, subcategories in self.taxonomy.items():
            for subcategory, anchors in subcategories.items():
                clean_anchors = [a.strip() for a in anchors if a and a.strip()]
                if not clean_anchors:
                    continue
                anchor_embs = self.bert_model.encode(clean_anchors, convert_to_tensor=True)
                self._anchor_embeddings[(global_category, subcategory)] = anchor_embs

    def extract(self, cleaned_text: str) -> dict[str, list[dict[str, Any]]]:
        """
        Классифицирует текст методом точечного (не усреднённого) zero-shot сравнения.

        Args:
            cleaned_text: Текст после clean_user_text().

        Returns:
            dict[str, list[dict[str, Any]]]: {"hobby": [...], "work": [...], "psychology": [...]}.
            {} если текст пуст, якорей нет, чанков нет или ни один не преодолел threshold.
        """
        if not cleaned_text or not self._anchor_embeddings:
            return {}

        chunks = split_into_chunks(cleaned_text)
        if not chunks:
            return {}

        # Эмбеддинги чанков пользователя: форма (n_chunks, embedding_dim)
        chunk_embeddings = self.bert_model.encode(chunks, convert_to_tensor=True)

        result: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for (global_category, subcategory), anchor_embs in self._anchor_embeddings.items():
            # Матрица сходства: форма (n_chunks, n_anchors)
            sims = util.cos_sim(chunk_embeddings, anchor_embs)

            # Глобальный максимум по всей матрице: какой чанк дал лучшее совпадение
            # с каким-либо из якорей подкатегории (не усреднение, а "лучший выстрел").
            max_idx = int(torch.argmax(sims))
            n_anchors = sims.shape[1]
            best_chunk_idx = max_idx // n_anchors
            best_score = float(sims[best_chunk_idx][max_idx % n_anchors])

            if best_score >= self.threshold:
                result[global_category].append(
                    {
                        "subcategory": subcategory,
                        "score": round(best_score, 3),
                        "evidence": chunks[best_chunk_idx],
                    }
                )

        if not result:
            return {}

        for global_category, matches in result.items():
            matches.sort(key=lambda item: item["score"], reverse=True)
            result[global_category] = matches[: self.max_per_group]

        return dict(result)
