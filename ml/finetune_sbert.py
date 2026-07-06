"""
finetune_sbert.py — Шаблон fine-tuning SBERT на парах сленг ↔ каноническое описание.

Цель: сдвинуть эмбеддинги paraphrase-multilingual-MiniLM-L12-v2 так, чтобы
«Flask» и «бэкенд на Python» оказались ближе в векторном пространстве.

Запуск (на домашней видеокарте с 8+ GB VRAM):
    python ml/finetune_sbert.py \
        --pairs data/slang_pairs.json \
        --output ml/artifacts/sbert-finetuned \
        --epochs 3 \
        --batch-size 16

Если файл пар не указан — используется встроенный стартовый датасет
из app/ai_profiler/semantic_ontology.py (FINETUNE_POSITIVE_PAIRS).

Зависимости: sentence-transformers, torch (уже в requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_pairs(pairs_path: Path | None) -> list[tuple[str, str]]:
    """
    Загружает пары (сленг, каноническое_описание) из JSON или встроенного датасета.

    Формат JSON:
        [
            {"anchor": "Flask", "positive": "бэкенд на Python"},
            {"anchor": "катнуть в кс", "positive": "компьютерные игры"}
        ]
    или кортежи: [["Flask", "бэкенд на Python"], ...]
    """
    if pairs_path is None or not pairs_path.exists():
        from app.ai_profiler.semantic_ontology import FINETUNE_POSITIVE_PAIRS
        logger.info(
            "Файл пар не найден, используем встроенный датасет (%d пар)",
            len(FINETUNE_POSITIVE_PAIRS),
        )
        return list(FINETUNE_POSITIVE_PAIRS)

    with pairs_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    pairs: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            pairs.append((item["anchor"], item["positive"]))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    logger.info("Загружено %d пар из %s", len(pairs), pairs_path)
    return pairs


def build_training_examples(
    pairs: list[tuple[str, str]],
) -> list:
    """Конвертирует пары в InputExample для sentence-transformers."""
    from sentence_transformers import InputExample

    examples = []
    for anchor, positive in pairs:
        examples.append(InputExample(texts=[anchor, positive]))
    return examples


def finetune(
    pairs: list[tuple[str, str]],
    output_dir: Path,
    *,
    base_model: str,
    epochs: int,
    batch_size: int,
    warmup_steps: int,
    evaluation_steps: int,
) -> None:
    """
    Запускает MultipleNegativesRankingLoss fine-tuning.

    MultipleNegativesRankingLoss — стандарт для обучения на парах:
    anchor и positive сближаются, остальные примеры батча — негативы in-batch.
    """
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    logger.info("Загрузка базовой модели: %s", base_model)
    model = SentenceTransformer(base_model)

    train_examples = build_training_examples(pairs)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Старт fine-tuning: %d пар, epochs=%d, batch_size=%d",
        len(pairs), epochs, batch_size,
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
        evaluation_steps=evaluation_steps if len(pairs) >= batch_size * 2 else 0,
    )

    logger.info("Модель сохранена в %s", output_dir)
    logger.info(
        "Для использования в Nexus установите: EMBEDDING_MODEL=%s", output_dir
    )


def export_pairs_template(output_path: Path) -> None:
    """Экспортирует шаблон JSON-файла для сбора пар из продакшена."""
    from app.ai_profiler.semantic_ontology import FINETUNE_POSITIVE_PAIRS

    template = [
        {"anchor": a, "positive": p, "source": "manual", "category": "work"}
        for a, p in FINETUNE_POSITIVE_PAIRS
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    logger.info("Шаблон пар экспортирован: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tuning SBERT на парах сленг ↔ каноническое описание"
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="JSON с парами anchor/positive (по умолчанию — встроенный датасет)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/artifacts/sbert-finetuned"),
        help="Директория для сохранения дообученной модели",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=f"Базовая SBERT (по умолчанию config.EMBEDDING_MODEL = {config.EMBEDDING_MODEL})",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--evaluation-steps", type=int, default=50)
    parser.add_argument(
        "--export-template",
        type=Path,
        default=None,
        help="Только экспортировать шаблон JSON пар и выйти",
    )

    args = parser.parse_args()

    if args.export_template:
        export_pairs_template(args.export_template)
        return

    pairs = load_pairs(args.pairs)
    if len(pairs) < 2:
        logger.error("Нужно минимум 2 пары для обучения, получено: %d", len(pairs))
        sys.exit(1)

    finetune(
        pairs,
        args.output,
        base_model=args.base_model or config.EMBEDDING_MODEL,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_steps=args.evaluation_steps,
    )


if __name__ == "__main__":
    main()
