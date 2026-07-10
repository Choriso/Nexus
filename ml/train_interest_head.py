"""
train_interest_head.py
Обучает CustomInterestClassifier на якорях таксономии и сохраняет веса.
Запуск: python train_interest_head.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ai_profiler.interest_extractor import (
    CustomInterestClassifier,
    build_labels_from_taxonomy,
)
from app.ai_profiler.taxonomy import INTEREST_TAXONOMY
from app.ai_profiler.contextual_adapter import get_contextual_adapter
from config import config


def build_training_data_from_taxonomy():
    label_order = build_labels_from_taxonomy(INTEREST_TAXONOMY)
    class_to_idx = {pair: idx for idx, pair in enumerate(label_order)}
    texts = []
    labels = []

    for global_cat, subcats in INTEREST_TAXONOMY.items():
        for subcat, anchors in subcats.items():
            pair = (global_cat, subcat)
            if pair not in class_to_idx:
                print(f"WARNING: {subcat} not in label_order")
                continue
            idx = class_to_idx[pair]
            for anchor in anchors:
                anchor_clean = anchor.strip()
                if anchor_clean:
                    texts.append(anchor_clean)
                    labels.append(idx)
    print(f"Training set: {len(texts)} examples, {len(label_order)} classes")
    return texts, labels, label_order


def main():
    adapter = get_contextual_adapter(enabled=False)
    bert_model = adapter.sbert_model

    texts, labels, label_order = build_training_data_from_taxonomy()

    embedding_dim = bert_model.get_embedding_dimension()
    classifier = CustomInterestClassifier(
        embedding_dim=embedding_dim,
        labels=label_order,
        hidden_dim=128,
        dropout=0.3,
        bert_model=bert_model,
    )

    classifier.train_on_data(texts, labels, epochs=15, lr=0.001)

    weights_path = getattr(config, "INTEREST_HEAD_WEIGHTS_PATH", "artifacts/interest_head.pth")
    classifier.save(weights_path)
    print(f"Model saved to {weights_path}")


if __name__ == "__main__":
    main()