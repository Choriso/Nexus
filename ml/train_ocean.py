"""
train_ocean.py — Обучение PersonalityClassifier с ординальной классификацией.

Нововведения:
  - Ordinal loss (Cumulative BCE) вместо регрессии.
  - Веса примеров: 5.0 для качественных, 1.0 для синтетики.
  - CosineAnnealingWarmRestarts + warmup.
  - Early Stopping с восстановлением.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional
import random
import json

from core import PersonalityClassifier, scores_to_bins, NUM_BINS, BIN_EDGES, bins_to_score

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Датасет
# ---------------------------------------------------------------------------
class PersonalityDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = torch.load(data_path)
        print(f"Загружено {len(self.data)} примеров.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item["features"], dtype=torch.float32)
        target = torch.tensor(item["target"], dtype=torch.float32)    # (5,) непрерывные значения
        weight = item.get("weight", 1.0)
        # Преобразуем таргет в бины для ординальной классификации
        target_bins = torch.tensor(scores_to_bins(item["target"], NUM_BINS), dtype=torch.long)  # (5,) индексы
        return features, target_bins, weight, target  # сохраняем оригинальный target для метрик


# ---------------------------------------------------------------------------
# Ordinal Loss (Cumulative Binary Cross-Entropy)
# ---------------------------------------------------------------------------
def ordinal_loss(logits, target_bins):
    """
    logits: (batch, num_traits, num_bins) — логиты до softmax
    target_bins: (batch, num_traits) — индексы бинов 0..num_bins-1
    """
    B, T, K = logits.shape
    # Строим кумулятивные метки: P(y_i > k) для k=0..K-2
    # target_bins: (B, T) -> (B, T, 1) сравнивается с порогами
    # Для бинарного классификатора для порога k: 1 если target > k
    cum_targets = (target_bins.unsqueeze(-1) > torch.arange(K-1, device=logits.device)).float()  # (B, T, K-1)
    # Логиты для кумулятивных вероятностей: берём логиты для всех бинов, кроме последнего?
    # Ординальная регрессия: модель выдаёт логиты для каждого бина; мы можем интерпретировать их как ненормализованные вероятности P(y = k).
    # Преобразуем в кумулятивные логиты: P(y > k) = sum_{j>k} P(y = j). Нужно сделать softmax по бинам, затем cumulative.
    # Стандартный способ: отдельные бинарные классификаторы для каждого порога. Проще сделать так:
    probs = F.softmax(logits, dim=-1)                     # (B, T, K)
    cum_probs = torch.cumsum(probs, dim=-1)               # P(y <= k)
    # Вероятность того, что y > k = 1 - P(y <= k)
    p_greater = 1.0 - cum_probs[:, :, :-1]                # (B, T, K-1)
    # Бинарная кросс-энтропия
    loss = F.binary_cross_entropy(p_greater, cum_targets, reduction='none')
    return loss.mean()


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=40, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_r2 = -float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, r2, model):
        if r2 > self.best_r2 + self.min_delta:
            self.best_r2 = r2
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"  [EarlyStopping] Восстановлены веса с лучшим R2={self.best_r2:.4f}")


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------
def train_model(train_loader, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalityClassifier(
        input_size=388,
        hidden_dims=config.get("hidden_dims", [256, 128, 64]),
        dropout=config.get("dropout", 0.3),
        num_bins=NUM_BINS
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    T_0 = config.get("T_0", 15)
    T_mult = config.get("T_mult", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=config.get("min_lr", 1e-6)
    )
    warmup_epochs = config.get("warmup_epochs", 5)
    base_lr = config.get("lr", 0.001)

    early_stopping = EarlyStopping(patience=config.get("es_patience", 40))
    epochs = config.get("epochs", 200)
    traits = ["O", "C", "E", "A", "N"]
    history = {"train_loss": [], "test_loss": [], "r2_scores": [], "lr": []}
    best_r2 = -float("inf")

    print(f"Конфиг: lr={base_lr}, epochs={epochs}, batch={config.get('batch_size', 32)}, "
          f"hidden_dims={config.get('hidden_dims')}, ordinal bins={NUM_BINS}\n")

    for epoch in range(epochs):
        # Warmup
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # --- TRAIN ---
        model.train()
        train_epoch_loss = 0.0
        for x, target_bins, weights, _ in train_loader:
            x, target_bins, weights = x.to(device), target_bins.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(x)                     # (batch, 5, num_bins)
            loss_per_sample = ordinal_loss(logits, target_bins)
            # Взвешиваем samples
            loss = (loss_per_sample.unsqueeze(0) * weights).mean()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        if epoch >= warmup_epochs:
            scheduler.step(epoch - warmup_epochs)

        # --- EVAL ---
        model.eval()
        test_epoch_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, target_bins, _, orig_targets in test_loader:
                x = x.to(device)
                target_bins = target_bins.to(device)
                logits = model(x)
                loss = ordinal_loss(logits, target_bins)
                test_epoch_loss += loss.item()

                # Преобразуем в непрерывные предикты
                probs = F.softmax(logits, dim=-1)   # (batch, 5, num_bins)
                scores = bins_to_score(probs)        # (batch, 5)
                all_preds.append(scores.cpu().numpy())
                all_targets.append(orig_targets.numpy())

        preds_np = np.vstack(all_preds)
        targets_np = np.vstack(all_targets)

        avg_train = train_epoch_loss / len(train_loader)
        avg_test = test_epoch_loss / len(test_loader)
        r2 = r2_score(targets_np, preds_np)
        r2_ind = r2_score(targets_np, preds_np, multioutput="raw_values")
        mae = mean_absolute_error(targets_np, preds_np)
        dir_acc = np.mean((preds_np > 0.5) == (targets_np > 0.5))
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train)
        history["test_loss"].append(avg_test)
        history["r2_scores"].append(r2)
        history["lr"].append(cur_lr)

        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), ARTIFACTS_DIR / "personality_model_best.pth")
            details = " | ".join([f"{traits[i]}: {r2_ind[i]:.3f}" for i in range(5)])
            print(f"Эпоха {epoch:03d} | MAE: {mae:.4f} | DirAcc: {dir_acc:.2%} | "
                  f"Loss: {avg_test:.4f} | LR: {cur_lr:.2e} | New Best R2: {r2:.4f} ({details})")
        elif epoch % 10 == 0:
            print(f"Эпоха {epoch:03d} | R2: {r2:.4f} | MAE: {mae:.4f} | "
                  f"DirAcc: {dir_acc:.2%} | Loss: {avg_test:.4f} | LR: {cur_lr:.2e}")

        if early_stopping.step(r2, model):
            print(f"\n[EarlyStopping] Остановка на эпохе {epoch}.")
            early_stopping.restore_best(model)
            break

    plot_training_results(history)

    print("\n--- АНАЛИЗ РАЗБРОСА (STDEV) ---")
    for i, trait in enumerate(traits):
        pred_std = np.std(preds_np[:, i])
        target_std = np.std(targets_np[:, i])
        print(f"  Черта {trait}: Предсказано STD={pred_std:.3f} | В данных STD={target_std:.3f}")

    return model


def plot_training_results(history):
    if not history["train_loss"]:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["test_loss"], label="Test")
    axes[0].set_title("Ordinal BCE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["r2_scores"], color="green")
    axes[1].axhline(max(history["r2_scores"]), color="green", linestyle="--", alpha=0.4)
    axes[1].set_title(f"R² Score (best: {max(history['r2_scores']):.4f})")
    axes[1].set_xlabel("Epoch")
    axes[2].plot(history["lr"], color="orange")
    axes[2].set_title("LR (Cosine Annealing)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "training_report.png", dpi=120)
    plt.close()
    print(f"\nГрафик сохранён: {ARTIFACTS_DIR / 'training_report.png'}")


def save_artifacts(model_obj):
    target = ARTIFACTS_DIR / "personality_model.pth"
    torch.save(model_obj.state_dict(), target)
    print(f"Веса модели сохранены: {target}")
    return target


def main():
    cache_path = "data/train_data_precomputed.pt"
    if not Path(cache_path).exists():
        print(f"❌ Файл {cache_path} не найден!")
        return

    full_dataset = PersonalityDataset(cache_path)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    config = {
        "lr": 0.001,
        "epochs": 200,
        "batch_size": 32,
        "hidden_dims": [256, 128, 64],
        "dropout": 0.3,
        "T_0": 15,
        "T_mult": 2,
        "min_lr": 1e-6,
        "warmup_epochs": 5,
        "es_patience": 40,
        "es_min_delta": 1e-4,
    }

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"])

    model_obj = train_model(train_loader, test_loader, config)
    if model_obj:
        save_artifacts(model_obj)
    print("\n✅ Обучение завершено.")


if __name__ == "__main__":
    main()