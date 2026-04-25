"""
train_ocean.py — Обучение PersonalityClassifier (OCEAN) с гетероскедастичной моделью.

Изменения:
  - Новая архитектура: гетероскедастичная (mu + logvar) с BatchNorm и Dropout.
  - Loss: Gaussian NLL + опциональный variance-aware penalty.
  - Поддержка взвешивания примеров (качество / синтетика).
  - CosineAnnealingWarmRestarts + линейный warmup.
  - Увеличенная patience и эпохи.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, Optional
import random

from core import PersonalityClassifier  # новый класс из core.py

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Датасет (с поддержкой весов примеров)
# ---------------------------------------------------------------------------
class PersonalityDataset(Dataset):
    """Загружает предвычисленные признаки и цели из .pt файла.
       Ожидает список словарей с ключами: 'features', 'target', опционально 'weight'.
       Если 'weight' отсутствует, вес = 1.0.
    """
    def __init__(self, data_path: str):
        self.data = torch.load(data_path)
        print(f"Загружено {len(self.data)} примеров.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item["features"], dtype=torch.float32)
        target = torch.tensor(item["target"], dtype=torch.float32)
        weight = item.get("weight", 1.0)   # 3.0 для качественных, 1.0 для синтетики
        return features, target, weight


# ---------------------------------------------------------------------------
# Функции потерь
# ---------------------------------------------------------------------------
def gaussian_nll_loss(pred_tuple, target):
    """
    Отрицательный логарифм правдоподобия для гауссова распределения.
    pred_tuple: (mu, logvar) — оба тензора (batch, 5)
    """
    mu, logvar = pred_tuple
    precision = torch.exp(-logvar)
    loss = precision * (mu - target) ** 2 + logvar
    return loss.mean()


def variance_aware_loss(pred_tuple, target, lambda_var=0.2):
    """
    Основной NLL + штраф за несовпадение дисперсии предсказаний и целей в батче.
    """
    mu, logvar = pred_tuple
    nll = gaussian_nll_loss((mu, logvar), target)
    # Дисперсия mu и target в батче (по размерности признаков)
    pred_var = torch.var(mu, dim=0)      # (5,)
    target_var = torch.var(target, dim=0)
    var_loss = torch.mean((pred_var - target_var) ** 2)
    return nll + lambda_var * var_loss


# ---------------------------------------------------------------------------
# Early Stopping (сохранено как раньше)
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 40, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_r2 = -float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, r2: float, model: nn.Module) -> bool:
        if r2 > self.best_r2 + self.min_delta:
            self.best_r2 = r2
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"  [EarlyStopping] Восстановлены веса с лучшим R2={self.best_r2:.4f}")


# ---------------------------------------------------------------------------
# Главная функция обучения
# ---------------------------------------------------------------------------
def train_model(train_loader, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalityClassifier(input_size=388, hidden_dims=config.get("hidden_dims", [256, 128, 64]),
                                  dropout=config.get("dropout", 0.3)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))

    # --- Планировщик Cosine Annealing с теплым стартом ---
    T_0 = config.get("T_0", 15)          # период первого перезапуска
    T_mult = config.get("T_mult", 2)     # множитель периода
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=config.get("min_lr", 1e-6)
    )

    # --- Линейный warmup (первые warmup_epochs эпох) ---
    warmup_epochs = config.get("warmup_epochs", 5)
    base_lr = config.get("lr", 0.001)

    # --- Early Stopping ---
    early_stopping = EarlyStopping(
        patience=config.get("es_patience", 40),
        min_delta=config.get("es_min_delta", 1e-4)
    )

    criterion = variance_aware_loss  # можно сменить на gaussian_nll_loss, убрав lambda_var=0
    epochs = config.get("epochs", 200)
    traits = ["O", "C", "E", "A", "N"]
    history = {"train_loss": [], "test_loss": [], "r2_scores": [], "lr": []}
    best_r2 = -float("inf")

    print(f"Конфиг: lr={base_lr}, epochs={epochs}, batch={config.get('batch_size', 32)}, "
          f"hidden_dims={config.get('hidden_dims', [256,128,64])}\n")

    for epoch in range(epochs):
        # --- Warmup learning rate ---
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # --- Train ---
        model.train()
        train_epoch_loss = 0.0
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device).unsqueeze(1)  # w: (batch,1)
            optimizer.zero_grad()
            pred = model(x)                     # (mu, logvar)
            loss_per_sample = variance_aware_loss(pred, y)  # среднее по всем, но мы хотим взвесить
            # Взвешивание: умножаем средний loss на веса (приближённо, т.к. loss уже средний по батчу)
            # Чтобы точно взвесить, пересчитаем loss без усреднения
            mu, logvar = pred
            precision = torch.exp(-logvar)
            nll_per_sample = (precision * (mu - y) ** 2 + logvar).mean(dim=1)  # (batch,)
            # var_loss тоже нужно взвесить? Просто добавляем к взвешенному NLL
            pred_var = torch.var(mu, dim=0)
            target_var = torch.var(y, dim=0)
            var_loss = torch.mean((pred_var - target_var) ** 2)
            loss = (nll_per_sample * w.squeeze(1)).mean() + config.get("lambda_var", 0.2) * var_loss

            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        # --- Шаг Cosine Annealing (после warmup) ---
        if epoch >= warmup_epochs:
            scheduler.step(epoch - warmup_epochs)  # передаём номер эпохи относительно первого перезапуска

        # --- Eval ---
        model.eval()
        test_epoch_loss = 0.0
        all_mu, all_targets = [], []
        with torch.no_grad():
            for x, y, w in test_loader:
                x, y = x.to(device), y.to(device)
                mu, logvar = model(x)
                # Loss для статистики (без весов, для оценки)
                loss = gaussian_nll_loss((mu, logvar), y)
                test_epoch_loss += loss.item()
                all_mu.append(mu.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds_np = np.vstack(all_mu)
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
            print(f"\n[EarlyStopping] Остановка на эпохе {epoch}. "
                  f"R2 не улучшался {early_stopping.patience} эпох.")
            early_stopping.restore_best(model)
            break

    plot_training_results(history)

    print("\n--- АНАЛИЗ РАЗБРОСА (STDEV) ---")
    for i, trait in enumerate(traits):
        pred_std = np.std(preds_np[:, i])
        target_std = np.std(targets_np[:, i])
        print(f"  Черта {trait}: Предсказано STD={pred_std:.3f} | В данных STD={target_std:.3f}")

    return model


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def plot_training_results(history):
    if not history["train_loss"]:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["test_loss"], label="Test")
    axes[0].set_title("Loss (Gaussian NLL)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["r2_scores"], color="green")
    axes[1].axhline(max(history["r2_scores"]), color="green", linestyle="--", alpha=0.4)
    axes[1].set_title(f"R² Score (best: {max(history['r2_scores']):.4f})")
    axes[1].set_xlabel("Epoch")
    axes[2].plot(history["lr"], color="orange")
    axes[2].set_title("Learning Rate (CosineAnnealing)")
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


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main():
    cache_path = "data/train_data_precomputed.pt"
    if not Path(cache_path).exists():
        print(f"❌ Ошибка: Файл {cache_path} не найден!")
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
        "lambda_var": 0.2,          # коэффициент variance-aware штрафа
    }

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"])

    model_obj = train_model(train_loader, test_loader, config)
    if model_obj:
        save_artifacts(model_obj)
    print("\n✅ Обучение завершено.")


if __name__ == "__main__":
    main()