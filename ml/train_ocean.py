"""
train_ocean.py — Обучение PersonalityClassifier (OCEAN).

Обновления:
  - ReduceLROnPlateau: снижает LR если test-loss не падает N эпох
  - Early Stopping: останавливает обучение если R2 не растёт
  - Weighted MSE Loss (оригинальный, сохранён без изменений)
  - Поддержка смешивания старого train_data.json и нового CSV через prepare_dataset.py
"""

import random
from pathlib import Path
import json
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from app.ai_profiler.core import PersonalityClassifier
from app.ai_profiler.core import AIProfiler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Загрузка датасета
# ---------------------------------------------------------------------------

def load_dataset(json_path: str = "data/train_data.json") -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"text": item["text"], "target": item["scores"]} for item in data]


# ---------------------------------------------------------------------------
# Dataset / Augmentation
# ---------------------------------------------------------------------------

# 1. Изменяем класс Dataset, чтобы он просто отдавал готовые тензоры
class PersonalityDataset(Dataset):
    def __init__(self, data_path: str):
        # Загружаем всё сразу в оперативку (у тебя 32 ГБ, это легко влезет!)
        self.data = torch.load(data_path)
        print(f"Загружено {len(self.data)} кешированных примеров.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # features уже 388 (384 + 4), просто превращаем в тензор
        features = torch.tensor(item["features"], dtype=torch.float32)
        target = torch.tensor(item["target"], dtype=torch.float32)
        return features, target


# 2. В функции main меняем способ загрузки


# ... далее код создания модели и цикла обучения остается прежним ...

def augment_text(text: str, scores: list[float]) -> str:
    """Лёгкая аугментация на основе экстраверсии и добросовестности."""
    if scores[2] > 0.8 and random.random() > 0.6:
        text = text.upper() + "!!!"
    if scores[1] < 0.2 and random.random() > 0.6:
        text = text.lower().replace(".", "").replace(",", "")
    return text


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Взвешенный MSE: сильнее штрафует предсказания вблизи экстремумов (0/1)."""
    weights = 1.0 + torch.abs(target - 0.5) * 2.5
    return (weights * (pred - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Останавливает обучение если R2 не улучшается patience эпох подряд.
    Лучшие веса автоматически восстанавливаются.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_r2 = -float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, r2: float, model: nn.Module) -> bool:
        """Возвращает True если нужно остановить обучение."""
        if r2 > self.best_r2 + self.min_delta:
            self.best_r2 = r2
            self.counter = 0
            # Сохраняем копию весов (без записи на диск)
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Восстанавливает веса лучшей эпохи."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"  [EarlyStopping] Восстановлены веса с лучшим R2={self.best_r2:.4f}")


# ---------------------------------------------------------------------------
# Основная функция обучения
# ---------------------------------------------------------------------------

# 1. Изменяем аргументы train_model
def train_model(train_loader: DataLoader, test_loader: DataLoader, config: Dict[str, Any]) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalityClassifier(input_size=388, num_traits=5).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 0.001),
    )

    # --- ReduceLROnPlateau ---
    # Снижает LR в factor раз если test_loss не улучшается patience эпох.
    # mode="min" потому что отслеживаем loss (хотим уменьшения).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.get("lr_factor", 0.5),
        patience=config.get("lr_patience", 8),
        min_lr=config.get("min_lr", 1e-5),
    )

    # --- Early Stopping ---
    early_stopping = EarlyStopping(
        patience=config.get("es_patience", 15),
        min_delta=config.get("es_min_delta", 1e-4),
    )

    criterion = weighted_mse_loss
    epochs = config.get("epochs", 80)
    traits = ["O", "C", "E", "A", "N"]
    history = {"train_loss": [], "test_loss": [], "r2_scores": [], "lr": []}
    best_r2 = -float("inf")

    print(f"Конфиг: lr={config.get('lr', 0.001)}, epochs={epochs}, "
          f"es_patience={config.get('es_patience', 15)}, "
          f"lr_patience={config.get('lr_patience', 8)}\n")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        # --- Eval ---
        model.eval()
        test_epoch_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_epoch_loss += criterion(pred, y).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

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

        # Сохраняем лучшую модель на диск
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), ARTIFACTS_DIR / "personality_model_best.pth")
            details = " | ".join([f"{traits[i]}: {r2_ind[i]:.2f}" for i in range(5)])
            print(
                f"Эпоха {epoch:03d} | MAE: {mae:.4f} | DirAcc: {dir_acc:.2%} | "
                f"Loss: {avg_test:.4f} | LR: {cur_lr:.2e} | New Best R2: {r2:.4f} ({details})"
            )
        elif epoch % 10 == 0:
            print(
                f"Эпоха {epoch:03d} | R2: {r2:.4f} | MAE: {mae:.4f} | "
                f"DirAcc: {dir_acc:.2%} | Loss: {avg_test:.4f} | LR: {cur_lr:.2e}"
            )

        # Шаг планировщика — передаём test_loss
        scheduler.step(avg_test)

        # Шаг Early Stopping — передаём R2
        if early_stopping.step(r2, model):
            print(f"\n[EarlyStopping] Остановка на эпохе {epoch}. "
                  f"R2 не улучшался {early_stopping.patience} эпох подряд.")
            early_stopping.restore_best(model)
            break

    plot_training_results(history)

    # Финальный анализ разброса
    print("\n--- АНАЛИЗ РАЗБРОСА (STDEV) ---")
    for i, trait in enumerate(traits):
        pred_std = np.std(preds_np[:, i])
        target_std = np.std(targets_np[:, i])
        print(f"  Черта {trait}: Предсказано STD={pred_std:.3f} | В данных STD={target_std:.3f}")

    return model


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def plot_training_results(history: dict) -> None:
    if not history["train_loss"]:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["test_loss"], label="Test")
    axes[0].set_title("Weighted MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["r2_scores"], color="green")
    axes[1].axhline(max(history["r2_scores"]), color="green", linestyle="--", alpha=0.4)
    axes[1].set_title(f"R² Score  (best: {max(history['r2_scores']):.4f})")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(history["lr"], color="orange")
    axes[2].set_title("Learning Rate (ReduceLROnPlateau)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "training_report.png", dpi=120)
    plt.close()
    print(f"\nГрафик сохранён: {ARTIFACTS_DIR / 'training_report.png'}")


def save_artifacts(model_obj: nn.Module) -> Path:
    target = ARTIFACTS_DIR / "personality_model.pth"
    torch.save(model_obj.state_dict(), target)
    print(f"Веса модели сохранены: {target}")
    return target


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def main() -> None:
    cache_path = "data/train_data_precomputed.pt"

    if not Path(cache_path).exists():
        print(f"❌ Ошибка: Файл {cache_path} не найден!")
        return

    # Загружаем кешированные данные
    full_dataset = PersonalityDataset(cache_path)

    # Разделяем на train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                      generator=torch.Generator().manual_seed(42))

    # Создаем лоадеры здесь и передаем их в функцию


    config = {
        "lr": 0.001,
        "epochs": 120,
        "batch_size": 32,  # На FAIDAI смело ставим 32
        "lr_factor": 0.5,
        "lr_patience": 8,
        "min_lr": 1e-5,
        "es_patience": 15,
        "es_min_delta": 1e-4,
    }
    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size", 32), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.get("batch_size", 32))
    # ПЕРЕДАЕМ ЛОАДЕРЫ НАПРЯМУЮ
    model_obj = train_model(train_loader, test_loader, config)

    if model_obj:
        save_artifacts(model_obj)

    print("\n✅ Обучение на FAIDAI (GTX 750 Ti) завершено.")


if __name__ == "__main__":
    main()
