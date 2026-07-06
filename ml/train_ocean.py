import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from pathlib import Path
import json

from app.ai_profiler.core import PersonalityClassifier, scores_to_bins, NUM_BINS, bins_to_score

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

class PersonalityDataset(Dataset):
    """
    Класс датасета для задачи OCEAN-предсказания.

    Поддерживаются два формата данных:
    1. .pt — тензоры torch с полями: features, target, weight.
    2. .json — список словарей с полями: features, target, weight,
       либо (новый формат) text и scores (в этом случае embeddings
       должны быть доступны в поле 'features').

    Args:
        data_path (str): Путь к файлу данных (.pt или .json).

    Raises:
        ValueError: При отсутствии необходимых полей в json.

    Attributes:
        data (list[dict] или list): Сырые или преобразованные данные.
        format (str): Формат данных ('features', 'pt').
    """

    def __init__(self, data_path: str):
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            self.data = raw_data
            sample = self.data[0]
            if "features" in sample:
                self.format = "features"
            elif "text" in sample and "scores" in sample:
                raise ValueError(
                    "JSON содержит 'text' и 'scores', но отсутствует 'features'. "
                    "Выполните скрипт preprocess.py для вычисления эмбеддингов."
                )
            else:
                raise ValueError(f"Неизвестный формат JSON. Найдены ключи: {list(sample.keys())}")
            print(f"Загружено {len(self.data)} примеров из JSON (формат: {self.format}).")
        else:
            self.data = torch.load(data_path)
            self.format = "pt"
            print(f"Загружено {len(self.data)} тензорных примеров.")

    def __len__(self) -> int:
        """Возвращает количество примеров в датасете.

        Returns:
            int: Размер датасета.
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """Достает пример из датасета по индексу.

        Args:
            idx (int): Индекс примера.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
                features: Тензор признаков (shape: [388]).
                target_bins: Целевые бины (long tensor).
                weight: Вес примера (float).
                target_raw: Оригинальные значения (float tensor).
        """
        item = self.data[idx]
        features = torch.tensor(item["features"], dtype=torch.float32)
        raw_scores = item.get("target", item.get("scores"))
        target_raw = np.array(raw_scores, dtype=np.float32)
        target_bins = torch.tensor(scores_to_bins(target_raw), dtype=torch.long)
        weight = float(item.get("weight", 1.0))
        return features, target_bins, weight, torch.tensor(target_raw)

def ordinal_loss(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    sample_weights: torch.Tensor
    ) -> torch.Tensor:
    """
    Ординальный (ordinal) лосс для задачи градуированного предсказания.

    Одинаково интерпретирует модельные предсказания и таргеты: P(Y <= k),
    где k — границы бинов по каждому признаку.

    Args:
        logits (torch.Tensor): Модельные логицы, shape (B, T, K).
        target_bins (torch.Tensor): Бины-таргеты, shape (B, T).
        sample_weights (torch.Tensor): Веса примеров, shape (B,).

    Returns:
        torch.Tensor: Значение лосса (скаляр).
    """
    B, T, K = logits.shape
    device = logits.device
    range_tensor = torch.arange(K - 1, device=device).view(1, 1, -1)
    cum_targets = (target_bins.unsqueeze(-1) <= range_tensor).float()
    smoothing = 0.02
    cum_targets = cum_targets * (1 - smoothing) + (1 - cum_targets) * smoothing
    probs = F.softmax(logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)[:, :, :-1].clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(cum_probs, cum_targets, reduction='none')
    sample_weights = sample_weights.to(device)
    loss = (bce.mean(dim=(1, 2)) * sample_weights).mean()
    return loss

class EarlyStopping:
    """
    Реализация ранней остановки по метрике R². Сохраняет наилучшую модель.

    Args:
        patience (int): Количество эпох ожидания без улучшения.
        min_delta (float): Минимальное приращение для улучшения лучшего результата.

    Attributes:
        patience (int): Порог терпимости.
        min_delta (float): Шаг улучшения.
        best_r2 (float): Лучшее достигнутое значение R².
        counter (int): Счетчик эпох без улучшения.
        best_state (dict или None): Снэпшот state_dict модели.
    """

    def __init__(self, patience: int = 30, min_delta: float = 5e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_r2 = -float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, current_r2: float, model: nn.Module) -> bool:
        """
        Проверяет и обновляет состояние ранней остановки.

        Args:
            current_r2 (float): Текущее значение метрики R².
            model (nn.Module): Обучаемая модель.

        Returns:
            bool: True если пора прервать обучение, иначе False.
        """
        if current_r2 > self.best_r2 + self.min_delta:
            self.best_r2 = current_r2
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """
        Восстанавливает веса лучшей модели по метрике.

        Args:
            model (nn.Module): Модель.
        """
        if self.best_state:
            model.load_state_dict(self.best_state)
            print(f"Восстановлена лучшая модель с R²={self.best_r2:.4f}")

def train_model(train_loader: DataLoader, val_loader: DataLoader, config: dict):
    """
    Главный цикл обучения. Осуществляет тренировку, валидацию, расчет метрик и лоссов.

    Args:
        train_loader (DataLoader): Даталоадер для обучающих данных.
        val_loader (DataLoader): Даталоадер для валидационных данных.
        config (dict): Конфиг гиперпараметров.

    Returns:
        Tuple[nn.Module, dict]: Обученная модель и история обучения.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalityClassifier(input_size=388, num_bins=NUM_BINS).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=12
    )
    early_stopping = EarlyStopping(patience=config["es_patience"])
    history = {"train_loss": [], "val_loss": [], "r2_scores": [], "mae_scores": [], "lr": []}
    for epoch in range(config["epochs"]):
        model.train()
        train_loss_accum = 0.0
        for x, target_bins, weights, _ in train_loader:
            x, target_bins = x.to(device), target_bins.to(device)
            weights = weights.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            logits = model(x)
            loss = ordinal_loss(logits, target_bins, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_accum += loss.item()
        model.eval()
        all_preds, all_targets = [], []
        val_loss_accum = 0.0
        with torch.no_grad():
            for x, target_bins, weights, orig_targets in val_loader:
                x, target_bins = x.to(device), target_bins.to(device)
                weights = weights.to(device, dtype=torch.float32)
                logits = model(x)
                val_loss_accum += ordinal_loss(logits, target_bins, weights).item()
                all_preds.append(bins_to_score(logits).cpu().numpy())
                all_targets.append(orig_targets.numpy())
        preds_np, targets_np = np.vstack(all_preds), np.vstack(all_targets)
        r2 = r2_score(targets_np, preds_np, multioutput='uniform_average')
        mae = mean_absolute_error(targets_np, preds_np)
        avg_train, avg_val = train_loss_accum / len(train_loader), val_loss_accum / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["r2_scores"].append(r2)
        history["mae_scores"].append(mae)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step(r2)
        if epoch % 5 == 0:
            print(f"E{epoch:03d} | L:{avg_val:.4f} | R2:{r2:.4f} | MAE:{mae:.4f}")
        if early_stopping.step(r2, model):
            break
    early_stopping.restore_best(model)
    return model, history

def plot_history(history: dict, save_dir: Path) -> None:
    """
    Визуализирует историю лоссов и метрик обучения.

    Args:
        history (dict): История обучения с ключами "train_loss", "val_loss", "r2_scores", "mae_scores", "lr".
        save_dir (Path): Путь к директории для сохранения графика.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history["r2_scores"], color="green", label="R²")
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("R² Score")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(history["lr"], color="orange", label="LR")
    axes[2].set_title("Learning Rate")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    save_path = save_dir / "training_history.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"График сохранён: {save_path}")

def main() -> None:
    """
    Главная точка входа. Готовит данные, запускает обучение, сохраняет модель и строит график.

    Returns:
        None
    """
    pt_path = Path("data/train_data_precomputed.pt")
    json_path = Path("data/generated_data_ocean.json")

    if pt_path.exists():
        data_path = str(pt_path)
    elif json_path.exists():
        data_path = str(json_path)
    else:
        print("Не найдены данные. Ожидается один из файлов:")
        print(f"  {pt_path}  (предобработанные тензоры)")
        print(f"  {json_path} (JSON с полем 'features')")
        return

    ds = PersonalityDataset(data_path)
    n = len(ds)
    print(f"Всего примеров: {n}")

    train_size = int(0.85 * n)
    val_size = n - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size], generator=generator
    )

    config = {
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "epochs": 200,
        "batch_size": 32,
        "es_patience": 40,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    print(f"\nКонфигурация: {config}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    model, history = train_model(train_loader, val_loader, config)
    save_path = ARTIFACTS_DIR / "personality_model_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nМодель сохранена: {save_path}")

    best_r2 = max(history["r2_scores"])
    best_epoch = history["r2_scores"].index(best_r2)
    best_mae = history["mae_scores"][best_epoch]
    print(f"Лучший R²: {best_r2:.4f} (epoch {best_epoch}), MAE: {best_mae:.4f}")

    plot_history(history, ARTIFACTS_DIR)

if __name__ == "__main__":
    main()
