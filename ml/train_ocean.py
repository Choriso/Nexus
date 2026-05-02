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


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------

class PersonalityDataset(Dataset):
    """
    Поддерживает два формата:
      1. .pt — предобработанные тензоры с полями: features, target, weight
      2. .json — сырые данные с полями: features, target, weight
         ИЛИ (новый формат): text, scores — в этом случае embeddings
         должны быть заранее вычислены внешним скриптом и переданы через
         поле "features".

    ВАЖНО: JSON-файл должен содержать уже вычисленные эмбеддинги в поле
    "features" (список float длиной 388). Поле "scores" — целевые OCEAN [0,1].
    Если хочешь подавать сырой текст — используй скрипт preprocess.py.
    """

    def __init__(self, data_path: str):
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            self.data = raw_data
            # Определяем формат: новый (text+scores) или старый (features+target)
            sample = self.data[0]
            if "features" in sample:
                self.format = "features"
            elif "text" in sample and "scores" in sample:
                # Нет features — нужна предобработка
                raise ValueError(
                    "JSON содержит 'text'+'scores' без 'features'. "
                    "Запусти preprocess.py сначала, чтобы вычислить эмбеддинги."
                )
            else:
                raise ValueError(f"Неизвестный формат JSON. Ключи: {list(sample.keys())}")
            print(f"Загружено {len(self.data)} примеров из JSON (формат: {self.format}).")
        else:
            self.data = torch.load(data_path)
            self.format = "pt"
            print(f"Загружено {len(self.data)} тензорных примеров.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        features = torch.tensor(item["features"], dtype=torch.float32)

        # Поддержка обоих ключей: "target" или "scores"
        raw_scores = item.get("target", item.get("scores"))
        target_raw = np.array(raw_scores, dtype=np.float32)

        target_bins = torch.tensor(scores_to_bins(target_raw), dtype=torch.long)
        weight = float(item.get("weight", 1.0))

        return features, target_bins, weight, torch.tensor(target_raw)


# ---------------------------------------------------------------
# Лосс-функция
# ---------------------------------------------------------------

def ordinal_loss(logits, target_bins, sample_weights):
    """
    Ординальный BCE-лосс без штрафа на std.

    Почему убрали std_penalty:
      - std_penalty форсировал разброс предсказаний НЕЗАВИСИМО от ошибки,
        что приводило к случайным предсказаниям и отрицательному R².
      - На малом датасете (1.7k) модель не может одновременно
        минимизировать BCE и поддерживать искусственный std.

    Вместо этого добавлен label_smoothing: он предотвращает
    вырождение в константу через смягчение таргетов.
    """
    B, T, K = logits.shape
    device = logits.device

    # Кумулятивные таргеты: для бина b маска [1,1,...,1,0,0,...,0]
    range_tensor = torch.arange(K - 1, device=device).view(1, 1, -1)
    cum_targets = (target_bins.unsqueeze(-1) > range_tensor).float()

    # Label smoothing: смягчаем таргеты от {0,1} к {eps, 1-eps}
    smoothing = 0.05
    cum_targets = cum_targets * (1 - smoothing) + (1 - cum_targets) * smoothing

    # Кумулятивные вероятности через softmax + cumsum
    probs = F.softmax(logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)[:, :, :-1].clamp(1e-6, 1.0 - 1e-6)

    # BCE по всем порогам
    bce = F.binary_cross_entropy(cum_probs, cum_targets, reduction='none')

    # Взвешенный mean по батчу
    sample_weights = sample_weights.to(device)
    loss = (bce.mean(dim=(1, 2)) * sample_weights).mean()

    return loss


# ---------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------

class EarlyStopping:
    """
    Останавливает обучение при отсутствии улучшений R² за patience эпох.
    Сохраняет лучшие веса модели.
    """
    def __init__(self, patience=30, min_delta=5e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_r2 = -float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, current_r2, model):
        if current_r2 > self.best_r2 + self.min_delta:
            self.best_r2 = current_r2
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
            print(f"Восстановлена лучшая модель с R²={self.best_r2:.4f}")


# ---------------------------------------------------------------
# Цикл обучения
# ---------------------------------------------------------------

def train_model(train_loader, val_loader, config):
    """
    Изменения относительно исходной версии:
      1. LR снижен: 3e-4 вместо 1e-3 — на маленьких данных нужен аккуратный шаг
      2. weight_decay снижен: 1e-5 вместо 1e-4 — меньше L2-давления
      3. Scheduler: ReduceLROnPlateau вместо CosineAnnealingWarmRestarts.
         Рестарты косинусного планировщика дестабилизируют обучение
         (резкий прыжок LR убивал накопленные знания).
      4. R² считается правильно: sklearn.r2_score по матрице (N, 5)
      5. MAE добавлен как вторичная метрика для мониторинга
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    model = PersonalityClassifier(input_size=388, num_bins=NUM_BINS).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # ReduceLROnPlateau: уменьшает LR, если val_loss не улучшается
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',       # следим за R² (больше = лучше)
        factor=0.5,       # LR * 0.5 при плато
        patience=10,      # 10 эпох без улучшения → снижаем LR
        min_lr=1e-6,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        patience=config["es_patience"],
        min_delta=5e-4,
    )

    history = {
        "train_loss": [], "val_loss": [],
        "r2_scores": [], "mae_scores": [], "lr": [],
    }

    for epoch in range(config["epochs"]):
        # --- Train ---
        model.train()
        train_loss_accum = 0.0

        for x, target_bins, weights, _ in train_loader:
            x = x.to(device)
            target_bins = target_bins.to(device)
            weights = torch.tensor(weights, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = ordinal_loss(logits, target_bins, weights)
            loss.backward()

            # Gradient clipping — защита от взрывных градиентов
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_accum += loss.item()

        # --- Validation ---
        model.eval()
        all_preds, all_targets = [], []
        val_loss_accum = 0.0

        with torch.no_grad():
            for x, target_bins, weights, orig_targets in val_loader:
                x = x.to(device)
                target_bins = target_bins.to(device)
                weights = torch.tensor(weights, dtype=torch.float32).to(device)

                logits = model(x)
                loss = ordinal_loss(logits, target_bins, weights)
                val_loss_accum += loss.item()

                preds = bins_to_score(logits)  # (B, 5)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(orig_targets.numpy())

        preds_np = np.vstack(all_preds)    # (N, 5)
        targets_np = np.vstack(all_targets)  # (N, 5)

        # R² считается по всей матрице: sklearn усредняет по трейтам
        r2 = r2_score(targets_np, preds_np, multioutput='uniform_average')
        mae = mean_absolute_error(targets_np, preds_np)

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["r2_scores"].append(r2)
        history["mae_scores"].append(mae)
        history["lr"].append(current_lr)

        # Планировщик следит за R²
        scheduler.step(r2)

        if epoch % 5 == 0 or epoch < 5:
            print(
                f"Epoch {epoch:03d} | "
                f"TrainLoss: {avg_train_loss:.4f} | "
                f"ValLoss: {avg_val_loss:.4f} | "
                f"R²: {r2:.4f} | "
                f"MAE: {mae:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        if early_stopping.step(r2, model):
            print(f"\nEarly stopping на эпохе {epoch}. Лучший R²: {early_stopping.best_r2:.4f}")
            break

    early_stopping.restore_best(model)
    return model, history


# ---------------------------------------------------------------
# Построение графиков
# ---------------------------------------------------------------

def plot_history(history, save_dir: Path):
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


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def main():
    # --- Выбор данных ---
    # Приоритет: предобработанные тензоры (.pt) → JSON с features
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

    # Разбивка 85/15 с фиксированным seed для воспроизводимости
    train_size = int(0.85 * n)
    val_size = n - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_size, val_size], generator=generator
    )

    config = {
        # LR снижен: на 1.7k примерах 1e-3 слишком агрессивен
        "lr": 3e-4,
        # weight_decay снижен: меньше L2-регуляризации, больше свободы
        "weight_decay": 1e-5,
        "epochs": 200,
        # Батч 32 — компромисс между стабильностью градиента и частотой обновлений
        "batch_size": 32,
        # patience увеличен: ReduceLROnPlateau уже адаптирует LR,
        # ES нужен только для окончательной остановки
        "es_patience": 40,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,   # отбрасываем неполный батч — стабилизирует BatchNorm/LN
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    print(f"\nКонфигурация: {config}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    model, history = train_model(train_loader, val_loader, config)

    # Сохранение модели
    save_path = ARTIFACTS_DIR / "personality_model_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nМодель сохранена: {save_path}")

    # Финальные метрики
    best_r2 = max(history["r2_scores"])
    best_epoch = history["r2_scores"].index(best_r2)
    best_mae = history["mae_scores"][best_epoch]
    print(f"Лучший R²: {best_r2:.4f} (epoch {best_epoch}), MAE: {best_mae:.4f}")

    # График
    plot_history(history, ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
