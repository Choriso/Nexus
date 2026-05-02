import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from pathlib import Path
import json

# Импортируем обновленные компоненты
from core import PersonalityClassifier, scores_to_bins, NUM_BINS, bins_to_score

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

class PersonalityDataset(Dataset):
    def __init__(self, data_path: str):
        # Если файл .pt существует — грузим его, иначе это подсказка, что нужен препроцессинг
        if data_path.endswith('.json'):
             with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
             self.data = raw_data
             print(f"Загружено {len(self.data)} сырых примеров из JSON.")
        else:
            self.data = torch.load(data_path)
            print(f"Загружено {len(self.data)} тензорных примеров.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item["features"], dtype=torch.float32)
        target_raw = np.array(item["target"], dtype=np.float32)

        # Для ординальной классификации нам нужны индексы бинов
        target_bins = torch.tensor(scores_to_bins(target_raw), dtype=torch.long)
        weight = item.get("weight", 1.0)

        return features, target_bins, weight, torch.tensor(target_raw)

def ordinal_loss_with_std(logits, target_bins, sample_weights, beta=0.1):
    """
    beta уменьшена до 0.1, так как данные 'неплохие' и нам не нужно
    сильно форсировать разброс искусственно.
    """
    B, T, K = logits.shape
    device = logits.device

    # 1. Cumulative BCE logic
    # Создаем маску: для бина 3 это [1, 1, 1, 0, 0...] (все что меньше или равно)
    range_tensor = torch.arange(K - 1, device=device).view(1, 1, -1)
    cum_targets = (target_bins.unsqueeze(-1) > range_tensor).float()

    # Предсказываем кумулятивные вероятности (через сигмоиду для каждого порога)
    # В этой версии мы используем лог-сумму для стабильности
    probs = F.softmax(logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)[:, :, :-1]
    cum_probs = cum_probs.clamp(1e-6, 1.0 - 1e-6)

    # Основной лосс
    bce_loss = F.binary_cross_entropy(cum_probs, cum_targets, reduction='none')
    main_loss = (bce_loss.mean(dim=(1, 2)) * sample_weights).mean()

    # 2. STD Penalty (опционально для поддержания живости предсказаний)
    pred_scores = bins_to_score(logits)
    batch_std = pred_scores.std(dim=0).mean()
    std_penalty = torch.relu(0.15 - batch_std) # Целевое отклонение 0.15

    return main_loss + beta * std_penalty, batch_std.item()

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4):
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

def train_model(train_loader, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PersonalityClassifier(input_size=388).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config["T_0"], T_mult=2, eta_min=1e-6
    )

    early_stopping = EarlyStopping(patience=config["es_patience"])
    history = {"train_loss": [], "test_loss": [], "r2_scores": [], "lr": []}

    for epoch in range(config["epochs"]):
        model.train()
        train_epoch_loss = 0.0

        for x, target_bins, weights, _ in train_loader:
            x, target_bins, weights = x.to(device), target_bins.to(device), weights.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss, _ = ordinal_loss_with_std(logits, target_bins, weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_epoch_loss += loss.item()

        scheduler.step()

        # Evaluation
        model.eval()
        all_preds, all_targets = [], []
        test_epoch_loss = 0.0
        with torch.no_grad():
            for x, target_bins, weights, orig_targets in test_loader:
                x, target_bins, weights = x.to(device), target_bins.to(device), weights.to(device)
                logits = model(x)
                loss, _ = ordinal_loss_with_std(logits, target_bins, weights)
                test_epoch_loss += loss.item()

                preds = bins_to_score(logits)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(orig_targets.numpy())

        preds_np = np.vstack(all_preds)
        targets_np = np.vstack(all_targets)
        r2 = r2_score(targets_np, preds_np)

        history["train_loss"].append(train_epoch_loss / len(train_loader))
        history["test_loss"].append(test_epoch_loss / len(test_loader))
        history["r2_scores"].append(r2)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {history['test_loss'][-1]:.4f} | R2: {r2:.4f}")

        if early_stopping.step(r2, model):
            print(f"Early stopping at epoch {epoch}")
            early_stopping.restore_best(model)
            break

    return model, history

def main():
    # Путь к данным относительно файла
    data_path = "data/train_data_precomputed.pt"
    # Если ты хочешь использовать JSON напрямую, поменяй путь здесь

    if not Path(data_path).exists():
        print(f"Файл {data_path} не найден. Проверь наличие сгенерированных тензоров.")
        return

    ds = PersonalityDataset(data_path)
    train_size = int(0.85 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

    config = {
        "lr": 1e-3,          # Стандартный LR для AdamW
        "epochs": 150,
        "batch_size": 64,    # Чуть увеличил батч для стабильности градиента
        "T_0": 10,
        "es_patience": 25,
    }

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"])

    model, history = train_model(train_loader, test_loader, config)

    # Сохранение
    save_path = ARTIFACTS_DIR / "personality_model_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}")

if __name__ == "__main__":
    main()