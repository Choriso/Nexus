import random
from pathlib import Path
import json
from typing import Any, Dict
import torch
import torch.nn as nn
from app.ai_profiler.core import PersonalityClassifier
from app.ai_profiler.core import AIProfiler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "ml" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    with open("data/train_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"text": item["text"], "target": item["scores"]} for item in data]


class PersonalityDataset(Dataset):
    def __init__(self, data, profiler):
        self.data = data
        self.profiler = profiler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_text = item['text']

        # 1. Получаем BERT эмбеддинг (384)
        # Очищаем текст, но для BERT лучше подавать lower, если мы вынесли капс в фичи
        clean_text = self.profiler.clean_text(raw_text).lower()
        emb = self.profiler.bert_model.encode(clean_text)
        emb_tensor = torch.tensor(emb, dtype=torch.float32)

        # 2. Получаем ручные фичи (4) через созданный нами метод
        m_feats = self.profiler.get_manual_features(raw_text)
        m_feats_tensor = torch.tensor(m_feats, dtype=torch.float32)

        # 3. СКЛЕИВАНИЕ (384 + 4 = 388)
        # dim=0, так как это одномерные тензоры в датасете
        combined_input = torch.cat((emb_tensor, m_feats_tensor), dim=0)

        return combined_input, torch.tensor(item['target'], dtype=torch.float32)


def augment_text(text, scores):
    if scores[2] > 0.8 and random.random() > 0.6:
        text = text.upper() + "!!!"
    if scores[1] < 0.2 and random.random() > 0.6:
        text = text.lower().replace(".", "").replace(",", "")
    return text

def weighted_mse_loss(pred, target):
    weights = 1.0 + torch.abs(target - 0.5) * 2.5
    return (weights * (pred - target)**2).mean()


def train_model(dataset_raw: Any, config: Dict[str, Any]) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiler = AIProfiler()

    train_data, test_data = train_test_split(dataset_raw, test_size=0.2, random_state=42)

    # Теперь даталоадер будет выдавать тензоры размером 388
    train_loader = DataLoader(PersonalityDataset(train_data, profiler), batch_size=16, shuffle=True)
    test_loader = DataLoader(PersonalityDataset(test_data, profiler), batch_size=16, shuffle=False)

    # ВАЖНО: меняем input_size на 388
    model = PersonalityClassifier(input_size=388, num_traits=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    criterion = weighted_mse_loss

    history = {"train_loss": [], "test_loss": [], "r2_scores": []}
    best_r2 = -float('inf')
    epochs = config.get("epochs", 80)

    print(f"🚀 Старт: {len(train_data)} обучающих примеров.")

    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        model.eval()
        test_epoch_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                test_epoch_loss += criterion(pred, y).item()
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds_stacked = np.vstack(all_preds)
        targets_stacked = np.vstack(all_targets)

        avg_train = train_epoch_loss / len(train_loader)
        avg_test = test_epoch_loss / len(test_loader)
        r2 = r2_score(targets_stacked, preds_stacked)
        r2_individual = r2_score(targets_stacked, preds_stacked, multioutput='raw_values')
        mae = mean_absolute_error(targets_stacked, preds_stacked)
        # Направление (правильно ли определили сторону 0.5)
        dir_acc = np.mean((preds_stacked > 0.5) == (targets_stacked > 0.5))

        history["train_loss"].append(avg_train)
        history["test_loss"].append(avg_test)
        history["r2_scores"].append(r2)

        traits = ['O', 'C', 'E', 'A', 'N']
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), ARTIFACTS_DIR / "personality_model_best.pth")
            details = " | ".join([f"{traits[i]}: {r2_individual[i]:.2f}" for i in range(5)])
            print(f"Эпоха {epoch:03d} | MAE: {mae:.4f} | DirAcc: {dir_acc:.2%} | Loss: {avg_test:.4f} | 🌟 New Best R2: {r2:.4f} ({details})")
        elif epoch % 10 == 0:
            print(f"Эпоха {epoch:03d} | R2: {r2:.4f} | MAE: {mae:.4f} | DirAcc: {dir_acc:.2%} | Loss: {avg_test:.4f}")

    plot_training_results(history)

    # Считаем разброс в конце
    all_preds_flat = np.vstack(all_preds)
    all_targets_flat = np.vstack(all_targets)
    print("\n--- АНАЛИЗ РАЗБРОСА (STDEV) ---")
    for i, trait in enumerate(traits):
        pred_std = np.std(all_preds_flat[:, i])
        target_std = np.std(all_targets_flat[:, i])
        print(f"Черта {trait}: Предсказано STD={pred_std:.3f} | В данных STD={target_std:.3f}")

    return model  # КРИТИЧЕСКИ ВАЖНО: возвращаем модель!

def plot_training_results(history):
    if not history["train_loss"]: return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["test_loss"], label="Test")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["r2_scores"], color='green', label="R2 Accuracy")
    plt.title(f"Max R2: {max(history['r2_scores']):.4f}")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "training_report.png")
    plt.close()

def save_artifacts(model_obj: nn.Module) -> Path:
    target = ARTIFACTS_DIR / "personality_model.pth"
    torch.save(model_obj.state_dict(), target)
    print(f"✅ Веса модели сохранены в: {target}")
    return target

def main() -> None:
    dataset = load_dataset()
    config = {"lr": 0.001, "epochs": 80}
    model_obj = train_model(dataset, config)
    if model_obj:
        save_artifacts(model_obj)
    print("✅ Все процессы завершены.")

if __name__ == "__main__":
    main()
