import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Для красивого прогресс-бара

# 1. СИНХРОНИЗАЦИЯ ИНДЕКСОВ (Важно!)
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]
type2idx = {t: i for i, t in enumerate(MBTI_TYPES)}


# Твоя архитектура из core.py (дублируем здесь для независимости скрипта)
class MBTIClassifier(nn.Module):
    def __init__(self, input_size=384, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x): return self.net(x)


class MBTIDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")

    # 1. Загрузка и подготовка данных
    df = pd.read_csv('data/mbti_1.csv')

    # Считаем веса классов для балансировки
    class_counts = df['type'].value_counts()
    # Вес класса = 1 / количество примеров
    weights = {t: 1.0 / class_counts[t] for t in MBTI_TYPES}
    sample_weights = df['type'].map(weights).values

    # 2. Превращаем тексты в эмбеддинги ЗАРАНЕЕ (чтобы не грузить BERT в цикле)
    bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    print("⏳ Encoding dataset (this may take a while)...")
    all_embeddings = bert_model.encode(df['posts'].tolist(), show_progress_bar=True, convert_to_tensor=True)
    all_labels = torch.tensor([type2idx[t] for t in df['type']])

    # 3. Разделение на Train и Val (80/20)
    train_embs, val_embs, train_labels, val_labels = train_test_split(
        all_embeddings.cpu().numpy(),
        all_labels.numpy(),
        test_size=0.2,
        stratify=all_labels.numpy(),  # Чтобы пропорция типов была одинаковой везде
        random_state=42
    )

    # Создаем Sampler для балансировки (чтобы редкие типы мелькали чаще)
    train_sample_weights = pd.Series(train_labels).map({type2idx[t]: weights[t] for t in MBTI_TYPES}).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(train_sample_weights),
        num_samples=len(train_sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        MBTIDataset(torch.tensor(train_embs), torch.tensor(train_labels)),
        batch_size=64,
        sampler=sampler
    )
    val_loader = DataLoader(
        MBTIDataset(torch.tensor(val_embs), torch.tensor(val_labels)),
        batch_size=64
    )

    # 4. Настройка модели
    model = MBTIClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 5. Цикл обучения с Early Stopping
    best_val_acc = 0
    patience = 12
    patience_counter = 0

    print(""
          "Start training loop...")
    for epoch in range(200):
        model.train()
        train_loss = 0
        for batch_embs, batch_labels in train_loader:
            batch_embs, batch_labels = batch_embs.to(device), batch_labels.to(device)

            outputs = model(batch_embs)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for v_embs, v_labels in val_loader:
                v_embs, v_labels = v_embs.to(device), v_labels.to(device)
                outputs = model(v_embs)
                _, predicted = torch.max(outputs.data, 1)
                total += v_labels.size(0)
                correct += (predicted == v_labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch {epoch + 1:03d} | Loss: {train_loss / len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # Early Stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "artifacts/mbti_model.pth")
            patience_counter = 0
            print("🌟 New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}. Best Val Acc: {best_val_acc:.2f}%")
                break


if __name__ == "__main__":
    train()