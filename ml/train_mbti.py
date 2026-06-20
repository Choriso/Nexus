import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]
type2idx = {t: i for i, t in enumerate(MBTI_TYPES)}


class MBTIClassifier(nn.Module):
    """
    Классификатор личностных типов MBTI на основе нейронной сети.

    Args:
        input_size (int): Размерность входного вектора (по умолчанию 384).
        num_classes (int): Количество классов (типов MBTI, по умолчанию 16).

    Attributes:
        net (nn.Sequential): Последовательная модель нейронной сети.

    Пример вызова:
        model = MBTIClassifier()
        logits = model(tensor)
    """
    def __init__(self, input_size: int = 384, num_classes: int = 16) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямое распространение входа через сеть.

        Args:
            x (torch.Tensor): Входной батч признаков (размера [batch_size, input_size])

        Returns:
            torch.Tensor: Логиты классов (размера [batch_size, num_classes])
        """
        return self.net(x)


class MBTIDataset(Dataset):
    """
    Dataset для хранения эмбеддингов и соответствующих меток MBTI.

    Args:
        embeddings (torch.Tensor): Матрица эмбеддингов (N x D).
        labels (torch.Tensor): Список меток (N,).

    Attributes:
        embeddings (torch.Tensor): Эмбеддинги.
        labels (torch.Tensor): Метки-классы.
    """
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        """
        Возвращает количество объектов в датасете.

        Returns:
            int: Количество примеров.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает эмбеддинг и метку по индексу.

        Args:
            idx (int): Индекс элемента.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (эмбеддинг, метка)
        """
        return self.embeddings[idx], self.labels[idx]


def train() -> None:
    """
    Запускает процесс обучения классификатора MBTI.

    Этапы:
        1. Загружает данные из файла.
        2. Вычисляет веса классов для балансировки.
        3. Получает эмбеддинги текстов с помощью SBERT.
        4. Делит данные на обучающую и валидационную выборки (80/20).
        5. Инициализирует модель, оптимизатор и функцию потерь.
        6. Обучает модель, используя early stopping.
        7. Сохраняет лучшую модель по валидационной точности.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    df = pd.read_csv('data/mbti_1.csv')
    class_counts = df['type'].value_counts()
    weights = {t: 1.0 / class_counts[t] for t in MBTI_TYPES}

    bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    print("Вычисление эмбеддингов...")
    all_embeddings = bert_model.encode(df['posts'].tolist(), show_progress_bar=True, convert_to_tensor=True)
    all_labels = torch.tensor([type2idx[t] for t in df['type']])

    train_embs, val_embs, train_labels, val_labels = train_test_split(
        all_embeddings.cpu().numpy(),
        all_labels.numpy(),
        test_size=0.2,
        stratify=all_labels.numpy(),
        random_state=42
    )

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

    model = MBTIClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 12
    patience_counter = 0

    print("Начало обучения...")
    for epoch in range(200):
        model.train()
        train_loss = 0.0
        for batch_embs, batch_labels in train_loader:
            batch_embs, batch_labels = batch_embs.to(device), batch_labels.to(device)
            outputs = model(batch_embs)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "artifacts/mbti_model.pth")
            patience_counter = 0
            print("Модель сохранена (новый лучший результат)!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch + 1}. Лучшая валидация: {best_val_acc:.2f}%")
                break


if __name__ == "__main__":
    train()