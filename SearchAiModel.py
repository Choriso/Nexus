import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer


class NewsClassifier(nn.Module):
    def __init__(self, input_size=384, num_classes=20):
        super(NewsClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)  # Было 256 → стало 512
        self.ln1 = nn.LayerNorm(512)  # LayerNorm вместо BatchNorm

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)  # Увеличен Dropout

        self.fc2 = nn.Linear(512, 256)  # Было 128 → стало 256
        self.ln2 = nn.LayerNorm(256)

        self.fc3 = nn.Linear(256, num_classes)  # Финальный слой

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)  # Используем LayerNorm
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)  # Без Softmax (он уже есть в CrossEntropyLoss)
        return x


def Learn():
    # ==== 1. Загружаем датасет ====
    categories = None  # Можно ограничить список тем
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    # ==== 2. Векторизуем текст ====
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_bert_embeddings(texts):
        return np.array([bert_model.encode(text) for text in texts])

    X = get_bert_embeddings(newsgroups.data)

    # ==== 3. Кодируем категории ====
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(newsgroups.target)

    # ==== 4. Разделяем данные ====
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Преобразуем в PyTorch тензоры
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # ==== 5. Создаём модель =

    # ==== 6. Обучаем модель ====
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))

    model = NewsClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    model.to(device)

    epochs = 10
    batch_size = 32

    for epoch in range(epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i + batch_size]
            y_batch = y_train_tensor[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "news_classifier.pth")
    print("✅ Веса модели сохранены в 'news_classifier.pth'")
    print("Обучение завершено!")
    joblib.dump(label_encoder, "label_encoder.pkl")

    # ==== 7. Функция предсказания темы текста ====
    def predict_topic(text, top_n=3):
        """Предсказывает top_n тем текста с вероятностями, используя BERT"""
        # Получаем эмбеддинг текста с помощью BERT
        text_tensor = get_bert_embeddings([text])  # Уже torch.Tensor

        # Преобразуем в тензор PyTorch, если вдруг это NumPy
        if isinstance(text_tensor, np.ndarray):
            text_tensor = torch.tensor(text_tensor, dtype=torch.float32)

        # Делаем предсказание
        model.eval()  # Отключает Dropout во время предсказаний
        with torch.no_grad():
            probabilities = model(text_tensor).squeeze(0).numpy()

        # Находим top_n вероятных тем
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_topics = [label_encoder.inverse_transform([idx])[0] for idx in top_indices]
        top_probs = [probabilities[idx] for idx in top_indices]

        return list(zip(top_topics, top_probs))
