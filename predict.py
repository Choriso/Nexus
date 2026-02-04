import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from Main import NewsClassifier  # Импортируем модель
import joblib  # Для загрузки label_encoder

# Загружаем LabelEncoder
label_encoder = joblib.load("label_encoder.pkl")

# Загружаем BERT-модель для эмбеддингов
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Загружаем модель классификации
model = NewsClassifier(input_size=384, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("news_classifier.pth"))
model.eval()  # Переключаем в режим предсказания


def get_bert_embeddings(texts):
    """Создаёт BERT-эмбеддинги"""
    return np.array([bert_model.encode(text) for text in texts])


def predict_topic(text, top_n=3):
    """Принимает текст и возвращает top_n тем с вероятностями"""
    text_vectorized = get_bert_embeddings([text])  # Получаем BERT-эмбеддинг
    text_tensor = torch.tensor(text_vectorized, dtype=torch.float32)

    with torch.no_grad():
        probabilities = model(text_tensor).squeeze(0).numpy()

    class_labels = label_encoder.classes_
    print(class_labels)
    # Получаем индексы top_n предсказаний
    top_indices = np.argsort(probabilities)[-top_n:][::-1]

    # Получаем названия тем по индексам
    top_topics = [class_labels[idx] for idx in top_indices]
    top_probs = [probabilities[idx] for idx in top_indices]

    return list(zip(top_topics, top_probs))  # [(тема, вероятность), ...]