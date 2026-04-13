import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from SearchAiModel import NewsClassifier  # Импортируем модель
import joblib  # Для загрузки label_encoder

# Загружаем LabelEncoder
label_encoder = joblib.load("label_encoder.pkl")

# Загружаем BERT-модель для эмбеддингов
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Загружаем модель классификации
model = NewsClassifier(input_size=384, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("news_classifier.pth"))
model.eval()  # Переключаем в режим предсказания

import re


def clean_text(text):
    if not text:
        return ""
    # 1. Приводим к нижнему регистру (чтобы "ПРИВЕТ" и "привет" были одним и тем же)
    text = text.lower()
    # 2. Удаляем ссылки (они только путают модель)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 3. Удаляем лишние спецсимволы, оставляя буквы, цифры и базовую пунктуацию
    text = re.sub(r'[^а-яа-еёz-a0-9?!.,\s]', '', text)
    # 4. Схлопываем множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_bert_embeddings(texts):
    """Очищает тексты и создаёт BERT-эмбеддинги"""
    # Сначала чистим каждый текст в списке
    cleaned_texts = [clean_text(t) for t in texts]

    # Теперь кодируем. Важно: encode в SentenceTransformer
    # умеет работать со списком сразу, это быстрее, чем цикл!
    return np.array(bert_model.encode(cleaned_texts))


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