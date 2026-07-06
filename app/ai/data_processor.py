import re

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from config import config

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextDataProcessor:
    """
    Класс для предобработки текстовых данных и работы с TF-IDF векторизацией.
    
    Позволяет очищать текст, обучать и применять TF-IDF vectorizer к отдельным сообщениям и корпусу, 
    а также сохранять и загружать модель векторизатора.

    Атрибуты:
        lemmatizer (WordNetLemmatizer): Лемматизатор WordNet для английского языка.
        stop_words (set[str]): Множество стоп-слов (по умолчанию — английские).
        vectorizer_path (str): Путь к сериализованному файлу TF-IDF vectorizer.
        vectorizer (TfidfVectorizer): Векторизатор для преобразования текста в числовой формат.
    """
    def __init__(self, model_path: str | None = None):
        """
        Инициализация TextDataProcessor с лемматизатором, списком стоп-слов и загрузкой/созданием TF-IDF vectorizer.
        
        Args:
            model_path (str | None): Путь к файлу сериализованного векторизатора.
                По умолчанию берётся из config.TFIDF_VECTORIZER_PATH.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer_path = model_path or config.TFIDF_VECTORIZER_PATH
        self.vectorizer = self._load_or_create_vectorizer()

    def _load_or_create_vectorizer(self) -> TfidfVectorizer:
        """
        Загружает сериализованный TF-IDF vectorizer или создает новый, если файл отсутствует.
        
        Returns:
            TfidfVectorizer: Загруженный или вновь созданный объект TF-IDF Vectorizer.
        """
        try:
            return joblib.load(self.vectorizer_path)
        except (FileNotFoundError, EOFError):
            print("TF-IDF Vectorizer не найден или поврежден. Будет создан новый при обучении.")
            return TfidfVectorizer(max_features=5000, stop_words=list(self.stop_words))

    def preprocess_text(self, text: str) -> str:
        """
        Преобразует строку в нижний регистр, удаляет лишние символы (оставляет латиницу и кириллицу),
        нормализует пробелы.
        
        Args:
            text (str): Исходный текст для очистки.
        
        Returns:
            str: Предобработанный текст.
        """
        text = text.lower()
        text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s?!]', '', text)
        return ' '.join(text.split())

    def fit_vectorizer(self, corpus: list[str]) -> None:
        """
        Обучает TF-IDF vectorizer на тексте корпуса и сохраняет сериализованную модель.

        Args:
            corpus (list[str]): Список документов для обучения.
        
        Returns:
            None
        """
        preprocessed_corpus = [self.preprocess_text(doc) for doc in corpus]
        self.vectorizer.fit(preprocessed_corpus)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"TF-IDF Vectorizer обучен и сохранён в {self.vectorizer_path}")

    def transform_text(self, text: str):
        """
        Преобразует отдельный текстовый документ в TF-IDF вектор.

        Args:
            text (str): Строка документа для преобразования.

        Returns:
            scipy.sparse.csr_matrix | None: TF-IDF векторизованный вид текста либо None, если vectorizer не обучен.
        """
        preprocessed_text = self.preprocess_text(text)
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
            print("Vectorizer не был обучен. Возвращается пустой набор признаков.")
            return None
        return self.vectorizer.transform([preprocessed_text])

    def transform_corpus(self, corpus: list[str]):
        """
        Преобразует список текстовых документов в матрицу признаков TF-IDF.

        Args:
            corpus (list[str]): Список документов для преобразования.
        
        Returns:
            scipy.sparse.csr_matrix | None: Матрица TF-IDF либо None, если vectorizer не обучен.
        """
        preprocessed_corpus = [self.preprocess_text(doc) for doc in corpus]
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
            print("Vectorizer не был обучен. Возвращается пустой набор признаков.")
            return None
        return self.vectorizer.transform(preprocessed_corpus)
