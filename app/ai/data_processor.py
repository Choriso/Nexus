import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextDataProcessor:
    def __init__(self, model_path="app/ai/models/tfidf_vectorizer.pkl"):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) # Можно добавить русские стоп-слова
        self.vectorizer_path = model_path
        self.vectorizer = self._load_or_create_vectorizer()

    def _load_or_create_vectorizer(self):
        try:
            return joblib.load(self.vectorizer_path)
        except (FileNotFoundError, EOFError):
            print("TF-IDF Vectorizer not found or corrupted. A new one will be created upon fitting.")
            return TfidfVectorizer(max_features=5000, stop_words=list(self.stop_words))

    def preprocess_text(self, text):
        text = text.lower()
        # Оставляем и латиницу, и кириллицу
        text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s?!]', '', text)
        return ' '.join(text.split())

    def fit_vectorizer(self, corpus):
        preprocessed_corpus = [self.preprocess_text(doc) for doc in corpus]
        self.vectorizer.fit(preprocessed_corpus)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"TF-IDF Vectorizer fitted and saved to {self.vectorizer_path}")

    def transform_text(self, text):
        preprocessed_text = self.preprocess_text(text)
        # Если vectorizer еще не обучен, он вернет пустую матрицу или ошибку
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
            print("Vectorizer has not been fitted. Returning empty feature set.")
            return None
        return self.vectorizer.transform([preprocessed_text])

    def transform_corpus(self, corpus):
        preprocessed_corpus = [self.preprocess_text(doc) for doc in corpus]
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.vectorizer.vocabulary_:
            print("Vectorizer has not been fitted. Returning empty feature set.")
            return None
        return self.vectorizer.transform(preprocessed_corpus)

