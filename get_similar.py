from concurrent.futures import ThreadPoolExecutor
import re
import gensim.downloader as api
import pymorphy3
import numpy as np
from functools import lru_cache
from spellchecker import SpellChecker

spell = SpellChecker(language='ru')
morph = pymorphy3.MorphAnalyzer()
model = api.load("word2vec-ruscorpora-300")
NO_WORDS = ['и', 'в', 'на', 'не', 'с', 'по', 'к', 'а', 'но', 'или']


def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_words = [word for word in cleaned_text.split() if word.lower() not in NO_WORDS]
    return " ".join(cleaned_words)


@lru_cache(maxsize=None)
def correct_text(text):
    """
    Корректирует текст с помощью проверки орфографии каждого слова во входном тексте.

    Параметры:
        text (str): Входной текст для коррекции.

    Возвращает:
        str: Исправленный текст после проверки орфографии.
    """
    corrected_text = [spell.correction(i) for i in text.split()]
    try:
        return ' '.join(corrected_text)
    except TypeError:
        return text


@lru_cache(maxsize=None)
def get_part_of_speech(word):
    """
    Получает часть речи слова с помощью pymorphy3 для указанного слова.

    Параметры:
        word (str): Слово, для которого необходимо получить часть речи.

    Возвращает:
        str: Часть речи указанного слова.
    """
    parsed_word = morph.parse(word)[0]
    return parsed_word.tag.POS


@lru_cache(maxsize=None)
def get_base_form(word):
    """
    Получить базовую форму слова, используя морфологическую аналитику.

    Параметры:
        word (str): Слово, для которого нужно получить базовую форму.

    Возвращает:
        str: Базовая форма слова.
    """
    parsed_word = morph.parse(word)[0]
    return parsed_word.normal_form


@lru_cache(maxsize=None)
def word2vec(word):
    """
    Функция, возвращающая вектор слова.

    Параметры:
        word (str): Слово, для которого должен быть получен вектор слов.

    Возвращает:
        float: Вектор для данного слова, если оно существует, в противном случае возвращает 0.
    """
    try:
        return model[word]
    except KeyError:
        return np.zeros(model.vector_size)


def cosdis(v1, v2):
    """
    Вычислите косинусное расстояние между двумя векторами.

    Параметры
        v1 (float): Первый вектор
        v2 (float): Второй вектор

    Возвращает:
        float: Косинусное сходство между двумя векторами
    """
    dot_product = np.dot(v1, v2)
    norm_vector1 = np.linalg.norm(v1)
    norm_vector2 = np.linalg.norm(v2)
    if norm_vector1 != 0 and norm_vector2 != 0:
        cosine_sim = dot_product / (norm_vector1 * norm_vector2)
    else:
        if norm_vector1 == norm_vector2:
            cosine_sim = 1.0
        else:
            cosine_sim = 0.0
    return cosine_sim


def line_vector_parallel(word):
    base_form, part_of_speech = word
    vector = word2vec(f"{base_form}_{part_of_speech}")
    return vector


def line_vector(line):
    """
     Вычислите векторное представление данной строки, суммируя векторы ее слов.

     Параметры:
         строка (str): строка текста, для которой вычисляется векторное представление.

     Возврат:
         float: векторное представление строки, вычисляемое путем суммирования векторов ее слов.
        Возвращает 0,0, если векторы не найдены.
     """
    words = correct_text(clean_text(line)).split()
    word_pairs = [(get_base_form(word), get_part_of_speech(word)) for word in words]
    with ThreadPoolExecutor() as executor:
        vectors = list(executor.map(line_vector_parallel, word_pairs))
    line_sim = [vector for vector in vectors if np.any(vector)]
    return sum(line_sim) / len(line_sim) if line_sim else 0.0
