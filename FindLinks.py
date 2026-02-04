import re


def contains_contact_info(text):
    # Регулярные выражения для поиска контактов
    url_pattern = r"(https?://\S+|www\.\S+|\S+\.(com|ru|org|net|info|ua|by|kz|uz|eu|de|fr))"
    phone_pattern = r"(\+?\d[\d\s\-\(\)]{8,})"  # Поиск номеров телефонов
    username_pattern = r"(@\w+)"  # Поиск юзернеймов (@username)

    # Поиск совпадений
    url_match = re.search(url_pattern, text, re.IGNORECASE)
    phone_match = re.search(phone_pattern, text)
    username_match = re.search(username_pattern, text)

    # Проверяем, есть ли в тексте контакты
    if url_match or phone_match or username_match:
        return True  # Есть контактная информация
    return False  # Контактов нет
