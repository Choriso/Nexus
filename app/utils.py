import os
import uuid
from PIL import Image
from flask import current_app


def save_optimized_avatar(file_storage):
    """Сжимает аватар, делает его квадратным и сохраняет в WebP"""
    # 1. Генерация уникального имени
    filename = uuid.uuid4().hex + ".webp"

    # Путь сохранения (используем путь к твоему проекту)
    upload_path = os.path.join(current_app.root_path, 'static/uploads/avatars', filename)

    # Убеждаемся, что папка существует
    os.makedirs(os.path.dirname(upload_path), exist_ok=True)

    # 2. Обработка изображения
    with Image.open(file_storage) as img:
        # Конвертируем в RGB (убирает прозрачность PNG для оптимизации)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Делаем "Crop" до квадрата (центровка)
        width, height = img.size
        min_side = min(width, height)
        left = (width - min_side) / 2
        top = (height - min_side) / 2
        right = (width + min_side) / 2
        bottom = (height + min_side) / 2
        img = img.crop((left, top, right, bottom))

        # Ресайз (200x200 или 400x400 вполне хватит для аватара)
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)

        # 3. Сохранение со сжатием
        img.save(upload_path, 'WEBP', quality=85, optimize=True)

    return f"uploads/avatars/{filename}"
