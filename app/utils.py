import os
import uuid
from PIL import Image
from flask import current_app

def save_optimized_avatar(file_storage) -> str:
    """
    Сжимает аватар пользователя, делает его квадратным и сохраняет в формате WebP.

    Args:
        file_storage (werkzeug.datastructures.FileStorage): Объект файла, содержащий исходное изображение (обычно получаемый из формы Flask).

    Returns:
        str: Путь к сохранённому аватару относительно папки static (например, "uploads/avatars/xxx.webp").
    """
    filename = uuid.uuid4().hex + ".webp"
    upload_path = os.path.join(current_app.root_path, 'static/uploads/avatars', filename)
    os.makedirs(os.path.dirname(upload_path), exist_ok=True)

    with Image.open(file_storage) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        min_side = min(width, height)
        left = (width - min_side) / 2
        top = (height - min_side) / 2
        right = (width + min_side) / 2
        bottom = (height + min_side) / 2
        img = img.crop((left, top, right, bottom))

        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        img.save(upload_path, 'WEBP', quality=85, optimize=True)

    return f"uploads/avatars/{filename}"
