# Используем официальный образ Python
FROM python:3.12-slim

# Устанавливаем необходимые системные зависимости
RUN apt-get update && \
    apt-get install -y build-essential libssl-dev libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем зависимости Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Указываем команду для запуска приложения
ENV FLASK_DEBUG=0
CMD ["python", "App.py"]
