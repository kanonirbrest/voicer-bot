FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов проекта
COPY . .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Запуск бота
CMD ["python", "bot.py"] 