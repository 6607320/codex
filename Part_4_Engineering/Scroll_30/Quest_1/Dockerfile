# --- Dockerfile для Квеста 30.1 ("Осадная Машина") ---

# Мы берем за основу легкий, чистый образ Python.
FROM python:3.10-slim

# Устанавливаем рабочую директорию.
WORKDIR /app

# Копируем "Свиток Зависимостей" для Locust.
COPY requirements.txt .

# Устанавливаем Locust.
RUN pip install --no-cache-dir -r requirements.txt

# Копируем наш "План Осады".
COPY locustfile.py .
