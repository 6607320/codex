# === Dockerfile (ФИНАЛЬНАЯ ВЕРСИЯ с чистой основой) ===

# Шаг 1: Выбираем "основу" от официальной гильдии PyTorch.
# Мы берем "слепок" для PyTorch 2.1.0 с CUDA 11.8.0 на Ubuntu 22.04.
# Это стабильная, проверенная временем комбинация.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Шаг 2: Подготовка "мастерской".
WORKDIR /app
COPY requirements.txt .

# Шаг 3: Призыв Гримуаров.
# Мы больше не удаляем torch, а просто устанавливаем наших помощников.
RUN pip install --no-cache-dir -r requirements.txt

# Шаг 4: Копируем наш "пергамент".
COPY quest_17_1.py .

# Шаг 5 и 6 остаются без изменений.
EXPOSE 8000
CMD ["uvicorn", "quest_17_1:app", "--host", "0.0.0.0", "--port", "8000"]