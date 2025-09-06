# Используем стандартный, полный образ Python. Он больше, но надежнее.
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем свиток зависимостей
COPY requirements.txt .

# Устанавливаем зависимости НАПРЯМУЮ в финальный образ
RUN pip install --no-cache-dir -r requirements.txt

# Копируем наш код
COPY main.py .

# Копируем и делаем исполняемым наш скрипт-пробудитель
COPY start.sh .
RUN chmod +x ./start.sh

# Запускаем Голема через скрипт-пробудитель
CMD ["./start.sh"]