# === main.py (Версия с Самодиагностикой) ===

# --- Акт 1: Подготовка Гримуаров ---
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
# --- > НОВЫЕ РУНЫ
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from validation_data import VALIDATION_SET

# --- Акт 2: Подготовка "Духа" внутри Портала ---
print("Призываю 'Духа Эмоций' для нашего амулета...")
sentiment_analyzer = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("'Дух Эмоций' готов к работе.")

# --- Акт 3: Чертеж "Магического Послания" ---
class TextInput(BaseModel):
    text: str

# --- Акт 4: Создание Портала и "Тревожного Колокола" ---
app = FastAPI(
    title="Амулет с Совестью",
    description="Магический портал, который определяет эмоции и следит за своей точностью.",
    version="2.0",
)

# --- > НОВЫЕ РУНЫ
# Стандартная инструментация для метрик запросов
Instrumentator().instrument(app).expose(app)
# Наш кастомный "Тревожный Колокол" (Gauge-метрика)
ACCURACY_GAUGE = Gauge(
    'model_accuracy',
    'Current accuracy of the sentiment analysis model on a validation set'
)

# --- Акт 5: Создание "Врат" для Анализа ---
@app.post("/analyze")
def analyze_sentiment(request: TextInput):
    text_to_analyze = request.text
    result = sentiment_analyzer(text_to_analyze)
    return {"result": result[0]}

# --- > НОВЫЙ АКТ 6: Ритуал Самопроверки ---
@app.post("/validate")
def validate_model():
    correct_predictions = 0
    total_samples = len(VALIDATION_SET)

    for item in VALIDATION_SET:
        text = item["text"]
        true_label = item["label"]
        
        # Получаем предсказание от нашего "Духа"
        prediction = sentiment_analyzer(text)[0]
        predicted_label = prediction['label']
        
        # Сверяемся с "Кристаллом Истины"
        if predicted_label == true_label:
            correct_predictions += 1
            
    # Вычисляем точность
    accuracy = correct_predictions / total_samples
    
    # Заставляем "Тревожный Колокол" звонить с новым значением
    ACCURACY_GAUGE.set(accuracy)
    
    # Возвращаем отчет о ритуале
    return {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_samples": total_samples
    }