# **Квест 17.1: Создание Амулета. Архитектурное руководство по инкапсуляции моделей классификации текста в микросервисы на базе FastAPI**

## **Аннотация**

Данный отчет представляет собой всестороннее исследование методологии развертывания моделей машинного обучения (ML) с использованием фреймворка FastAPI. В контексте метафорической задачи "Квест 17.1: Создание Амулета" рассматривается процесс трансформации "сырой" ML-модели (артефакта) в полноценный, производительный и надежный веб\-сервис ("Амулет"). Отчет охватывает полный жизненный цикл разработки: от архитектурного проектирования и оптимизации инференса с помощью ONNX до управления конкурентностью, валидации данных и стратегий промышленного развертывания. Особое внимание уделяется решению проблем производительности в CPU-bound задачах, управлению жизненным циклом приложения (Lifespan Events) и построению отказоустойчивой архитектуры микросервисов. Объем и детализация материала ориентированы на создание исчерпывающей базы знаний для инженеров MLOps и Python-разработчиков.

## ---

**Глава 1\. Эволюция Архитектуры ML-систем и Выбор Инструментария**

### **1.1. Смена Парадигмы: От Монолита к Микросервисам**

В современной разработке программного обеспечения наблюдается фундаментальный сдвиг в подходах к интеграции искусственного интеллекта. Традиционные монолитные архитектуры, где логика предсказания была жестко связана с основным бэкендом или исполнялась в виде пакетных оффлайн-скриптов, уступают место гибким микросервисным решениям. В этой новой парадигме модель машинного обучения рассматривается как независимый функциональный блок — "Амулет", обладающий четко определенным интерфейсом и скрытой внутренней сложностью.

Изоляция ML-логики в отдельный микросервис решает несколько критических задач:

1. **Разделение жизненных циклов:** Модели обновляются и переобучаются с другой частотой, нежели основной код приложения.
2. **Масштабируемость:** ML-сервисы часто требуют специфического оборудования (GPU, большой объем RAM), отличного от требований к обычным веб\-серверам.
3. **Технологическая агностичность:** Основное приложение может быть написано на Go или Java, взаимодействуя с Python-сервисом ML через стандартные протоколы HTTP или gRPC.1

### **1.2. FastAPI как Стандарт Де-факто**

Выбор FastAPI для реализации ML-микросервисов не случаен и обусловлен рядом архитектурных преимуществ перед предшественниками, такими как Flask или Django.

#### **Производительность и Асинхронность (ASGI)**

Ключевым отличием FastAPI является его база — стандарт ASGI (Asynchronous Server Gateway Interface) и использование Starlette под капотом. В отличие от WSGI (используемого Flask), который обрабатывает запросы синхронно и блокирующе, ASGI позволяет обрабатывать тысячи конкурентных соединений, используя механизм async/await в Python. Это обеспечивает высокую пропускную способность, сравнимую с сервисами на Node.js или Go, что критически важно для высоконагруженных систем.2

#### **Строгая Типизация и Pydantic**

FastAPI глубоко интегрирован с библиотекой Pydantic, что позволяет определять схемы данных (Data Contracts) с использованием стандартных подсказок типов Python. Это превращает код в "исполняемую спецификацию": валидация входных данных происходит автоматически до выполнения бизнес-логики, что исключает целый класс ошибок, связанных с некорректным форматом данных. Для ML-моделей, чувствительных к типам входных тензоров, это свойство является незаменимым.4

#### **Таблица 1.1. Сравнительный анализ фреймворков для ML-сервисов**

| Характеристика          | Flask                          | Django REST Framework             | FastAPI                          |
| :---------------------- | :----------------------------- | :-------------------------------- | :------------------------------- |
| **Основа**              | WSGI (синхронный)              | WSGI (преимущественно синхронный) | ASGI (асинхронный)               |
| **Валидация данных**    | Расширения (Marshmallow и др.) | Встроенные сериализаторы          | Нативная интеграция Pydantic     |
| **Скорость разработки** | Высокая (минимализм)           | Средняя (много boilerplate)       | Очень высокая (авто-доки)        |
| **Производительность**  | Низкая/Средняя                 | Средняя                           | Высокая                          |
| **Документация API**    | Требует настройки (Flasgger)   | Требует настройки (drf-yasg)      | Автоматическая (OpenAPI/Swagger) |
| **Поддержка WebSocket** | Через расширения               | Через Django Channels             | Нативная                         |

### **1.3. Архитектура Проекта: Принципы Clean Architecture**

Создание "Амулета" требует строгой организации кодовой базы. Хранение всей логики в одном файле main.py допустимо лишь для прототипов. Для промышленного решения рекомендуется структура, разделяющая ответственность компонентов.

Рекомендуемая структура проекта:

- app/
  - main.py — Точка входа, инициализация приложения.
  - api/ — Маршрутизация (routers) и обработчики запросов.
  - core/ — Конфигурация, настройки (pydantic-settings), логирование.
  - models/ — Pydantic-схемы (DTO) для запросов и ответов.
  - services/ — Бизнес-логика: загрузка модели, препроцессинг, инференс.
  - utils/ — Вспомогательные утилиты.
- tests/ — Модульные и интеграционные тесты.
- Dockerfile — Инструкции для сборки образа.

Такое разделение позволяет изолировать инфраструктурный код (веб-сервер) от доменной логики (машинное обучение), облегчая тестирование и поддержку.6

## ---

**Глава 2\. Подготовка Артефакта Модели: Оптимизация и Экспорт**

### **2.1. Проблематика "Сырых" Моделей**

Обученная модель классификации текста (например, на базе архитектуры BERT или DistilBERT) в формате PyTorch (pytorch_model.bin) или TensorFlow часто является субоптимальной для инференса. "Сырые" веса содержат информацию, необходимую для обучения (градиенты, вспомогательные слои), которая избыточно потребляет память и вычислительные ресурсы в процессе эксплуатации. Кроме того, зависимость от полновесных фреймворков (PyTorch/TF) значительно увеличивает размер Docker-образа (до 2-3 ГБ), что усложняет деплой и масштабирование.

### **2.2. Стандарт ONNX: Универсальный Формат Обмена**

Решением проблемы является конвертация модели в формат ONNX (Open Neural Network Exchange). ONNX предоставляет унифицированное представление графа вычислений, которое может быть исполнено на высокопроизводительном движке ONNX Runtime.

Преимущества использования ONNX Runtime:

1. **Оптимизация графа:** Слияние операторов (Operator Fusion), удаление неиспользуемых узлов, constant folding.
2. **Аппаратная гибкость:** Единый API для исполнения на CPU, GPU (CUDA, TensorRT) и специализированных ускорителях (OpenVINO, DirectML).
3. **Легковесность:** Runtime-библиотека занимает значительно меньше места, чем фреймворки обучения.8

### **2.3. Квантование: Сжатие без Потери Качества**

Квантование (Quantization) — это метод оптимизации, при котором веса модели и функции активации приводятся к меньшему битовому разрешению. Стандартные модели используют 32-битные числа с плавающей точкой (float32). Конвертация в 8-битные целые числа (int8) позволяет:

- Уменьшить размер модели в 2-4 раза.
- Ускорить вычисления на CPU за счет использования векторных инструкций (AVX2, AVX-512, VNNI).
- Снизить пропускную способность памяти, требуемую для загрузки весов.8

Для моделей трансформеров наиболее эффективным методом на CPU часто является **динамическое квантование** (Dynamic Quantization). В этом режиме веса линейных слоев хранятся в int8, но активации вычисляются в float32 (с конвертацией "на лету"), что обеспечивает баланс между скоростью и точностью.

### **2.4. Инструментарий Hugging Face Optimum**

Библиотека Optimum предоставляет интерфейс высокого уровня для экспорта и оптимизации моделей из экосистемы Transformers. Она автоматизирует сложные процессы трассировки графа и конфигурации экспорта.

#### **Процесс Экспорта и Оптимизации**

Использование optimum-cli позволяет выполнить экспорт, оптимизацию графа и квантование одной командой.

Пример команды для экспорта модели DistilBERT с уровнем оптимизации O3:

Bash

optimum-cli export onnx \--model distilbert-base-uncased-finetuned-sst-2-english \--optimize O3 distilbert_onnx_quantized/

Параметр \--optimize O3 включает:

- Базовые оптимизации графа.
- Слияние специфичных для трансформеров операторов (например, Multi-Head Attention fusion).
- Аппроксимацию функции активации GELU.
- (Опционально) Квантование, если указаны соответствующие флаги.10

Полученная директория distilbert_onnx_quantized/ будет содержать model.onnx и конфигурационные файлы токенизатора (tokenizer.json, vocab.txt). Этот набор файлов и есть наш подготовленный "Амулет", готовый к интеграции.

#### **Таблица 2.1. Уровни оптимизации в Optimum**

| Уровень | Описание                                                                | Применение                                   |
| :------ | :---------------------------------------------------------------------- | :------------------------------------------- |
| **O1**  | Базовые общие оптимизации (constant folding, dead code elimination)     | Безопасно для всех моделей                   |
| **O2**  | Расширенные оптимизации \+ фьюзинг операторов трансформеров             | Значительное ускорение BERT-подобных моделей |
| **O3**  | O2 \+ аппроксимация GELU (быстрее, но микроскопическая потеря точности) | Рекомендуемый уровень для продакшена         |
| **O4**  | O3 \+ смешанная точность (fp16)                                         | Только для GPU (CUDA)                        |

## ---

**Глава 3\. Управление Жизненным Циклом Приложения (Lifespan Events)**

### **3.1. Критичность Правильной Инициализации**

Одной из наиболее распространенных ошибок при разработке ML-сервисов является загрузка модели внутри обработчика запроса.

- **Неправильно:** Загружать модель внутри функции predict. Это приводит к тому, что каждый запрос инициирует чтение диска и аллокацию памяти, увеличивая время ответа с миллисекунд до секунд и вызывая переполнение памяти (OOM) под нагрузкой.
- **Правильно:** Загружать модель один раз при старте приложения и хранить её в памяти, переиспользуя для всех запросов.13

### **3.2. Lifespan Protocol и AsyncContextManager**

В ранних версиях FastAPI использовались декораторы @app.on_event("startup") и @app.on_event("shutdown"). Начиная с версии 0.93.0 и перехода на Starlette 0.26.1, рекомендуемым подходом является использование параметра lifespan.

Lifespan представляет собой асинхронный контекстный менеджер, который охватывает все время жизни приложения. Это позволяет явно управлять ресурсами в едином блоке кода, гарантируя, что ресурсы, выделенные при старте (startup), будут корректно освобождены при остановке (shutdown), даже в случае ошибок.

#### **Реализация Lifespan для ML-модели**

Ниже приведен пример реализации загрузчика модели с использованием библиотеки optimum и fastapi.

Python

from contextlib import asynccontextmanager  
from fastapi import FastAPI  
from transformers import AutoTokenizer  
from optimum.onnxruntime import ORTModelForSequenceClassification

\# Глобальное хранилище для моделей (или можно использовать app.state)  
ml_models \= {}

@asynccontextmanager  
async def lifespan(app: FastAPI):  
 \# \--- Блок Startup \---  
 \# Этот код выполняется до того, как приложение начнет принимать запросы.  
 print("Инициализация артефактов модели...")

    model\_path \= "distilbert\_onnx\_quantized/" \# Путь к локальной оптимизированной модели

    \# Загрузка токенизатора и модели в память
    \# ORTModelForSequenceClassification автоматически загружает ONNX runtime
    ml\_models\["tokenizer"\] \= AutoTokenizer.from\_pretrained(model\_path)
    ml\_models\["model"\] \= ORTModelForSequenceClassification.from\_pretrained(model\_path)

    print(f"Модель загружена из {model\_path}")

    yield \# Передача управления приложению

    \# \--- Блок Shutdown \---
    \# Этот код выполняется после остановки обработки запросов.
    print("Очистка ресурсов ML...")
    ml\_models.clear()
    \# Здесь можно также закрыть соединения с БД или пулами потоков

app \= FastAPI(lifespan=lifespan)

Использование lifespan обеспечивает гарантию того, что приложение не перейдет в статус "Ready" (в терминах Kubernetes Readiness Probe), пока модель полностью не загрузится. Если загрузка модели завершится ошибкой (например, файл не найден), приложение аварийно остановится на старте, что предотвратит запуск неработоспособного пода.13

### **3.3. Lazy Loading: За и Против**

Существует паттерн "Ленивой загрузки" (Lazy Loading), когда модель загружается при первом запросе. Хотя это ускоряет старт приложения, для ML-сервисов в продакшене это часто является анти-паттерном. Первый пользователь получит огромную задержку (Cold Start latency), а конкурентные первые запросы могут вызвать состояние гонки (Race Condition) при инициализации. Поэтому для основного "Амулета" рекомендуется жадная загрузка (Eager Loading).15

## ---

**Глава 4\. Проектирование Контрактов Данных с Pydantic**

В основе надежного API лежит строгая спецификация данных. FastAPI использует Pydantic для валидации, сериализации и генерации JSON Schema.

### **4.1. Модели Запроса (Request Models)**

Для классификации текста входные данные должны быть строго типизированы. Мы определяем структуру JSON, который ожидаем от клиента.

Python

from pydantic import BaseModel, Field

class TextClassificationRequest(BaseModel):  
 text: str \= Field(  
 ...,  
 title="Текст для анализа",  
 description="Входная строка для классификации. Не должна быть пустой.",  
 min_length=1,  
 max_length=5000, \# Ограничение для защиты от DoS атак  
 examples=\["FastAPI — отличный инструмент для ML сервисов\!"\]  
 )

    \# Конфигурация для документации (Pydantic v2 style)
    model\_config \= {
        "json\_schema\_extra": {
            "example": {
                "text": "Анализ тональности показывает отличные результаты."
            }
        }
    }

Использование Field позволяет задать ограничения (валидаторы) непосредственно в схеме. Если клиент отправит пустую строку или слишком длинный текст, FastAPI автоматически вернет ошибку 422 Unprocessable Entity с детальным описанием проблемы, не нагружая модель инференса.4

### **4.2. Модели Ответа (Response Models)**

Четкое определение формата ответа позволяет фронтенд-разработчикам и потребителям API автоматически генерировать клиентский код.

Python

class ClassificationPrediction(BaseModel):  
 label: str \= Field(..., description="Предсказанный класс (например, POSITIVE, NEGATIVE)")  
 score: float \= Field(..., description="Уверенность модели (вероятность)", ge=0.0, le=1.0)

class ServiceResponse(BaseModel):  
 result: ClassificationPrediction  
 processing_time_ms: float | None \= None

Указание response_model в декораторе эндпоинта включает автоматическую фильтрацию данных. Даже если ваша функция вернет словарь с лишними служебными полями, FastAPI удалит их перед отправкой клиенту, обеспечивая безопасность данных.18

## ---

**Глава 5\. Конкурентность и Производительность: Sync vs Async**

Один из самых сложных и часто неправильно понимаемых аспектов FastAPI — это выбор между def (синхронными функциями) и async def (асинхронными корутинами) для ML-задач.

### **5.1. GIL и Блокировка Event Loop**

Python использует Global Interpreter Lock (GIL), который позволяет только одному потоку исполнять байт-код Python в один момент времени.

- **I/O-Bound задачи:** Операции, ожидающие ответа от внешних систем (БД, сеть, диск). async/await позволяет освободить Event Loop во время ожидания, давая возможность обрабатывать другие запросы.
- **CPU-Bound задачи:** Интенсивные вычисления, полностью загружающие процессор. Инференс нейронной сети (прогон тензоров через слои, токенизация) — это классическая CPU-Bound задача.3

### **5.2. Опасность async def для CPU задач**

Если вы определите эндпоинт как async def, но внутри будете выполнять тяжелые вычисления (например, model(input)) без await (так как библиотеки ML часто синхронны), вы **заблокируете Event Loop**. Весь сервер "зависнет" для всех пользователей, пока вычисления не закончатся. Даже запросы на /health не будут обрабатываться.21

### **5.3. Стратегии Обработки Инференса**

#### **Стратегия А: Синхронный Эндпоинт (def)**

Если объявить эндпоинт просто как def (без async), FastAPI автоматически запустит эту функцию в отдельном потоке из внешнего пула потоков (Thread Pool).

Python

@app.post("/predict")  
def predict(request: TextClassificationRequest):  
 \# Выполняется в отдельном потоке. Event Loop не блокируется.  
 \# Но количество одновременных запросов ограничено размером пула потоков (обычно 40).  
 return run_inference(request.text)

Это безопасный вариант по умолчанию, но он имеет накладные расходы на переключение контекста потоков (Context Switching) и ограничен масштабируемостью пула.3

#### **Стратегия Б: Асинхронный Эндпоинт с run_in_executor**

Для максимального контроля и производительности рекомендуется использовать async def и явно отправлять блокирующие задачи в исполнитель (executor).

Python

import asyncio

@app.post("/predict")  
async def predict(request: TextClassificationRequest):  
 loop \= asyncio.get_running_loop()  
 \# Явная отправка тяжелой задачи в ThreadPoolExecutor  
 prediction \= await loop.run_in_executor(None, run_inference_logic, request.text)  
 return prediction

Это позволяет комбинировать асинхронные операции (например, запись лога запроса в БД) с синхронными вычислениями модели, не блокируя основной цикл.3

#### **Таблица 5.1. Выбор стратегии конкурентности**

| Тип задачи                      | Синтаксис эндпоинта          | Механизм выполнения     | Рекомендация                |
| :------------------------------ | :--------------------------- | :---------------------- | :-------------------------- |
| **I/O Bound** (БД, внешние API) | async def                    | Native await            | **Идеально**                |
| **CPU Bound** (ML Inference)    | async def (прямой вызов)     | Блокировка Event Loop   | **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** |
| **CPU Bound** (ML Inference)    | def                          | Thread Pool (Starlette) | **Хорошо** (простота)       |
| **CPU Bound** (ML Inference)    | async def \+ run_in_executor | Thread/Process Pool     | **Отлично** (гибкость)      |

## ---

**Глава 6\. Реализация Эндпоинта "Амулета"**

Интегрируем все концепции в единый модуль реализации. Мы будем использовать run_in_executor для обеспечения неблокирующего поведения.

### **6.1. Логика Инференса**

Сначала определим функцию, которая выполняет "грязную работу" по предсказанию. Она должна быть изолирована от HTTP-контекста.

Python

\# app/services/inference.py  
import numpy as np

def softmax(x):  
 """Вычисление Softmax для получения вероятностей."""  
 e_x \= np.exp(x \- np.max(x))  
 return e_x / e_x.sum()

def run_model_inference(text: str, tokenizer, model) \-\> dict:  
 """  
 Синхронная функция, выполняющая CPU-intensive работу.  
 """  
 \# 1\. Токенизация (CPU Bound)  
 inputs \= tokenizer(text, return_tensors="np", padding=True, truncation=True)

    \# Подготовка данных для ONNX Runtime (требует int64 для input\_ids)
    ort\_inputs \= {k: v.astype(np.int64) for k, v in inputs.items()}

    \# 2\. Инференс модели (CPU Bound)
    \# run(output\_names, input\_feed)
    logits \= model.run(None, ort\_inputs)

    \# 3\. Постобработка (Softmax)
    probs \= softmax(logits)
    pred\_idx \= np.argmax(probs)

    \# Маппинг (пример для SST-2: 0=NEGATIVE, 1=POSITIVE)
    labels \=
    label \= labels\[pred\_idx\] if pred\_idx \< len(labels) else "UNKNOWN"

    return {"label": label, "score": float(probs\[pred\_idx\])}

### **6.2. Роутер FastAPI**

Теперь создадим роутер, который связывает HTTP запрос с функцией инференса.

Python

\# app/api/endpoints.py  
from fastapi import APIRouter, HTTPException, Request, Depends  
from app.models.schemas import TextClassificationRequest, ClassificationPrediction  
from app.services.inference import run_model_inference  
import asyncio  
from functools import partial

router \= APIRouter()

@router.post("/predict", response_model=ClassificationPrediction, status_code=200)  
async def predict_sentiment(  
 request: Request,  
 payload: TextClassificationRequest  
):  
 """  
 Эндпоинт для классификации текста.  
 Использует загруженную в lifespan модель.  
 """  
 \# Извлечение модели из глобального состояния или переменной  
 \# В данном примере используем глобальный словарь ml_models, импортированный из main  
 from app.main import ml_models

    if "model" not in ml\_models or "tokenizer" not in ml\_models:
        raise HTTPException(status\_code=503, detail="Модель еще не инициализирована")

    loop \= asyncio.get\_running\_loop()

    \# Используем partial для передачи аргументов в функцию, запускаемую в экзекьюторе
    inference\_func \= partial(
        run\_model\_inference,
        text=payload.text,
        tokenizer=ml\_models\["tokenizer"\],
        model=ml\_models\["model"\]
    )

    \# Неблокирующий запуск инференса
    result \= await loop.run\_in\_executor(None, inference\_func)

    return result

Этот код демонстрирует чистую интеграцию: асинхронный контроллер делегирует тяжелую работу пулу потоков, обеспечивая отзывчивость сервиса.13

## ---

**Глава 7\. Стратегия Тестирования (Quality Assurance)**

Создание надежного "Амулета" невозможно без комплексного тестирования. В ML-проектах тестирование имеет свои особенности: необходимость мокирования (подмены) тяжелых моделей для ускорения тестов.

### **7.1. Инструментарий: Pytest и TestClient**

FastAPI предоставляет TestClient (обертка над библиотекой httpx), который позволяет отправлять запросы к приложению напрямую, без поднятия реального сервера.22

### **7.2. Использование Фикстур (Fixtures) для Мокирования**

Загружать реальную модель BERT в юнит-тестах нецелесообразно (долго, дорого по памяти). Мы используем pytest фикстуры и unittest.mock для подмены логики инференса.

Python

\# tests/conftest.py  
import pytest  
from fastapi.testclient import TestClient  
from unittest.mock import MagicMock  
from app.main import app, ml_models

@pytest.fixture  
def client():  
 \# Контекстный менеджер TestClient вызывает события startup/shutdown  
 with TestClient(app) as c:  
 yield c

@pytest.fixture(autouse=True)  
def mock_ml_inference(monkeypatch):  
 """  
 Автоматически подменяет реальную модель на мок для всех тестов.  
 Это предотвращает загрузку тяжелой модели.  
 """  
 mock_tokenizer \= MagicMock()  
 mock_tokenizer.return_value \= {"input_ids": }

    mock\_model \= MagicMock()
    \# Эмуляция вывода ONNX Runtime: \[logits\]
    mock\_model.run.return\_value \= \[\[\[-0.5, 2.5\]\]\] \# Логиты, где класс 1 имеет больший вес

    \# Подмена глобального словаря
    ml\_models\["tokenizer"\] \= mock\_tokenizer
    ml\_models\["model"\] \= mock\_model

    yield

    ml\_models.clear()

### **7.3. Сценарии Тестирования**

Тест 1: Успешная Классификация (Happy Path)  
Проверяем, что API возвращает корректный статус 200 и структуру JSON при валидных данных.

Python

def test_predict_positive(client):  
 response \= client.post("/predict", json={"text": "Это отличный результат\!"})  
 assert response.status_code \== 200  
 data \= response.json()  
 assert data\["label"\] \== "POSITIVE"  
 assert data\["score"\] \> 0.5

Тест 2: Валидация Данных  
Проверяем реакцию на некорректный ввод (пустая строка).

Python

def test_predict_validation_error(client):  
 response \= client.post("/predict", json={"text": ""}) \# Пусто, а min_length=1  
 assert response.status_code \== 422  
 assert "detail" in response.json()

Такой подход обеспечивает выполнение принципа "Пирамиды тестирования": много быстрых юнит-тестов с моками и минимум медленных end-to-end тестов с реальной моделью.24

## ---

**Глава 8\. Развертывание и Операционная Среда (Deployment)**

### **8.1. Сервер Приложений: Uvicorn vs Gunicorn**

Для запуска FastAPI в продакшене используется ASGI-сервер Uvicorn.  
Команда разработки: uvicorn main:app \--reload.  
Команда продакшена: uvicorn main:app \--host 0.0.0.0 \--port 8000\.

#### **Проблема Воркеров и Памяти**

Uvicorn позволяет запускать несколько рабочих процессов (workers) через флаг \--workers.

Bash

uvicorn main:app \--workers 4

Важное предостережение для ML: В отличие от обычных веб\-приложений, ML-сервисы потребляют много памяти. Если модель весит 1 ГБ, то запуск 4 воркеров потребует 4+ ГБ оперативной памяти, так как каждый процесс загружает свою копию модели (Python не всегда эффективно разделяет память между процессами из\-за Copy-on-Write).  
В среде Kubernetes часто эффективнее запускать один процесс Uvicorn на один Под и масштабировать количество Подов, а не воркеров внутри пода. Это упрощает управление ресурсами и автоскейлинг.26

### **8.2. Контейнеризация (Docker)**

Создание Docker-образа для "Амулета" требует оптимизации размера. Использование slim-версий Python и многоэтапной сборки (multi-stage builds) является лучшей практикой.

Пример Dockerfile:

Dockerfile

\# Используем легкий базовый образ  
FROM python:3.10\-slim

\# Установка переменных окружения для оптимизации Python  
ENV PYTHONDONTWRITEBYTECODE=1 \\  
 PYTHONUNBUFFERED=1

WORKDIR /app

\# Установка системных зависимостей (если нужны для компиляции)  
\# RUN apt-get update && apt-get install \-y \--no-install-recommends gcc...

\# Копирование зависимостей  
COPY requirements.txt.  
RUN pip install \--no-cache-dir \-r requirements.txt

\# Копирование кода приложения  
COPY./app./app

\# Копирование артефактов модели (Амулета)  
\# В реальном CI/CD модель часто скачивается с S3/DVC при сборке или маунтится как volume  
COPY./distilbert_onnx_quantized./distilbert_onnx_quantized

\# Создание непривилегированного пользователя для безопасности  
RUN useradd \-m appuser  
USER appuser

\# Запуск  
CMD \["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"\]

Этот образ содержит все необходимое для автономной работы сервиса.6

## ---

**Заключение**

Интеграция модели машинного обучения в FastAPI endpoint — это комплексная инженерная задача, выходящая далеко за рамки написания скрипта predict.py. Созданный "Амулет" представляет собой сложную систему, где каждый слой — от квантованной модели ONNX до асинхронного контроллера и Pydantic-схем — играет критическую роль в обеспечении производительности и надежности.

Ключевые выводы исследования:

1. **Архитектура первична:** Разделение ответственности и использование асинхронных паттернов (Lifespan, run_in_executor) предотвращают блокировки и обеспечивают масштабируемость.
2. **Оптимизация обязательна:** Использование "сырых" моделей в продакшене неэффективно. Экспорт в ONNX и квантование с помощью Optimum позволяют кратно увеличить пропускную способность.
3. **Контракты данных:** Pydantic обеспечивает строгую типизацию интерфейсов, защищая сервис от некорректных данных.
4. **Среда исполнения:** Правильная конфигурация Uvicorn и Docker с учетом специфики потребления памяти ML-моделями критична для стабильности.

Следуя представленному руководству, разработчик переходит от статуса "экспериментатора с ноутбуками" к инженеру, создающему промышленные ML-сервисы уровня Enterprise.

#### **Источники**

1. FastAPI for Scalable Microservices: Best Practices & Optimisation \- Webandcrafts, дата последнего обращения: декабря 21, 2025, [https://webandcrafts.com/blog/fastapi-scalable-microservices](https://webandcrafts.com/blog/fastapi-scalable-microservices)
2. Building a Machine Learning Microservice with FastAPI | NVIDIA Technical Blog, дата последнего обращения: декабря 21, 2025, [https://developer.nvidia.com/blog/building-a-machine-learning-microservice-with-fastapi/](https://developer.nvidia.com/blog/building-a-machine-learning-microservice-with-fastapi/)
3. Asynchronous vs. Synchronous Functions in FastAPI When to Pick Which | Leapcell, дата последнего обращения: декабря 21, 2025, [https://leapcell.io/blog/asynchronous-vs-synchronous-functions-in-fastapi-when-to-pick-which](https://leapcell.io/blog/asynchronous-vs-synchronous-functions-in-fastapi-when-to-pick-which)
4. Request Body \- FastAPI, дата последнего обращения: декабря 21, 2025, [https://fastapi.tiangolo.com/tutorial/body/](https://fastapi.tiangolo.com/tutorial/body/)
5. Declare Request Example Data \- FastAPI, дата последнего обращения: декабря 21, 2025, [https://fastapi.tiangolo.com/tutorial/schema-extra-example/](https://fastapi.tiangolo.com/tutorial/schema-extra-example/)
6. Building Scalable Microservices with Python FastAPI: Design and Best Practices \- Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@kanishk.khatter/building-scalable-microservices-with-python-fastapi-design-and-best-practices-0dd777141b29](https://medium.com/@kanishk.khatter/building-scalable-microservices-with-python-fastapi-design-and-best-practices-0dd777141b29)
7. Building Enterprise Python Microservices with FastAPI in 2025 (1/10): Introduction, дата последнего обращения: декабря 21, 2025, [https://blog.devops.dev/building-enterprise-python-microservices-with-fastapi-in-2025-1-10-introduction-c1f6bce81e36](https://blog.devops.dev/building-enterprise-python-microservices-with-fastapi-in-2025-1-10-introduction-c1f6bce81e36)
8. Quantization \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/docs/optimum/concept_guides/quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)
9. Optimizing Transformers with Hugging Face Optimum \- Philschmid, дата последнего обращения: декабря 21, 2025, [https://www.philschmid.de/optimizing-transformers-with-optimum](https://www.philschmid.de/optimizing-transformers-with-optimum)
10. Optimization \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/optimization](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/optimization)
11. Optimization \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/optimization](https://huggingface.co/docs/optimum-onnx/en/onnxruntime/usage_guides/optimization)
12. Introducing Optimum by HuggingFace | by Vinish M \- Medium, дата последнего обращения: декабря 21, 2025, [https://vinishm.medium.com/introducing-optimum-by-huggingface-52729f0065af](https://vinishm.medium.com/introducing-optimum-by-huggingface-52729f0065af)
13. Lifespan Events \- FastAPI, дата последнего обращения: декабря 21, 2025, [https://fastapi.tiangolo.com/advanced/events/](https://fastapi.tiangolo.com/advanced/events/)
14. FastAPI: After the Getting Started | by Marc Nealer \- Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@marcnealer/fastapi-after-the-getting-started-867ecaa99de9](https://medium.com/@marcnealer/fastapi-after-the-getting-started-867ecaa99de9)
15. The Lazy Loading Pattern: How to Make Python Programs Feel Instant \- YouTube, дата последнего обращения: декабря 21, 2025, [https://www.youtube.com/watch?v=ENnDxEOAKKc](https://www.youtube.com/watch?v=ENnDxEOAKKc)
16. Intro to FastAPI: Tips and Tricks for ML \- YouTube, дата последнего обращения: декабря 21, 2025, [https://www.youtube.com/watch?v=nvLRjtvu1nk](https://www.youtube.com/watch?v=nvLRjtvu1nk)
17. Optimal way to initialize heavy services only once in FastAPI \- Stack Overflow, дата последнего обращения: декабря 21, 2025, [https://stackoverflow.com/questions/67663970/optimal-way-to-initialize-heavy-services-only-once-in-fastapi](https://stackoverflow.com/questions/67663970/optimal-way-to-initialize-heavy-services-only-once-in-fastapi)
18. Response Model \- Return Type \- FastAPI, дата последнего обращения: декабря 21, 2025, [https://fastapi.tiangolo.com/tutorial/response-model/](https://fastapi.tiangolo.com/tutorial/response-model/)
19. FastAPI Tutorial \- Read the Docs, дата последнего обращения: декабря 21, 2025, [https://fastapi-tutorial.readthedocs.io/](https://fastapi-tutorial.readthedocs.io/)
20. Concurrency and async / await \- FastAPI, дата последнего обращения: декабря 21, 2025, [https://fastapi.tiangolo.com/async/](https://fastapi.tiangolo.com/async/)
21. actual difference between synchronous and asynchronous endpoints : r/FastAPI \- Reddit, дата последнего обращения: декабря 21, 2025, [https://www.reddit.com/r/FastAPI/comments/1gyql0a/actual_difference_between_synchronous_and/](https://www.reddit.com/r/FastAPI/comments/1gyql0a/actual_difference_between_synchronous_and/)
22. Testing \- FastAPI, дата последнего обращения: декабря 21, 2025, [https://fastapi.tiangolo.com/tutorial/testing/](https://fastapi.tiangolo.com/tutorial/testing/)
23. Test Client \- TestClient \- FastAPI, дата последнего обращения: декабря 21, 2025, [https://fastapi.tiangolo.com/reference/testclient/](https://fastapi.tiangolo.com/reference/testclient/)
24. Testing FastAPI Applications with pytest | CodeSignal Learn, дата последнего обращения: декабря 21, 2025, [https://codesignal.com/learn/courses/model-serving-with-fastapi/lessons/testing-fastapi-applications-with-pytest](https://codesignal.com/learn/courses/model-serving-with-fastapi/lessons/testing-fastapi-applications-with-pytest)
25. Testing FastAPI Application with Pytest | by Fedor GNETKOV | Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@gnetkov/testing-fastapi-application-with-pytest-57080960fd62](https://medium.com/@gnetkov/testing-fastapi-application-with-pytest-57080960fd62)
26. Server Workers \- Uvicorn with Workers \- FastAPI, дата последнего обращения: декабря 21, 2025, [https://fastapi.tiangolo.com/deployment/server-workers/](https://fastapi.tiangolo.com/deployment/server-workers/)
27. Mastering Gunicorn and Uvicorn: The Right Way to Deploy FastAPI Applications \- Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@iklobato/mastering-gunicorn-and-uvicorn-the-right-way-to-deploy-fastapi-applications-aaa06849841e](https://medium.com/@iklobato/mastering-gunicorn-and-uvicorn-the-right-way-to-deploy-fastapi-applications-aaa06849841e)
