# main.py Specification

## 1. Meta Information

- Domain: ML/NLP
- Complexity: Medium
- Language: Python
- Frameworks: FastAPI, Pydantic, transformers, prometheus_fastapi_instrumentator, prometheus_client
- Context: Independent Artifact

## 2. Goal & Purpose (Легенда)

Этот скрипт превращает простой мониторинг «жив ли сервис» в управляемый бизнес-инструмент качества AI-модели. Через ритуал метрики он добавляет кастомную метрику model_accuracy и открывает врата самоанализа через /validate, чтобы Амулет держал курс на точность модели на валидаторе.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: API Request
- Format: JSON
- Schema: InputData
  - text: string

### 3.2. Outputs (Выходы)

- Destination: API Response
- Format: JSON
- Success Criteria: 200 OK
- Schema: OutputResult
  - Для анализа:
    - result: { label: string, score: number }
  - Для самопроверки:
    - accuracy: number
    - correct_predictions: number
    - total_samples: number

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Активируется ритуал подготовки — собираются необходимые артефакты: FastAPI порталы, чародейский свиток Pydantic и мистический дух анализа.
2. Призывается дух анализа речи, чтобы он мог ставить ярлыки эмоциональной окраски текста через предтренированную модель.
3. В распоряжении появляется Текстовый Рунник (пользовательский вход) — поле text.
4. Открывается Врата анализа (/analyze): получаем текст, прогоняем его через дух анализа и возвращаем послание вида result с меткой и силой уверенности.
5. Создаются Врата Самопроверки (/validate): циклом прогоняем каждый эталонный свиток из VALIDATION_SET, сравниваем модельное предсказание с истинной меткой, считаем количество верных ответов.
6. Вычисляется точность как отношение верных ответов к общему числу образцов; точность записывается в Тревожный Колокол (GAUGE) под именем model_accuracy.
7. Возвращается детальный отчет: accuracy, correct_predictions и total_samples.
8. Вся телесная часть портала дополнительно обрамлена инструментатором мониторинга, который выставляет метрики и делает их доступными для визуализации.

### 4.2. Declarative Content (Для конфигураций и данных)

- Валидативный набор (VAL_SET): набор эталонных текстов с истинными метками (POSITIVE/NEGATIVE), используемый для подсчета точности и обновления метрики.
- Метрика: ACCURACY_GAUGE с именем model_accuracy и описанием текущей точности анализа сентимент-сводки.
- Дух анализа: sentiment_analyzer — пайплайн типа sentiment-analysis на основе трансформеров (например, distilbert-based SST-2).
- Портал: FastAPI приложение с именем Амулет с Совестью, версией 2.0, маршруты /analyze и /validate.
- Инструменты: Instrumentator для автоматического экспонирования метрик Prometheus.

## 5. Structural Decomposition (Декомпозиция структуры)

- TextInput — модель данных для входящих запросов (поле text: string).
- app — FastAPI приложение с титулом и описанием, версия 2.0.
- sentiment_analyzer — дух анализа текста, инициализированный как пайплайн "sentiment-analysis".
- ACCURACY_GAUGE — метрика Prometheus типа Gauge с именем model_accuracy и описанием.
- /analyze endpoint — обрабатывает входной текст и возвращает результат анализа.
- /validate endpoint — прогоняет валидатор на VALIDATION_SET, обновляет модельную точность и возвращает детальный отчет.
- Instrumentator — инструмент мониторинга, подключающий метрики к приложению.
- validation_data.VALIDATION_SET — источник эталонных пар текста и истинной метки.
- Импортируемые модули: FastAPI, Pydantic BaseModel, transformers pipeline, Prometheus инструментатор и Gauge, VALIDATION_SET.

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Synchronous endpoints (def-установленные функции) обрабатываются синхронно
- Dependencies: fastapi, pydantic, transformers, prometheus_fastapi_instrumentator, prometheus_client
- Эфирная память: использование трансформеров предполагает умеренный потребление памяти в сочетании с локальным пайплайном; загружаются модели по запросу в момент инициализации, не в каждом вызове

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (использовать .env, переменные окружения)
- DO NOT выводить сырые данные в консоль в продакшене
- DO NOT выполнять блокирующие сетевые вызовы в основном цикле обработки
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты
- DO NOT менять версии библиотек или путей реконструкции проекта без явного уведомления

## 7. Verification & Testing (Верификация)

```
Фича: Функциональность амулета
  Сценарий: Успешный анализ
    Предусловия: Запущен сервер и доступен эндпоинт /analyze
    Когда выполняется POST /analyze с телом {"text": "The service is excellent"}
    Тогда ответ имеет статус 200 и содержит результат с полем label и score

  Сценарий: Ошибка валидации ввода
    Предусловия: Сервер запущен
    Когда выполняется POST /analyze с телом {}
    Тогда ответ имеет статус 422 Unprocessable Entity
```

ИЗСЛЕДУЕМЫЙ АРТЕФАКТ: main.py
