# main.py Specification

## 1. Meta Information

- **Domain:** Infrastructure
- **Complexity:** Medium
- **Language:** Python
- **Frameworks:** FastAPI, Pydantic, transformers, prometheus_fastapi_instrumentator, prometheus_client
- **Context:** ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

Легенда: Этот артефакт — Амулет с Развилкой, который в реальном времени слепо сравнивает две модели через API, собирая метрики и выводя решения на постоянном потоке для принятия решений о выкладке.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** API Request
- **Format:** JSON
- **Schema:**

```ts
interface InputData {
  text: string;
}
```

### 3.2. Outputs (Выходы)

- **Destination:** API Response
- **Format:** JSON
- **Success Criteria:** 200 OK
- **Schema:**

```ts
interface OutputResult {
  // Для анализа через /analyze
  model_version?: "A" | "B";
  result?: {
    label: string;
    score: number;
  };
  // Для самопроверки через /validate
  model_tested?: "A";
  accuracy?: number;
  correct_predictions?: number;
  total_samples?: number;
}
```

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Вступительный ритуал призывает двух духоподобных духов анализа: дух А и дух Б. Первый дух отвечает за DistilBERT версия A, второй дух — RoBERTa версия B.
2. Сотворённый портал именуется Амулетом с Развилкой и готов к приёму посланий через портал /analyze.
3. Вводное послание приводится к тексту и отправляется на выбор духа по случайному жребию духа Случайности; выпадение левой или правой стороны монеты определяет A или B путь.
4. Кристалл-призма (Counter) инкрементирует счётчик для выбранной версии модели по мере обработки каждого запроса, используя измерение model_version.
5. Запрос направляется к выбранному духу, который возвращает своё предсказание (label и score). Результат возвращается через портал вместе с указанной версией.
6. Второй маршрут /validate проводит самопроверку: последовательно прогоняет базовый дух A по всем записям из валидатора, подсчитывает точность и устанавливает Манометр точности ACCURACY_GAUGE значением полученной точности.
7. По итогам в ответах портала возвращаются детализированные свитки: для анализа — результат и версия; для валидации — точность, число верных ответов и общее число вопросов.

### 4.2. Declarative Content (Для конфигураций и данных)

[Указ Ткачу] Размещение конфигураций и данных в артефакте:

- Амулет с Развилкой — веб-портал на FastAPI, центральный узел ритуалов анализа.
- Дух А — первый призванный дух анализа, основанный наDistilBERT
- Дух Б — второй призванный дух анализа, основанный на RoBERTa
- Свиток VALIDATION_SET — набор примеров из validation_data, используемый для самопроверки
- Монометр ACCURACY_GAUGE — измеритель точности для валидатора
- Кристалл-счетчик REQUEST_COUNTER — счётчик обращений с разделением по модель_version
- Инструментатор Prometheus — сбор и публикация метрик
- Текстовый чертёж TextInput — входной формат данных

## 5. Structural Decomposition (Декомпозиция структуры)

- Функции и классы
  - TextInput — модель данных (Pydantic) с полем text
  - app — экземпляр FastAPI
  - model_a — пайплайн анализа настроенный на sentiment-analysis с distilbert-base-uncased-finetuned-sst-2-english
  - model_b — пайплайн анализа настроенный на sentiment-analysis с cardiffnlp/twitter-roberta-base-sentiment-latest
  - ACCURACY_GAUGE — Gauge измеряющий точность на валидации
  - REQUEST_COUNTER — Counter измеряющий число запросов по model_version
  - analyze_sentiment — обработчик маршрута POST /analyze
  - validate_model — обработчик маршрута POST /validate
  - Instrumentator().instrument(app).expose(app) — интеграция метрик Prometheus с порталом

- Основные логические блоки
  - Инициализация и призыв духов
  - Создание портала и внедрение инструментов мониторинга
  - Разделение потока запросов между версиями A и B по случайности
  - Активация и обновление счетчиков и метрик
  - Самопроверка через валидатор и обновление точности
  - Ритуал возврата данных клиенту

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** Standard CPU
- **Concurrency:** Sync endpoints (FastAPI поддерживает синхронные обработчики)
- **Dependencies:** fastapi, pydantic, transformers, prometheus_fastapi_instrumentator, prometheus_client
- В коде используются внешние артефакты и данные: validation_data.VALIDATION_SET и две модели через transformers pipeline

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT hardcode secrets в коде; используйте .env для секретов
- DO NOT выводить сырые данные в консоль в продакшене
- DO NOT использовать синхронные сетевые вызовы в критических путях ритуала
- DO NOT встраивать конфигурационные файлы .yaml/.json внутрь скриптов
- DO NOT менять версии или пути во время реконструкции артефакта
- DO NOT копировать код в локации, когда запрашивается логика; опирайтесь на концепты и описания

## 7. Verification & Testing (Верификация)

Герхин-сценарии

Feature: [Script Functionality]
Scenario: Successful analyze execution
Given the Amulet with Forks is deployed and both spirits A and B are loaded
When a valid text payload is posted to /analyze with a non-empty text
Then the response is 200 OK
And the payload contains a result object with label and score
And model_version is either A or B
And the model_requests_total metric has been incremented for the chosen version

Scenario: Invalid input for analyze
Given the Amulet is deployed
When a POST to /analyze is made with an empty or missing text field
Then the response indicates a 422 Unprocessable Entity (input validation failure)

Scenario: Validation (self-check)
Given the Amulet is deployed
When /validate is invoked
Then the response includes model_tested as A
And accuracy, correct_predictions, total_samples are present and accuracy is between 0 and 1

ИССЛЕДУЕМЫЙ АРТЕФАКТ: main.py

ИСТОЧНЫЙ КОД: (см. описание в лоре — артефакт содержит ритуалы, не копируем код здесь)
