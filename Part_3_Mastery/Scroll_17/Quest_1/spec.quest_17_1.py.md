# quest_17_1.py Specification

## 1. Meta Information

- **Domain:** ML/NLP
- **Complexity:** Medium
- **Language:** Python
- **Frameworks:** FastAPI, Pydantic, transformers
- **Context:** ../AGENTS.md
- **Примечание по атмосфере:** Ритуал запуска веб-портала для анализа эмоций текста через артефакт-пайплайн.

## 2. Goal & Purpose (Цель и Назначение)

- **Context for Creator:** Бизнес-задача — выдать онлайн-сервис, принимающий текст и возвращающий его эмоциональную окраску через модель анализа тональности. Этот модуль превращает локальный AI-скрипт в веб-сервис и предоставляет фронт-слой взаимодействия с текстовым эфиром пользователей.
- **Instruction for AI:** Этот раздел задаёт WHY: создать устойчивый HTTP-API для анализа сентиментальности с использованием заранее обученного трансформер-пайплайна и вернуть структурированный результат в формате JSON.

Описание: Этот файл реализует портал Amulet of TechnoMancer — веб-сервис на FastAPI, который принимает текст через POST-запрос и возвращает результат анализа sentiment-analysis с использованием distilbert-base-uncased-finetuned-sst-2-english. Сквозной эффект — показать, как локальный AI-скрипт может работать в production-like окружении через веб-слой.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** API Request
- **Format:** JSON
- **Schema:**
  interface InputData {
  text: string;
  }

### 3.2. Outputs (Выходы)

- **Destination:** API Response
- **Format:** JSON
- **Success Criteria:** 200 OK
- **Schema:**
  interface OutputResult {
  result: {
  label: string;
  score: number;
  };
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. При старте инициализируется дух портала: создаётся пайплайн для сентимент-анализа с моделью по умолчанию distilbert-base-uncased-finetuned-sst-2-english.
2. Определяется чертеж послания TextInput с одним полем text типа строка.
3. Сотворяется портал FastAPI с мистическим именем и описанием.
4. На пороге порта открываются врата /analyze, которые принимают POST-запрос. Входные данные валидируются через TextInput.
5. Из полученного текста извлекается текстовое сообщение и отправляется в дух анализа для обработки.
6. Результат анализа, который возвращает пайплайн, подготавливается и возвращается как JSON в виде поля result, содержащее метку и вероятность.
7. В чистом виде скрипт не запускает сервис сам по себе; для пробуждения портала требуется запустить uvicorn quest_17_1:app --reload.

### 4.2. Declarative Content (Для конфигураций и данных)

- Архитектура портала:
  - Название портала: Говорящий Амулет Техноманта
  - Описание: Магический портал, который определяет эмоциональную окраску текста
  - Версия: 1.0
  - Врата: POST /analyze
- Чертежи и духи:
  - Дух анализа: sentiment_analyzer, пайплайн с задачей sentiment-analysis и моделью distilbert-base-uncased-finetuned-sst-2-english
  - Послание: TextInput с полем text
- Взаимодействие:
  - Ввод: текст в формате JSON
  - Вывод: JSON с полем result, содержащим label и score

## 5. Structural Decomposition (Декомпозиция структуры)

- Классы
  - TextInput: Pydantic модель с полем text: string
- Функции
  - analyze_sentiment: обработчик POST /analyze, валидирует вход, вызывает sentiment_analyzer и возвращает первый элемент результата
- Глобальные сущности
  - app: экземпляр FastAPI с названием, описанием и версией
  - sentiment_analyzer: пайплайн анализа сентимента
- Импортируемые элементы
  - FastAPI, BaseModel, pipeline

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Runtime/Environment:** Python среда с установленными FastAPI, Pydantic, transformers
- **Performance:** стандартный CPU; возможно ускорение на GPU, но не обязательно
- **Concurrency:** синхронный обработчик запроса (endpoint может работать в контексте асинхронности FastAPI, но реализация синхронная)
- **Dependencies:** fastapi, pydantic, transformers
  - Модель и пайплайн задаются через distilbert-base-uncased-finetuned-sst-2-english
- **Необходимые права доступа:** сеть для входящих HTTP-запросов
- **Портативность:** портал не начинает работу автоматически; для запуска нужен uvicorn

### 6.2. Prohibited Actions (Negative Constraints)

- НЕ хранить секреты в открытом виде
- НЕ выводить сырые данные в консоль в боевом режиме
- НЕ использовать синхронные сетевые вызовы в основном цикле, если планируется масштабирование
- НЕ оборачивать конфигурационные файлы YAML/JSON в скрипты
- НЕ менять версии или пути в процессе реконструкции artefact

## 7. Verification & Testing (Верификация)

### Герхин-сценарии (1–2 сценария счастья и один негатив)

Формат:
Feature: [Script Functionality]
Scenario: Successful execution
Given [Preconditions]
When [Action is taken]
Then [Expected result]

Примеры:

Feature: Sentiment Analysis Portal
Scenario: Successful execution
Given валидный JSON-пейлоад с текстом "I am thrilled with technology"
When отправлен POST запрос к /analyze
Then возвращается HTTP статус 200 и поле result содержит label и score

Scenario: Missing text field
Given JSON без поля text
When отправлен POST запрос к /analyze
Then возвращается HTTP статус 422 Unprocessable Entity

Scenario: Invalid text type
Given JSON поле text имеет неверный тип (например число)
When отправлен POST запрос к /analyze
Then возвращается HTTP статус 422 Unprocessable Entity
