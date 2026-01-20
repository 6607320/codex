# validation_data.py Specification

## 1. Meta Information

- Domain: ML/NLP
- Complexity: Low
- Language: Python
- Frameworks: None (pure Python)
- Context: ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

Легенда: этот артефакт хранит нерушимый эталон — Камень истины для тестирования и калибровки моделей: десять текстовых свитков с известными истинами-метками. Файл VALIDATION_SET будет использоваться священным артефактом validate_model.py для измерения точности духов-методов.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: N/A
- Format: N/A
- Schema:
  interface InputData {
  id?: string;
  text?: string;
  label?: string;
  }

### 3.2. Outputs (Выходы)

- Destination: N/A
- Format: JSON
- Success Criteria: N/A
- Schema:
  interface OutputResult {
  status?: 'OK' | 'ERROR';
  message?: string;
  item_count?: number;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Определяется нерушимая константа VALIDATION_SET как массив объектов.
2. Каждый объект содержит две сущности: текст (text) и метку (label), где метки принимают значения POSITIVE или NEGATIVE.
3. Набор состоит из десяти элементов, чередуя позитивные и негативные высказывания в репрезентативной форме.
4. Этот файл предназначен как эталон для проверки точности моделей; основная логика проверки находится в другом артефакте (validate_model.py).

### 4.2. Declarative Content (Для конфигураций и данных)

Это инвентарь Камня Испытаний, представлен в виде десяти свитков с текстом и ясной сутью.

- 🏰 Набор Испытаний (VALIDATION_SET): 10 записей
  - 🛡️ Запись 1: текст: "I love this product, it is absolutely amazing!"; метка: POSITIVE
  - 🛡️ Запись 2: текст: "This is the worst service I have ever received in my life."; метка: NEGATIVE
  - 🛡️ Запись 3: текст: "The movie was fantastic, a true masterpiece."; метка: POSITIVE
  - 🛡️ Запись 4: текст: "I am so disappointed with the quality, it broke after one day."; метка: NEGATIVE
  - 🛡️ Запись 5: текст: "What a wonderful experience, I would recommend it to everyone."; метка: POSITIVE
  - 🛡️ Запись 6: текст: "A complete waste of time and money, I regret buying this."; метка: NEGATIVE
  - 🛡️ Запись 7: текст: "The team was very helpful and friendly."; метка: POSITIVE
  - 🛡️ Запись 8: текст: "The food was terrible and the waiter was rude."; метка: NEGATIVE
  - 🛡️ Запись 9: текст: "An outstanding performance by the entire cast."; метка: POSITIVE
  - 🛡️ Запись 10: текст: "I will never come back to this place again."; метка: NEGATIVE

Подлинный характер Камня: этот набор служит измерительным камнем мудрости духов и подпитывает ритуал самопроверки.

## 5. Structural Decomposition (Декомпозиция структуры)

- Константа VALIDATION_SET: массив объектов; каждый объект имеет поля text и label.
- Другие функции/классы отсутствуют; файл служит исключительно как источник данных для эталона.

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Sync
- Dependencies: None

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap конфигурационные файлы (.yaml, .json) в скрипты
- DO NOT change верcии или пути во время реконструкции

## 7. Verification & Testing (Верификация)

Gherkin сценарии

Feature: Validation Data Script
Scenario: Successful loading of the validation set
Given the module validation_data.py is loaded in a clean Python environment
When the VALIDATION_SET constant is read
Then there should be 10 records with text and POSITIVE or NEGATIVE labels

Scenario: Invalid data configuration
Given the module contains an entry with a missing label
When the dataset is loaded
Then a data validation error should be reported indicating the missing label
