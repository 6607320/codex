# main.py Specification

## 1. Meta Information

- **Domain:** Infrastructure
- **Complexity:** Medium
- **Language:** Python
- **Frameworks:** FastAPI, redis
- **Context:** ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

**Context for Creator:** Легенда гласит, что этот модуль — душа простого веб-сервиса, обретшего форму в многослойном Docker-артефакте. Он служит для наглядного сравнения монолитного и многоступенчатого образа, обеспечивая минимальную точку входа и реальный функционал запоминания и воспоминания через Хранителя Памяти.  
**Instruction for AI:** Этот раздел передает WHY скрипта — зачем он нужен и какую бизнес-задачу решает.

Этот модуль предоставляет простой веб-сервис для сохранения и извлечения значений по ключу через Redis, чтобы демонстрировать работу хранилища и точки входа в приложении в контексте Docker-мульти-этапности.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** API Request
- **Format:** Text (Query-параметры) / JSON (при расширении)
- **Schema:**
  interface InputData {
  key: string;
  value: string;
  }

### 3.2. Outputs (Выходы)

- **Destination:** API Response
- **Format:** JSON
- **Success Criteria:** 200 OK
- **Schema:**
  interface OutputResult {
  status: "ok" | "not_found" | "found";
  key: string;
  remembered_value?: string;
  recalled_value?: string;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Входная точка: приложение инициализирует окружение, считывает переменную REDIS_HOST и по умолчанию устанавливает localhost, формируя адрес Хранителя Памяти.
2. Создается магический канал к Хранителю Памяти с помощью Redis-клиента, который общается по порту 6379 и базе 0; ответы декодируются в строки.
3. Врата запоминания открываются через маршрут POST /remember/{key}. При обращении эта руна отдает приказ Хранителю: сохранить значение value под ключом key; затем возвращает подтверждение и запомненное значение.
4. Врата вспоминания открываются через маршрут GET /recall/{key}. Презирающий ветер Хранителя глядит в эфир и возвращает either найденное значение под ключом или сообщение о том, что воспоминание не найдено.
5. Жизнь сервиса зависит от жизни Хранителя памяти: если Хранитель не отвечает, отклик нулевой или исключения — эта связка отражается в ответах и поведении endpoints.
6. Весь поток построен вокруг одного-единственного клиента Redis, созданного при импорте модуля и разделяемого между запросами.

### 4.2. Declarative Content (Для конфигураций и данных)

[Сюда Инфо о конфигурациях и точных данных не вносится кодом; описано выше как часть архитектуры.]

## 5. Structural Decomposition (Декомпозиция структуры)

- app: экземпляр FastAPI
- r: Redis-клиент (redis.Redis)
- remember_value: функция-обработчик для POST /remember/{key}
- recall_value: функция-обработчик для GET /recall/{key}
- Конфигурация окружения: REDIS_HOST (с дефолтом "localhost"), порт 6379, db 0, decode_responses=True

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** Стандартный режим для микросервисов; рассчитан на типичные серверные конфигурации.
- **Concurrency:** Синхронные обработчики (не async) в рамках FastAPI; поддержка одновременных запросов через ASGI-сервер.
- **Dependencies:** FastAPI, redis (redis-py)

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде (использовать переменные окружения/.env).
- DO NOT печатать сырые данные в консоль в продакшн-режиме.
- DO NOT выполнять синхронные сетевые вызовы внутри главного цикла без надобности.
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты.
- DO NOT менять версии библиотек или пути во время реконструкции артефакта.

## 7. Verification & Testing (Верификация)

```gherkin
Feature: Script Functionality
  Scenario: Successful execution
    Given Redis доступен по адресу из REDIS_HOST (или localhost)
    When I POST to /remember/{key} with value "exampleValue"
    Then I receive 200 OK
    And the response contains status "ok", key, and remembered_value "exampleValue"

  Scenario: Recall existing key
    Given a value stored under a specific key
    When I GET /recall/{key}
    Then I receive 200 OK
    And the response contains status "found", key, and recalled_value equals the stored value

  Scenario: Recall missing key
    Given no value exists for a given missingKey
    When I GET /recall/{missingKey}
    Then I receive 200 OK
    And the response contains status "not_found" and key
```

ИЗОЛОЧЕНИЕ АРТЕФАКТА: main.py

ИССЛЕДУЕМЫЙ АРТЕФАКТ: main.py
ИСХОДНЫЙ КОД: описан выше в легенде и в технических деталях.
