# prometheus.yml Specification

## 1. Meta Information

- Domain: Infrastructure
- Complexity: Medium
- Language: Go
- Frameworks: Docker, Prometheus (Skryzial и Свиток Мониторинга)
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Context for Creator: Этот Ритуал описывает Скрижаль мониторинга для приложения FastAPI. Он задаёт глобальные параметры и конкретный набор охранных задач (метрик) для стража-Летописца, чтобы он опрашивал целевую сущность и собирал Эфир (метрики) через сотворённый путь /metrics на порту 8000.
Instruction for AI: Этот раздел объясняет WHY — зачем нужен этот Ритуал и какую бизнес-задачу он обслуживает: надёжный сбор метрик и наблюдаемость сервиса.

Описание на русском языке: Скрижаль prometheus.yml задаёт интервал сбора метрик (15s) и конфигурацию наблюдения за fastapi-app через целевой узел app:8000, обеспечивая единый поток Эфира для последующей диагностики и алертинга.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: File
- Format: YAML
- Schema:
  InputData:
  - path: string // путь к Скрижали (prometheus.yml)
  - content: string // содержание Скрижали (Эфир)
  - metadata?: Record<string, string> // дополнительные метаданные

### 3.2. Outputs (Выходы)

- Destination: File
- Format: YAML
- Success Criteria: File Created
- Schema:
  OutputResult:
  - success: boolean
  - outputPath?: string
  - message?: string
  - diagnostics?: string[]

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Входной Эфир читается из InputData.path; если файл недоступен — поднимается хаос с диагностикой.
2. Точный Разбор Эфира: распознаётся структура Скрижали в формате YAML, чтобы извлечь раздел global и свиток scrape_configs.
3. Верификация глобального параметра: наличие scrape_interval; если отсутствует — регистрируем предупреждение и переходим к безопасному умолчанию, если политика позволяет.
4. Верификация свитков слежения: проверяем наличие списка scrape_configs и их целевые элементы.
5. Сценарий проверки первого элемента scrape_configs: должно быть job_name равное "fastapi-app" и static_configs с targets, содержащим "app:8000".
6. Если проверки прошли успешно — возвращаем успешный Эфир и сохраняем обновлённую Скрижаль в указанный путь вывода; если нет — формируем Хаос-диагностику и возвращаем соответствующий результат.
7. Логика завершения: отправляем OutputResult с флагом успех и текстовыми диагностическими сообщениями; если требуется — обновляем файл; иначе просто возвращаем статус.

### 4.2. Declarative Content (Для конфигураций и данных)

- Изначальный Эфир: global:
  scrape_interval: 15s

- Свиток Обязанностей (scrape_configs):
  - job_name: "fastapi-app"
    static_configs:
    - targets: ["app:8000"]

- Контекстная структура Свитка: Цель — направлять Стража к точке /metrics на порту 8000 для существа с именем app.

## 5. Structural Decomposition (Декомпозиция структуры)

- Классы и функции:
  - class PrometheusYamlRitual (главный исполнитель Скрижали)
  - function loadFile(path: string): string
  - function parseYaml(text: string): YamlAst
  - function validateScrapeConfig(ast: YamlAst): ValidationResult
  - function renderOutput(result: OutputResult): void
  - function reportDiagnostics(diagnostics: string[]): void

- Для конфигураций и данных (если расширение в будущем):
  - Module: YamlParser
  - Module: ScrapeConfigValidator
  - Module: OutputWriter

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Async
- Dependencies: yaml/ruamel.yaml (в зависимости от реализации), fs/promises (или эквивалент в выбранном стеке), process/env для конфигураций

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде (используйте .env, если нужны ключи).
- DO NOT выводить сырые данные в консоль в продакшн-режиме.
- DO NOT использовать синхронные сетевые вызовы в основном цикле.
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты.
- DO NOT менять версии библиотек или пути реконструкции без документации.

## 7. Verification & Testing (Верификация)

Геркин-сценарии:

Feature: Prometheus YAML Validator
Scenario: Successful validation
Given a prometheus.yml with global: scrape_interval: 15s and scrape_configs: - job_name: "fastapi-app" with static_configs: - targets: ["app:8000"]
When the ritual validates the Скрижаль
Then the result is success and the Скрижаль is saved to the output path with no Хаос

Scenario: Missing required fields
Given a prometheus.yml lacking scrape_configs
When the ritual validates the Скрижаль
Then the result is failure and diagnostics describe missing scrape_configs and potential remedies

ИССЛЕДУЕМЫЙ АРТЕФАКТ: prometheus.yml

ИСХОДНЫЙ КОД:
Мы начинаем главу 'Великий Указ' (global), где устанавливаем общие, незыблемые законы для нашего 'Стража-Летописца'.
global:

# Мы приказываем 'Стражу' заглядывать в мир и опрашивать своих подопечных (собирать метрики) каждые 15 секунд.

scrape_interval: 15s

# Здесь начинается 'Свиток Обязанностей' (scrape_configs), где мы перечисляем, за кем именно должен следить наш 'Страж'.

scrape_configs:

# Мы начинаем первую запись в 'Свитке' и даем этому заданию ('job_name') имя 'fastapi-app',

# чтобы потом легко понимать, откуда пришли летописи (метрики).

- job_name: "fastapi-app"
  # Мы объявляем, что будем использовать 'Пакт Прямого Наблюдения' (static_configs),
  # указывая точные координаты цели вручную.
  static_configs:
  # Это — финальный, самый точный приказ. Мы говорим 'Стражу':
  # "Направь свой взор на существо по имени `app` (наш 'Говорящий Амулет')
  # и загляни в его 'Окно Наблюдения' (`/metrics`) на порту `8000`".
  # Магия 'Магической Сети' (Docker-Compose) сама укажет 'Стражу' путь к 'app'.
  - targets: ["app:8000"]
