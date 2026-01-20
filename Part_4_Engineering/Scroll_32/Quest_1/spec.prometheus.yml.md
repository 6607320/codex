# prometheus.yml Specification

## 1. Meta Information

- Domain: Infrastructure
- Complexity: Low
- Language: Go
- Frameworks: Prometheus, Docker
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Context for Creator: Этот артефакт — Скрижаль мониторинга. Она задаёт глобальный интервал сбора эхосов и миссии для Стражей метрик: за задачу fastapi-app отвечает дух app:8000. Назначение — обеспечить надёжный сбор метрик, видеть здоровье сервиса и подавать данные на панораму мониторинга.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: Text
- Format: Text
- Schema:

```ts
interface InputData {
  global: {
    scrape_interval: string; // например "15s"
  };
  scrape_configs: Array<{
    job_name: string; // например "fastapi-app"
    static_configs?: Array<{
      targets: string[]; // например ["app:8000"]
    }>;
  }>;
}
```

### 3.2. Outputs (Выходы)

- Destination: File
- Format: YAML
- Success Criteria: File Created
- Schema:

```ts
interface OutputResult {
  path: string; // например "prometheus.yml"
  success: boolean; // true/false
  message?: string; // приunts ошибок
}
```

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

Ритуал начинается с призыва к входной скрижали, из неё извлекается глобальный регламент: частота сбора, записанная как scrape_interval. Затем вызывается Летопись миссий: для каждого задания в scrape_configs читается имя задания (job_name) и доспехи цели (static_configs). Если у задания есть статические цели, извлекаются targets. Для каждого Target проверяется корректность формата и наличия хотя бы одного элемента. Затем Ткач Формул складывает итоговую конфигурацию в единый Эфир YAML, объединяя глобальные правила и списки миссий. Завершающий этап — запись этого Эфира в файл promethеus.yml, после чего возвращается результат выполнения: путь, статус и сообщение. В случае ошибок — возвращается хаос с описанием проблемы.

### 4.2. Declarative Content (Для конфигураций и данных)

Точные данные конфигурации:

- Global:
  - scrape_interval: "15s"
- Scrape_configs:
  - job_name: "fastapi-app"
    static_configs:
    - targets: ["app:8000"]

## 5. Structural Decomposition (Декомпозиция структуры)

- GlobalSection
  - поле: scrape_interval
- ScrapeConfigsSection
  - List<JobConfig>:
    - JobConfig
      - job_name
      - static_configs (опционально)
        - StaticConfig
          - targets: string[]
- Targets
  - Каждый элемент targets может быть одним или несколькими адресами
- OutputWriter
  - Преобразование структур в YAML
  - Запись в файл promеtheus.yml
- ValidationLayer
  - Проверка форматов и наличия обязательных полей
- ErrorHandler
  - Управление хаосом и уведомлениями

Инвентарь (Inventory) — как RPG-предметы:

- 🏰 Мастер-Глобал: global.scrape_interval = 15s
- 🛡️ Воин-Миссия: scrape_configs — задача "fastapi-app"
- 🗡️ Стражи-Цели: static_configs.targets = ["app:8000"]

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU, низкая нагрузка, память не критична
- Concurrency: Sync (один проход сборки конфигурации)
- Dependencies: Prometheus, YAML-складки (для записи), возможна интеграция с Docker

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap configuration files (.yaml, .json) into scripts (like Python/Bash)
- DO NOT change versions or paths during reconstruction

## 7. Verification & Testing (Верификация)

```gherkin
Feature: Prometheus config generation

  Scenario: Successful execution
    Given global.scrape_interval = "15s" and scrape_configs with job_name "fastapi-app" and targets ["app:8000"]
    When the generator runs
    Then the file "prometheus.yml" is created with valid YAML and exit status 0

  Scenario: Failure when targets are missing
    Given global.scrape_interval = "15s" and scrape_configs with job_name "fastapi-app" but missing static_configs
    When the generator runs
    Then the process fails with non-zero exit code and an error message describing missing targets
```
