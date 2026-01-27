# docker-compose.yml Specification

## 1. Meta Information

- **Domain:** Infrastructure
- **Complexity:** Medium
- **Language:** TypeScript
- **Frameworks:** Docker
- **Context:** Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Легенда: этот Ритуал Оркестрации служит Скрижалью для проверки прочности системы через две силы: Целевое приложение target-app и Легион Locust. Цель — развернуть и связать две службы, чтобы запустить нагрузочное испытание и получить количественный отклик о устойчивости API.

Context for Creator: Эти две службы образуют дуэт для проверки под нагрузкой: целевое приложение создаёт стену, Locust создает волну запросов на фоне, чтобы имитировать реальную рабочую нагрузку.

Instruction for AI: This section provides high-level intent. Use it to understand the "WHY" of the code.

Описание на русском языке: Ритуал нацелен на автоматизацию развёртывания двух компонентов и синхронной их работы: целевой сервис через образ и контекст сборки, а также тестовый агент Locust, который направляет трафик на целевой сервис, чтобы измерить устойчивость и пропускную способность under load.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** CLI Args
- **Format:** Text
- **Schema:**
  - InputData
    - version: string
    - services: Record<string, ServiceSpec>
  - ServiceSpec
    - build?: BuildSpec
    - image?: string
    - ports?: string[]
    - hostname?: string
    - depends_on?: string[]
    - command?: string
  - BuildSpec
    - context?: string
    - dockerfile?: string

### 3.2. Outputs (Выходы)

- **Destination:** File
- **Format:** JSON
- **Success Criteria:** Exit Code 0
- **Schema:**
  - OutputResult
    - valid: boolean
    - version?: string
    - services?: string[]
    - errors?: string[]

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Загружаем Скрижаль docker-compose.yml с диска.
- Распарсиваем её содержимое в внутренний эфир представления.
- Проверяем главное: наличие версии и секции services.
- Подтверждаем что версия равна 3.8.
- Для каждого духа сервиса проверяем поля:
  - target-app: если есть build, удостоверяемся, что context указывает на существующий путь ../../Scroll_29/Quest_2; образ image существует как codex/app-29-2-telemetry; порты содержат 8000:8000; имя хоста соответствует target-app.
  - locust-tester: сборка через текущий контекст; образ codex/locust-30-1-tester; порты 8089:8089; команда запуска locust с указанием файла и цели host http://target-app:8000; зависимость от target-app.
- Сообщаем о корректности или собираем Хаос (ошибки) с указанием недочётов.
- При отсутствии ошибок записываем итоговый отчёт в файл в формате JSON и возвращаем код успеха.

### 4.2. Declarative Content (Для конфигураций и данных)

- Версия Скрижали: 3.8
- Службы:
  - target-app
    - build.context: ../../Scroll_29/Quest_2
    - image: codex/app-29-2-telemetry
    - ports: 8000:8000
    - hostname: target-app
  - locust-tester
    - build.context: .
    - image: codex/locust-30-1-tester
    - ports: 8089:8089
    - command: locust -f /app/locustfile.py --host http://target-app:8000
    - depends_on: [ target-app ]

## 5. Structural Decomposition (Декомпозиция структуры)

- Top level: version, services
- services
  - target-app
    - build: context, (опционально dockerfile)
    - image
    - ports
    - hostname
  - locust-tester
    - build: context
    - image
    - ports
    - command
    - depends_on

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** Standard CPU/Memory; ориентировано на локальные тесты через Docker.
- **Concurrency:** Небольшая кооперативная схема; два сервиса, последовательный запуск через depends_on.
- **Dependencies:** Docker Engine и поддержка Docker Compose версии 3.8; образы codex/app-29-2-telemetry и codex/locust-30-1-tester; путь ../../Scroll_29/Quest_2 должен существовать.

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в явном виде в этой Скрижали.
- DO NOT печатать сырые данные в консоль в продакшн-режиме.
- DO NOT выполнять синхронные сетевые вызовы в блоках, где это опасно для задержек.
- DO NOT оборачивать конфигурации (.yaml, .json) в скрипты.
- DO NOT менять версию или пути во время реконструкции без осмысленного обоснования.

## 7. Verification & Testing (Верификация)

### Гჰеркин-сценарии

Feature: Docker Compose Specification

Scenario: Valid docker-compose.yml is parsed and reported as valid
Given the docker-compose.yml exists at repository root with version 3.8 and two services
When the specification processor validates the manifest
Then the result is valid and an output JSON report is generated

Scenario: Error when required field is missing or path invalid
Given the docker-compose.yml has target-app with a missing or invalid build.context
When the specification processor validates the manifest
Then an error is reported and a non-zero exit code is returned

ИССЛЕДУЕМЫЙ АРТЕФАКТ: docker-compose.yml

ИСХОДНЫЙ КОД РИТУАЛА ЛЕГЕНДЫ

- Версия Скрижали: 3.8
- Службы:
  - target-app
    - build.context: ../../Scroll_29/Quest_2
    - image: codex/app-29-2-telemetry
    - ports: 8000:8000
    - hostname: target-app
  - locust-tester
    - build.context: .
    - image: codex/locust-30-1-tester
    - ports: 8089:8089
    - command: locust -f /app/locustfile.py --host http://target-app:8000
    - depends_on: target-app

Эти чертежи позволят Великану проверить протечку времени через две силы — целевой сервис и тестовый легион Locust — и зафиксировать результат в виде Эфира-отчета.
