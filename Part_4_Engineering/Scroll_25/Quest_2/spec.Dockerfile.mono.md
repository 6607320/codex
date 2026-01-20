# Dockerfile.mono Specification

## 1. Meta Information

- Domain: Infrastructure
- Complexity: Medium
- Language: Bash
- Frameworks: Docker
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Context for Creator: Опишите бизнес-задачу кратко. Что делает этот модуль?

- Это одностадийный ритуал сборки образа Docker для Python-сервиса. Он берет базовый образ python:3.10, подготавливает рабочее пространство /app, копирует файл libraries.list, устанавливает зависимости из него, копирует исходный код и задаёт точку входа через uvicorn для запуска на порту 8000. Ритуал призван сделать воспроизводимый и повторяемый образ, готовый к развёртыванию в окружении, где нужен Web-сервис на uvicorn.

Instruction for AI: This section provides high-level intent. Use it to understand the "WHY" of the code.

- Легенда и намерение: создать минимальный, воспроизводимый окружной контейнер для запуска сервиса на uvicorn, используя предписанный список зависимостей и исходники, и обеспечить запуск через стандартную команду uvicorn.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: Text
- Schema:
  - baseImage: string // например, "python:3.10"
  - workDir: string // например, "/app"
  - librariesListPath: string // например, "libraries.list"
  - codeContextPath: string // например, "."
  - cmd: string[] // например, ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

### 3.2. Outputs (Выходы)

- Destination: API Response
- Format: JSON
- Success Criteria: Exit Code 0
- Schema:
  - imageName: string
  - imageTag: string
  - status: "success" | "failure"
  - durationMs?: number
  - log: string[]
  - error?: string

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Ритуал начинается с призыва базового образа: берётся образ python:3.10 как основа.
- Затем устанавливается рабочий лDesk — рабочая директория в образе становится /app.
- Затем копируется файл libraries.list в корневое место внутри образа, чтобы он стал артефактом установки зависимостей.
- Далее выполняется заклинание установки зависимостей: pip install --no-cache-dir -r libraries.list. Это превращает эфир зависимостей в устойчивый набор пакетов внутри образа.
- После этого копируются остальные артефакты кода: копирование текущего контекста проекта внутрь образа (обычно . на стороне сборки → /app внутри образа).
- Наконец активируется команда запуска сервиса: CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"].
- Весь этот ритуал сопровождается выводом логов в процессе сборки. В случае ошибок на любом этапе сборки формируется хаос и возвращается детальное сообщение об ошибке.

### 4.2. Declarative Content (Для конфигураций и данных)

- Ритуал опирается на «толстый» образ и превращает мусор от сборки в финальный образ, а не в временный артефакт. Путь /app становится центром силы. Файл libraries.list — артефакт скрижалей, откуда читаются зависимости. Команда запуска подготавливает врата приложения, чтобы uvicorn слушал на 0.0.0.0:8000.

## 5. Structural Decomposition (Декомпозиция структуры)

- Блок BaseImage
- Блок WorkingDirectory
- Блок CopyLibraries
- Блок DependencyInstall
- Блок CopyCode
- Блок Entrypoint/Command

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Async (uvicorn в основе, поддерживает асинхронную обработку)
- Dependencies: Python 3.10 образ, pip, uvicorn, зависимости из libraries.list

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap configuration files (.yaml, .json) into scripts
- DO NOT change versions or paths during reconstruction

## 7. Verification & Testing (Верификация)

```
Feature: Dockerfile.mono functionality
  Scenario: Successful execution
    Given имаются корректные рабочие файлы и libraries.list
    When сборка и запуск через ритуал проходят без ошибок
    Then статус = "success", imageName и imageTag заполняются, log содержит вывод сборки

  Scenario: Missing libraries.list
    Given libraries.list отсутствует
    When сборка запускается
    Then статус = "failure", log содержит сообщение об отсутствии libraries.list, error описан и возвращается соответствующий хаос
```

ИССЛЕДУЕМЫЙ АРТЕФАКТ: Dockerfile.mono

ИСХОДНЫЙ КОД:

- Одностадийный билд на основе python:3.10
- WORKDIR /app
- COPY libraries.list .
- RUN pip install --no-cache-dir -r libraries.list
- COPY . .
- CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

Примечание по стилю: скрипт называется ритуалом; конфигурация — Скрижаль; данные — Эфир; ошибки — Хаос. Вся логика описана словами, без копирования кода. Входные и выходные интерфейсы заданы как TS-представления, чтобы интегрировать этот артефакт в систему автоматизированной сборки и тестирования.
