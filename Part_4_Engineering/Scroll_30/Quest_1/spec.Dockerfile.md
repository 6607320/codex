# Dockerfile Specification

## 1. Meta Information

- Domain: Infrastructure
- Complexity: Low
- Language: Bash
- Frameworks: Docker, Locust
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

**Context for Creator:** Этот артефакт — Ритуал сборки легковесного окружения для нагрузки с Locust. Он превращает исходный набор сета в единый образ, который можно развернуть и запустить для тестирования.  
**Instruction for AI:** Этот раздел раскрывает WHY этого кода: зачем нужен Dockerfile и как он обеспечивает воспроизводимое окружение для нагрузочного тестирования.

Данный Dockerfile обеспечивает сборку образа на основе Python 3.10-slim, копирует файл зависимостей libraries.list и Locust-скрипт locustfile.py, устанавливает зависимости и подготавливает готовый к запуску контейнер для нагрузочного тестирования.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: JSON
- Schema:
  - baseImage: string — базовый образ (например, "python:3.10-slim")
  - workDir: string — рабочая директория внутри образа (например, "/app")
  - libraryListPath: string — путь к списку зависимостей (например, "libraries.list")
  - locustfilePath: string — путь к Locust-скрипту (например, "locustfile.py")

### 3.2. Outputs (Выходы)

- Destination: API Response
- Format: JSON
- Success Criteria: imageId не пуст; статус операции успешен
- Schema:
  - imageId: string
  - imageName?: string
  - builtAt?: string
  - sizeMB?: number
  - status: "success" | "failure"
  - log?: string

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Ритуал начинается с призыва базового образа, заданного baseImage, например python:3.10-slim, и установки WORKDIR в указанную рабочую зону.
- Затем через клятву Копирования получает свиток зависимостей — libraries.list — и помещает его в рабочую директорию внутри образа.
- Примем решение: чародейские чанты pip устанавливают все зависимости из списка без кеша, чтобы ритуал был чистым и воспроизводимым.
- Далее копируется Locust-скрипт (locustfile.py) в рабочую директорию, чтобы манифест тестирования был готов к запуску.
- В завершение образ завершает свое формирование и возвращает готовый артефакт — Docker-образ, который можно запустить для нагрузочного тестирования.
- Важное: все команды выполняются в контексте единого слоя сборки, чтобы минимизировать хаос и сохранить ясность конфигурации.

### 4.2. Declarative Content (Для конфигураций и данных)

- Базовый образ: python:3.10-slim
- Рабочий каталог: /app
- Зависимости: libraries.list
- Тестовый скрипт: locustfile.py
- Входные файлы: Libraries и Locust размещаются в корневом контексте сборки
- Ритуал установки: pip install -r libraries.list
- Финальный артефакт: сборочный образ, готовый к запуску

## 5. Structural Decomposition (Декомпозиция структуры)

- FROM: базовый образ Python 3.10-slim
- WORKDIR: установка рабочей директории /app
- COPY libraries.list: перенос списка зависимостей
- RUN pip install --no-cache-dir -r libraries.list: установка зависимостей
- COPY locustfile.py: перенос Locust-скрипта
- (В контексте) Итог: сформированный Docker-образ, готовый к запуску

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Sync
- Dependencies: Docker, Python 3.10, pip, зависимости из libraries.list, Locust

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в явном виде внутри образа (используйте .env или секретные механизмы сборки).
- DO NOT выводить сырые данные в консоль в продакшне.
- DO NOT выполнять сетевые вызовы синхронно в критических участках цикла сборки.
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) внутрь скриптов.
- DO NOT менять версии базового образа или пути в процессе реконструкции.

## 7. Verification & Testing (Верификация)

Геркин-сценарии

Feature: Dockerfile Build
Scenario: Successful build
Given сборочный контекст содержит базовый образ, libraries.list и locustfile.py
When выполняется сборка образа через Dockerfile
Then сборка завершается успешно (exit code 0) и возвращается идентификатор образа

Scenario: Failure due to missing libraries.list
Given библиотеки не размещены в контексте (libraries.list отсутствует)
When выполняется сборка образа
Then сборка завершается с не нулевым кодом и лог содержит упоминание о пропавшем файле библиотек

ИССЛЕДУЕМЫЙ АРТЕФАКТ: Dockerfile
ИСХОДНЫЙ КОД: Как выше, Ритуал сборки легковесного окружения для Locust.
