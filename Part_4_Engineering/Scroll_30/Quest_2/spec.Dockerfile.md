# Dockerfile Specification

## 1. Meta Information

- Domain: Infrastructure
- Complexity: Medium
- Language: Bash
- Frameworks: Docker, Python, uvicorn, FastAPI
- Context: Independent Artifact

Lore notes: Этот артефакт — ритуал для призыва и содержания жизнеспаспорта Python-приложения внутри контейнера. Скрижаль библиотеки и Свиток зависимостей превращаются в Эфир, а запуск стартового скрипта открывает портал к сервису.

## 2. Goal & Purpose (Цель и Назначение)

Context for Creator: Виртуозный модуль превращает набор зависимостей из libraries.list в живой образ, копирует ВСЕ пергаменты проекта, делает стартовый скрипт исполняемым и запускает сервис через uvicorn. Цель — обеспечить надёжный, переносимый контейнер для Python-приложения, слушающего на 0.0.0.0 и порту, указанному через PORT.

Instruction for AI: This section provides high-level intent. Use it to understand the "WHY" of the code.
Описание на русском языке: Этот Dockerfile собирает и запускает веб-приложение как единый артефакт, пригодный к развёртыванию в любом окружении. Он обеспечивает воспроизводимый образ, управляет зависимостями через библиотею libraries.list и запускает сервер через стартовый скрипт, чтобы артефакт ожил в мирe контейнеров.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args | STDIN | API Request
- Format: JSON | Text
- Schema:
  interface InputData {
  source: string; // например "CLI" или "API"
  payload: any;
  env?: Record<string, string>;
  }

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: Text
- Success Criteria: Exit Code 0
- Schema:
  interface OutputResult {
  success: boolean;
  message?: string;
  imageId?: string;
  logs?: string;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Шаг 1: База — выбирается прочный фундамент: образ Python 3.10. Этот выбор гарантирует совместимость и надёжность, как крепостной камень для башни.
- Шаг 2: Устанавливается рабочая площадь внутри контейнера: рабочая директория устанавливается в /app.
- Шаг 3: Свиток зависимостей копируется в образ: libraries.list передаётся в собранный контекст.
- Шаг 4: Зависимости разворачиваются напрямую в финальный образ командой pip install --no-cache-dir -r libraries.list.
- Шаг 5: Все пергаменты проекта копируются внутрь образа, чтобы артефакт имел полный свод файлов.
- Шаг 6: Скрипт start.sh становится исполняемым с помощью разрешения на запуск.
- Шаг 7: Главный заклинательный элемент (CMD) указывает системе исполнить start.sh при запуске контейнера.
- Шаг 8: Внутри start.sh запускается уикорн (uvicorn), чтобы призвать артефакт app из главного скрипта/модуля, слушать на адресе 0.0.0.0 и порт, заданный PORT, тем самым открывая портал для мира.

### 4.2. Declarative Content (Для конфигураций и данных)

- Скрижаль образа: python:3.10
- Рабочая площадь: /app
- Свиток зависимостей: libraries.list
- Пергаменты проекта: содержимое текущего каталога копируется в образ
- Руна исполнения: start.sh становится исполняемым
- Главный заклятие запуска: CMD "./start.sh" (пробуждает сервис uvicorn)

## 5. Structural Decomposition (Декомпозиция структуры)

- Блок “FROM”: Базовый образ Python 3.10 — фундамент артефакта.
- Блок “WORKDIR”: Устанавливает рабочую локацию внутри контейнера.
- Блок “COPY libraries.list”: Приводит свиток зависимостей в мир контейнера.
- Блок “RUN pip install”: Превращает скрижаль зависимостей в функциональный эфир.
- Блок “COPY . .”: Перемещает все пергаменты проекта внутрь чаши.
- Блок “RUN chmod +x ./start.sh”: Дарует стартовому скрипту право на жизнь.
- Блок “CMD [\"./start.sh\"]”: Пробуждает артефакт при старте контейнера.

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Async ( uvicorn запускается как асинхронный сервер )
- Dependencies: Python 3.10, uvicorn, FastAPI (как предполагаемые зависимости внутри libraries.list)

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap configuration files (.yaml, .json) into scripts (like Python/Bash)
- DO NOT change versions or paths during reconstruction

## 7. Verification & Testing (Верификация)

### Герхин-сценарии

Feature: Dockerfile Build and Run
Scenario: Successful build and startup
Given a repository with a valid libraries.list and start.sh
When docker build -t app:latest .
And docker run -e PORT=8000 -p 8000:8000 app:latest
Then the container starts a server reachable at http://localhost:8000/health and returns 200 OK.

Scenario: Build fails due to missing dependencies
Given libraries.list points to missing packages
When docker build is attempted
Then the build should fail with a non-zero exit code.

ИССЛЕДУЕМЫЙ АРТЕФАКТ: Dockerfile
ИСХОДНЫЙ КОД: описано выше в ритуалах и структурах артефакта.
