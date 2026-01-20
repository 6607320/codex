# Dockerfile Specification

## 1. Meta Information

- **Domain:** Infrastructure
- **Complexity:** Medium
- **Language:** Bash
- **Frameworks:** Docker
- **Context:** Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

**Context for Creator:** Монолитный сервис на Python упакован в толстый образ: он копирует зависимости и код через библиотеки.list и main.py, устанавливает зависимости и настраивает запуск через uvicorn.
**Instruction for AI:** This section provides high-level intent. Use it to understand the "WHY" of the code.
Описание на русском языке: этот Dockerfile предназначен для разворачивания монолитного API в самодостаточном образе. Он стартует с тяжелого Python образа, устанавливает зависимости из libraries.list, копирует код из main.py и запускает uvicorn на порту 8000.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** CLI Args
- **Format:** Text
- **Schema:**
  interface InputData {
  dockerfilePath: string;
  contextDir?: string;
  librariesListPath?: string;
  mainScript?: string;
  baseImage?: string;
  buildArgs?: Record<string, string>;
  noCache?: boolean;
  }

### 3.2. Outputs (Выходы)

- **Destination:** STDOUT | Image ID
- **Format:** JSON
- **Success Criteria:** Exit code 0
- **Schema:**
  interface OutputResult {
  imageId: string;
  imageTag?: string;
  success: boolean;
  logs?: string[];
  error?: string;
  buildTimeMs?: number;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Начни с базового толстого образа Python:3.10.
2. Установи рабочую директорию /app.
3. Скопируй файл libraries.list в рабочий контекст образа.
4. Выполни установку зависимостей командой pip install —no-cache-dir -r libraries.list.
5. Скопируй код из main.py в образ.
6. Задай команду запуска на запуск uvicorn с параметрами: main:app, хост 0.0.0.0 и порт 8000.
7. Образ строится как самодостаточный и в финальном артефакте может сохранять мусор сборки, характерный для толстого образа.
8. Результатом становится готовый образ, который можно запустить в контейнере и получить сервис на порту 8000.

### 4.2. Declarative Content (Для конфигураций и данных)

- Базовый образ: python:3.10
- Рабочая директория: /app
- Исходники и рецепты: libraries.list, main.py
- Команда установки: pip install --no-cache-dir -r libraries.list
- Команда запуска: uvicorn main:app --host 0.0.0.0 --port 8000
- Примечание: образ не содержит явного EXPOSE в файле, поэтому порт может потребовать конфигурацию на стороне оркестратора.

## 5. Structural Decomposition (Декомпозиция структуры)

- FROM python:3.10 — базовый фундамент
- WORKDIR /app — пространство работы
- COPY libraries.list — перенесение рецептов зависимостей
- RUN pip install --no-cache-dir -r libraries.list — алхимия установки пакетов
- COPY main.py — копирование кода монолита
- CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] — призыв к запуску сервиса

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** Standard CPU
- **Concurrency:** Sync
- **Dependencies:** python:3.10, pip, uvicorn, libraries.list, main.py

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (используй .env или секреты оркестратора)
- DO NOT выводить сырые данные в консоль в продакшн-режиме
- DO NOT использовать синхронные сетевые вызовы в основном рабочем ходе приложения
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты
- DO NOT менять версии образов или пути в процессе реконструкции

## 7. Verification & Testing (Верификация)

1. Гхеркин сценарий: успешная сборка и запуск
   Feature: Dockerfile монолитный образ
   Scenario: Successful build and run
   Given репозиторий содержит Dockerfile, libraries.list и main.py
   When выполняется docker build -t monolith-app .
   Then выполняется docker run -p 8000:8000 monolith-app
   And сервис доступен на http://localhost:8000

2. Гхеркин сценарий: ошибка сборки из-за отсутствия libraries.list
   Feature: Dockerfile монолитный образ
   Scenario: Build fails due to missing libraries.list
   Given libraries.list отсутствует
   When выполняется docker build -t monolith-app .
   Then сборка завершается с не нулевым кодом выхода и логами об отсутствии файла

ИЗНАННЫЙ АРТЕФАКТ: Dockerfile
ИСТОЧНЫЙ КОД: предоставленные выше данные о толстом образе содержат этапы копирования библиотек, установки зависимостей и запуска uvicorn в образе.
