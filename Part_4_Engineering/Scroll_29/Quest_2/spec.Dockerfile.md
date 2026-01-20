# Dockerfile Specification

## 1. Meta Information

- **Domain:** Infrastructure
- **Complexity:** Medium
- **Language:** Bash
- **Frameworks:** Docker
- **Context:** Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

**Context for Creator:** Этот артефакт описывает создание контейнерного образа для питоновского сервиса: задаёт базовый образ, рабочий каталог, копирует список зависимостей, устанавливает пакеты через pip, копирует все файлы проекта, даёт исполнение стартовому скрипту и запускает его.  
**Instruction for AI:** Этот раздел объясняет «Зачем» кода.

Данный Dockerfile формирует надёжный образ, на котором разворачивается приложение на Python с предсказуемой установкой зависимостей и безопасным запуском через стартовый скрипт start.sh.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** CLI Args
- **Format:** Text
- **Schema:**

```typescript
interface InputData {
  baseImage: string; // например: "python:3.10"
  workdir: string; // например: "/app"
  dependenciesList: string; // например: "libraries.list"
  copySource: string; // например: "."
  copyDestination: string; // например: "."
  startupScript: string; // например: "start.sh"
  finalCommand: string[]; // например: ["./start.sh"]
}
```

### 3.2. Outputs (Выходы)

- **Destination:** STDOUT
- **Format:** Text / JSON
- **Success Criteria:** Exit code 0 (успешная сборка образа), образ создан
- **Schema:**

```typescript
interface OutputResult {
  status: "success" | "failure";
  exitCode?: number;
  imageTag?: string;
  message?: string;
  logs?: string[];
}
```

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Выбор базового образа на основе строки baseImage, например python:3.10, что задаёт набор системных инструментов и окружение.
2. Установка рабочей директории в образе через команду, создающую каталог и устанавливающую текущую директорию для следующих шагов.
3. Копирование в образ файла libraries.list из контекста сборки в корневую директорию образа по пути dependenciesList.
4. Выполнение шага установки зависимостей при помощи пакетного менеджера Python, используя файл libraries.list и флаг без кэширования, чтобы итоговый слой был воспроизводимым и меньшим по размеру.
5. Копирование всех пергаментов проекта из контекста в рабочий каталог образа, чтобы финальная сборка содержала весь код и ресурсы.
6. Применение разрешения на исполнение к стартовому скрипту start.sh, чтобы он мог быть запущен как главный процесс контейнера.
7. Установка главного запускающего заклинания в виде команды, которая активирует стартовый скрипт при запуске контейнера.

### 4.2. Declarative Content (Для конфигураций и данных)

- Базовый образ Python: python:3.10
- Рабочая директория: /app
- Файл зависимостей: libraries.list
- Пергаменты проекта: все файлы из контекста копируются в рабочий каталог
- Стартовый скрипт: start.sh
- Разрешение на исполнение стартового скрипта: установленное
- Главная запускающая команда: ./start.sh

## 5. Structural Decomposition (Декомпозиция структуры)

- Этапы инициализации: выбор базового образа и настройка рабочей директории
- Этап загрузки зависимостей: копирование libraries.list и установка зависимостей через pip
- Этап внедрения кода проекта: копирование всех файлов проекта
- Этап подготовки запуска: установка исполняемости для start.sh
- Этап запуска: указание стартовой команды через запуск стартового скрипта

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** Standard CPU
- **Concurrency:** Sync
- **Dependencies:** Docker, Python:3.10, pip

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets в явном виде внутри образа; применяйте переменные окружения или секретные масштабы (secret management) для конфигурации.
- DO NOT выводить несанкционированные данные в консоль в продакшн-режиме.
- DO NOT вставлять синхронные сетевые вызовы в основной поток выполнения сборки образа.
- DO NOT оборачивать конфигурационные файлы (yaml/json) в скрипты для сборки.
- DO NOT изменять версии образов или путей в ходе реконструкции артефакта.

## 7. Verification & Testing (Верификация)

1. Гверкин-Сценарий: Успешная сборка и запуск
   Feature: Dockerfile Build and Run
   Scenario: Successful build and startup
   Given базовый образ python:3.10 доступен
   When выполняется сборка образа по данному Dockerfile
   Then образ успешно создаётся и стартовый скрипт start.sh запускается без ошибок

2. Гверкин-Сценарий: Ошибка при отсутствии dependencies
   Feature: Dockerfile Build Validation
   Scenario: Missing dependencies file
   Given файл libraries.list отсутствует в контексте сборки
   When выполняется сборка образа по данному Dockerfile
   Then сборка завершается с ошибкой о недостающем файле зависимостей
