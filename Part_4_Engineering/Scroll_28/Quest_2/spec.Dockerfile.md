# Dockerfile Specification

## 1. Meta Information

- **Domain:** Infrastructure
- **Complexity:** Medium
- **Language:** Bash
- **Frameworks:** Docker, Pip, Python 3.10
- **Context:** Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

**Context for Creator:** Этот ритуал упаковывает повторяемую среду Python внутри контейнера. Он загружает зависимости, призывает модель через скрипт загрузки, сохраняет основной код и запускает Главный Заклинатель Жизни через стартовый скрипт. Всё это обеспечивает надёжный, предсказуемый запуск в облачном окружении и простоту переноса между средами.

**Instruction for AI:** Это описывает общую идею и мотивацию кода: собрать корректную контейнерную среду, гарантировать установку зависимостей, подготовку модели и запуск приложения через отдельный стартовый сценарий, чтобы воспроизвести поведение в любом окружении (например в Cloud Run).

Описание на русском языке: Этот артефакт описывает путь к созданию полностью подготовленного Docker-образа на базе Python 3.10, где сначала копируются списки библиотек, затем они устанавливаются, затем выполняется загрузка модели, и далее копируется основной код и стартовый сценарий, который запускается как команда по умолчанию. Такой подход обеспечивает изоляцию, воспроизводимость и гибкость в развёртывании.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** CLI Args
- **Format:** Text
- **Schema:** InputData

* InputData: данных по факту в Dockerfile нет явных входов для рантайма пакета. Этот контракт определяет формальную возможность ввода, но в рамках артефакта входы не используются во время выполнения образа.

### 3.2. Outputs (Выходы)

- **Destination:** STDOUT
- **Format:** Text
- **Success Criteria:** Exit Code 0
- **Schema:** OutputResult

* OutputResult: status (string), exitCode (number), log (string, опционально)

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Начинается ритуал с выбора топлива для призыва: базовый образ Python 3.10 ставится как FROM python:3.10, создавая прочную оболочку среду.

2. Внесение рабочих умов в мастерскую: устанавливается рабочая директория /app через функциональное письмо WORKDIR.

3. Приводим в свиток зависимостей: копируется файл libraries.list в рабочую среду как источник письменных чар.

4. Призываем зависимые менторы: выполняется установка зависимостей с помощью команды pip install --no-cache-dir -r libraries.list, наполняя эфир необходимыми пакетами.

5. Призыв к источнику знания: копируется download_model.py и запускается python download_model.py, чтобы эфир модели был загружен и готов к призыву.

6. Призываем основное знание: копируются main.py и стартовый свиток start.sh для последующего начала заклинания.

7. Подготовка к живому пробуждению: копируется start.sh, ему даётся право на жизнь (chmod +x).

8. Само заклинание жизни: CMD запускает ./start.sh, инициируя жилой цикл приложения в чистой среде и под окружение облачного окружения (Cloud Run).

### 4.2. Declarative Content (Для конфигураций и данных)

- Базовый образ: python:3.10
- Рабочая директория: /app
- Файлы конфигурации: libraries.list
- Скрипты: download_model.py, main.py, start.sh
- Команды установки: pip install -r libraries.list
- Призыв к загрузке модели: python download_model.py
- Право на исполнение стартового скрипта: chmod +x start.sh
- Точка входа: CMD ./start.sh

## 5. Structural Decomposition (Декомпозиция структуры)

- FROM python:3.10 — Начало ритуала.
- WORKDIR /app — Установка мастерской пространства.
- COPY libraries.list . — Перемещаем свиток зависимостей.
- RUN pip install --no-cache-dir -r libraries.list — Вызов алхимии установки зависимостей.
- COPY download_model.py . — Призыв к источнику знания.
- RUN python download_model.py — Исполнение заклинания загрузки модели.
- COPY main.py . — Перенос основного кода.
- COPY start.sh . — Перенос стартового ритуала.
- RUN chmod +x ./start.sh — Дарование права на жизнь стартовому скрипту.
- CMD ["./start.sh"] — Финальный призыв к пробуждению.

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** Standard CPU
- **Concurrency:** Single process at startup (нет параллельного запуска внутри контейнера на этапе инициализации)
- **Dependencies:** Docker, Python 3.10, доступ к интернету для загрузки библиотеки и модели

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use env vars или секреты окружения).
- DO NOT print raw data to console in production mode.
- DO NOT use synchronous network calls in the main loop (контекст выполнения должен оставаться предсказуемым).
- DO NOT wrap конфигурационные файлы (.yaml, .json) в скрипты (лучше держать их отдельно).
- DO NOT change версионирование или пути во время реконструкции образа.

## 7. Verification & Testing (Верификация)

1. Gherkin сценарий: Успешное выполнение
   Feature: Dockerfile Artifacts
   Scenario: Successful image build and startup
   Given существующие файлы libraries.list, download_model.py, main.py и start.sh
   When выполняется сборка Docker образа и запуск контейнера
   Then контейнер запускается успешно и основной сервис входит в рабочий режим, отдаются корректные логи и код выхода 0

2. Gherkin сценарий: Ошибка отсутствия зависимостей
   Feature: Dockerfile Artifacts
   Scenario: Missing libraries.list leads to build failure
   Given libraries.list отсутствует в контексте сборки
   When выполняется сборка Docker образа
   Then сборка завершается с ненулевым кодом выхода и появляется сообщение об отсутствии библиотеки/файла libraries.list

ИССЛЕДУЕМЫЙ АРТЕФАКТ: Dockerfile
