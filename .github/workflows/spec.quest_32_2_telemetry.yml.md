# quest_32_2_telemetry.yml Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language: Bash
- Frameworks: GitHub Actions, Docker, pytest, Google Cloud Workload Identity, docker/login-action, docker/build-push-action, actions/checkout, actions/upload-artifact
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Легенда: Этот артефакт описывает единый конвейер непрерывной сборки и телеметрии для Capstone Quest 32.2. Он управляет процессом извлечения кода, очистки окружения, установки зависимостей, тестирования и генерации покрытия, публикации артефактов, а затем сборкой и публикацией образа Docker в реестр, усиливая видимость качества и воспроизводимости сборок.

Instruction for AI: Этот файл служит центральным регистром жизненного цикла проекта Quest 32.2, обеспечивая повторяемый, надёжный и воспроизводимый процесс CI/CD с телеметрией тестирования.

Описание на русском языке: Конвейер выполняет Checkout репозитория, освобождает место, устанавливает зависимости, запускает тесты и формирует отчет по покрытию, выгружает артефакт покрытия, а затем аутентифицируется в облаке, входит в реестр артефактов, строит Docker-образ и публикует его. Весь процесс синхронен и управляется шагами в рамках единого задания build-test-deploy.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: STDIN
- Format: JSON
- Schema:
  interface InputData {
  source: 'STDIN';
  format: 'JSON';
  payload?: unknown;
  }

### 3.2. Outputs (Выходы)

- Destination: File
- Format: JSON
- Success Criteria: Exit Code 0
- Schema:
  interface OutputResult {
  success: boolean;
  exitCode?: number;
  coveragePath?: string;
  artifactUrls?: string[];
  logPath?: string;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Пролог ritual: инициализация окружения и загрузка контекста проекта Capstone Quest 32.2 через Checkout репозитория.
2. Ритуал очищения: освободить место на диске, удалив устаревшие инструментальные комплекты, чтобы обеспечить свободу пространства для сборки и артефактирования.
3. Проклятие зависимостей: установить необходимые зависимости, считанные из файлов requirements.txt и requirements_dev.txt, чтобы обеспечить полноту тестирования и разработки.
4. Испытание и нагромождение: запустить pytest с измерением покрытия, собрать xml-отчет и проверить прохождение тестов внутри области квеста.
5. Признак доказательств: загрузить артефакт покрытия (coverage.xml) в артефакт-репозиторий.
6. Воссоздание положения облачного духа: аутентифицироваться в Google Cloud через Workload Identity Provider с использованием секретов проекта.
7. Доступ к мастерской артефактов: войти в Artifact Registry с помощью OAuth2 токена.
8. Созидание образа: собрать Docker-образ из директории квеста и отправить его в реестр с тегом, основанным на идентификаторе проекта и sha коммита.
9. Эпилог: логирование итогов и завершение конвейера.

### 4.2. Declarative Content (Для конфигураций и данных)

Указ Ткачу и точные данные для воссоздания 1-в-1:

- Название конвейера: Capstone Pipeline with Telemetry (Quest 32.2)
- Триггер: событие push
- Ветка: main
- Путь триггера: Part_4_Engineering/Scroll_32/Quest_2/\*\*
- Разрешения: contents: read, id-token: write
- Задание: build-test-deploy
- runs-on: ubuntu-latest
- Шаги:
  - Checkout Repository: uses: actions/checkout@v4
  - Free Up Disk Space: очистка дискового пространства, удаление dotnet, ghc, boost, AGENT_TOOLSDIRECTORY
  - Install Dependencies: pip install -r ./Part_4_Engineering/Scroll_32/Quest_2/requirements.txt; pip install -r ./Part_4_Engineering/Scroll_32/Quest_2/requirements_dev.txt
  - Run Tests and Generate Coverage Report: pytest с покрытием по ./Part_4_Engineering/Scroll_32/Quest_2; вывод coverage.xml
  - Upload Coverage Report Artifact: artifact coverage.xml с именем coverage-report-32-2
  - Authenticate to Google Cloud: google-github-actions/auth@v2 с workload_identity_provider: GCP_WORKLOAD_IDENTITY_PROVIDER и service_account: GCP_SERVICE_ACCOUNT
  - Login to Artifact Registry: docker/login-action@v3 с registry europe-west3-docker.pkg.dev и OAuth2 token
  - Build and Push Docker Image: docker/build-push-action@v5 с контекстом ./Part_4_Engineering/Scroll_32/Quest_2, push: true, тегами europe-west3-docker.pkg.dev/${steps.auth.outputs.project_id}/codex-golems/amulet-telemetry:${github.sha}

Примечание: технические названия и пути сохранены без изменений для точной воспроизводимости.

## 5. Structural Decomposition (Декомпозиция структуры)

- Главные узлы:
  - name: Capstone Pipeline with Telemetry (Quest 32.2)
  - on: push (branches: main, paths: "Part_4_Engineering/Scroll_32/Quest_2/\*\*")
  - permissions: contents: read, id-token: write
  - jobs: build-test-deploy
- Подузлы внутри jobs:
  - name: Build, Test, Deploy, and Collect Artifacts
  - runs-on: ubuntu-latest
  - steps:
    - Checkout Repository
    - Free Up Disk Space
    - Install Dependencies
    - Run Tests and Generate Coverage Report
    - Upload Coverage Report Artifact
    - Authenticate to Google Cloud
    - Login to Artifact Registry
    - Build and Push Docker Image

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU, обычный лимит памяти CI-оникса
- Concurrency: Синхронный последовательный конвейер в рамках одного задания
- Dependencies: actions/checkout@v4, google-github-actions/auth@v2, docker/login-action@v3, docker/build-push-action@v5, pytest, pip, AWS/Azure/GCP SDK по мере необходимости

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap конфигурационные файлы (.yaml, .json) в скрипты (как Python/Bash)
- DO NOT change versions or paths during reconstruction

## 7. Verification & Testing (Верификация)

1. Герхин-сценарий: Успешное выполнение
   Функционал: конвейер успешно проходит шаги: Checkout, очистка диска, установка зависимостей, тесты, генерация покрытия, загрузка артефакта, аутентификация в облаке, вход в реестр, сборка и публикация Docker-образа.
   Дракон: все шаги завершаются успешно и артефакт покрытия доступен.

2. Герхин-сценарий: Ошибка в тестах
   Функционал: тесты не проходят на шаге Run Tests and Generate Coverage Report; конвейер завершается с ошибкой на этом шаге, соответствующий статус помечен как неуспешный, и последующие шаги не выполняются.

Герхин:
Feature: [Script Functionality]
Scenario: Successful execution
Given Preconditions
When Action is taken
Then Expected result

Scenario: Tests fail
Given Preconditions
When Action is taken
Then Expected result

ИССЛЕДУЕМЫЙ АРТЕФАКТ: quest_32_2_telemetry.yml

ИСТОЧНЫЙ КОД: Capstone Pipeline with Telemetry (Quest 32.2) — YAML-конфигурация GitHub Actions, включающая триггеры, разрешения, блок jobs и восемь шагов, включая очистку окружения, установку зависимостей, тесты с покрытием, аутентификацию в Google Cloud, вход в реестр и публикацию Docker-образа.

ИНФРАСТРУКТУРА АРТЕФАКТА (Inventory)

- 🏰 quest_32_2_telemetry.yml — Рабочий фолиант конвейера
- 🛡️ Триггерные руны — on: push; ветка main; путь к квесту
- 🏰 Скрижаль разрешений — contents: read; id-token: write
- 🛡️ Глава задач — Build, Test, Deploy, and Collect Artifacts
- 🏰 Каталоги и артефакты — Part_4_Engineering/Scroll_32/Quest_2, coverage.xml
- 🛡️ Реестр и доступ — europe-west3-docker.pkg.dev, OAuth2 токен
- 🏰 Образ — codex-golems/amulet-telemetry:${github.sha} в реестре
- 🛡️ Данные телеметрии — coverage.xml и лог-файлы конвейера

Готово.
