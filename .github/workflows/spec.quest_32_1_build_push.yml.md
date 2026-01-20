# quest_32_1_build_push.yml Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language: YAML
- Frameworks: actions/checkout@v4, google-github-actions/auth@v2, docker/login-action@v3, docker/build-push-action@v5
- Context: ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

Context for Creator: Этот Ритуал призывает собрать голема-образ и отпустить его в Хранилище Артефактов (GAR) при каждом пуше в главную ветку, ограниченного путём волшебного Свода Порогов по пути Part_4_Engineering/Scroll_32/Quest_1. Он обеспечивает аутентификацию через Workload Identity Federation, авторизацию к GAR и автоматическую публикацию образа в регистр. Это ускоряет поставку и обеспечивает строгий контроль версий за счёт привязки образа к sha-рипе кода.

Instruction for AI: Этот секционный свиток задаёт общий замысел и WHY: зачем нужен файл — автоматизация сборки и публикации Docker-образа в GAR при соответствующем триггере, с безопасной аутентификацией и отслеживаемостью версии.

Описание на русском языке: Этот файло-рим прокладывает шаги для автоматического извлечения кода, аутентификации в облаке, входа в GAR, сборки образа из контекста Quest_1 и отправки образа в артефакт-хранилище с тегом, равным текущему sha коммита. Весь процесс защищён и повторяем в каждой итерации выпуска.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

Source: GitHub Actions Workflow (Ритуал)  
Format: YAML (Скрижаль управляет данными через Эфир окружения и секреты)  
Schema:

- InputData
  - eventName?: string
  - repository?: string
  - ref?: string
  - sha?: string
  - secrets?: Record<string, string>

Примечание: в этой Скрижали входы прямо не задаются явным образом; контекст действий доступен через окружение GitHub Actions и секретаы.

### 3.2. Outputs (Выходы)

Destination: GAR (Артефактное Хранилище) и логи GH Actions  
Format: JSON (метаданные) / Text (логирование)  
Success Criteria: Exit 0 / 200 OK (успешное создание и пуш образа)  
Schema:

- OutputResult
  - registryUrl?: string
  - imageTag?: string
  - success?: boolean
  - message?: string

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

Шаг первый: Ритуал просыпается при событии push в ветке main и при наличии изменений в Part_4_Engineering/Scroll_32/Quest_1.  
Шаг второй: Дух-исполнитель обретает ход через Checkout Repository — копирует текущее хранилище на алтарь.  
Шаг третий: Дева-телепатия Google Cloud активируется через Authenticate to Google Cloud с параметрами token_format: access_token и указанием workload_identity_provider и service_account, взятыми из секрета GCP_WORKLOAD_IDENTITY_PROVIDER и GCP_SERVICE_ACCOUNT.  
Шаг четвертый: Включается Прозрение (DEBUG - Check Access Token) — выводится длина полученного токена и проверяется, что токен действительно существует.  
Шаг пятый: Врата GAR открываются через Login to Google Artifact Registry (docker/login-action) с регистрием europe-west3-docker.pkg.dev и подлинными данными из access_token.  
Шаг шестой: Великий кузнец образов (Build and push Docker image) — с помощью docker/build-push-action создаётся Docker-образ из контекста ./Part_4_Engineering/Scroll_32/Quest_1, и образ отправляется в регистр GAR с тегом, составленным из идентификатора проекта и sha текущего коммита.  
Шаг седьмой: Мастер-текст логов подтверждает успешный запуск и прокатывает результаты обратно в Эфир и GAR.

### 4.2. Declarative Content (Для конфигураций и данных)

Скрижаль описывает следующие элементы артефактов и данных:

- Контекст сборки: Part_4_Engineering/Scroll_32/Quest_1 — мастерская, где лежат Dockerfile и материалы
- Целевой регистр: europe-west3-docker.pkg.dev
- Условия доступа: GCP_WORKLOAD_IDENTITY_PROVIDER и GCP_SERVICE_ACCOUNT из секретов
- Метка артефакта: тег образа включает sha текущего коммита
- Инструменты: actions/checkout, google-github-actions/auth, docker/login-action, docker/build-push-action

## 5. Structural Decomposition (Декомпозиция структуры)

- Trigger section: on push ветка main с путём в Quest_1
- Permissions: доступ на чтение к репозиторию и создание токена
- Jobs: build-and-push
  - Step: Checkout Repository
  - Step: Authenticate to Google Cloud
  - Step: DEBUG - Check Access Token
  - Step: Login to Google Artifact Registry
  - Step: Build and push Docker image

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU (GitHub Runner — ubuntu-latest)
- Concurrency: Синхронный, последовательный вызов шагов
- Dependencies:
  - actions/checkout@v4
  - google-github-actions/auth@v2
  - docker/login-action@v3
  - docker/build-push-action@v5
  - Secrets: GCP_WORKLOAD_IDENTITY_PROVIDER, GCP_SERVICE_ACCOUNT

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT stores secrets in plain text (использовать секреты)
- DO NOT print raw data to console в продакшн-режиме
- DO NOT использовать синхронные сетевые вызовы в главной петле ритуала
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты
- DO NOT менять версии или пути во время реконструкции

## 7. Verification & Testing (Верификация)

Геркин-сценарии для счастья пути и одного сбоя

Feature: Quest 32-1 Build Push Verification
Scenario: Successful execution
Given пуш-событие в ветке main с изменениями в Part_4_Engineering/Scroll_32/Quest_1
When все шаги ритуала выполняются успешно
Then Docker-образ строится и отправляется в europe-west3-docker.pkg.dev/codex-golems/amulet с тегом, равным sha коммита
Scenario: Failure due to missing credentials
Given те же условия, но креды GCP недоступны
When ритуал запускается
Then процесс завершается с ошибкой на шаге аутентификации или публикации образа

ИСТОЧНИК АРТЕФАКТА: quest_32_1_build_push.yml

ИНДИФИКАТОР АРТЕФАКТА: Independent Artifact

Легенда завершена — древний свиток зафиксирован и готов к чтению Великим Хранителям Кодекса.
