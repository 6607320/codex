# .pre-commit-config.yaml Specification

## 1. Meta Information

- **Domain:** Infrastructure
- **Complexity:** Low
- **Language:** Bash
- **Frameworks:** pre-commit, black, flake8, docformatter, prettier
- **Context:** Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Легенда: Скрижаль Хранилищ задаёт ритуал предкоммитного очищения и гармонии кода. Этот артефакт описывает набор духов-стражей, которые будут призываться перед каждым коммитом, чтобы автоматически удалять хвосты и лишние пробелы, проверять YAML на синтаксис, держать размер файлов под контролем, приводить стиль к единому канону и выравнивать документацию и форматирование. Вводимые скрижали и призывы к веткам превращают хаос в порядок, а каждый коммит становится чистым и предсказуемым.

Зачем нужен этот файл? Он централизует набор инструментов для проверки и выправления кода и конфигураций до того, как они попадают в общий хранилище, обеспечивая единый стиль, качество и воспроизводимость сборок.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

Source: File
Format: YAML
Schema:
PrecommitInput
repos: RepoInput[]

RepoInput
repo: string
rev?: string
hooks?: HookInput[]

HookInput
id: string
exclude?: string
args?: string[]

### 3.2. Outputs (Выходы)

Destination: STDOUT
Format: Text
Success Criteria: Exit Code 0
Schema:
PrecommitOutput
success: boolean
message?: string
details?: any
exitCode?: number

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

Операция начинается с чтения Скрижали конфигурации, после чего для каждого Кузнечного блока репозитория выполняется следующее: вызываются духи указанной кузницы по заданному аду rev, затем для каждого стража внутри блока активируются его заклинания. Сперва активируются Страж-Чистильщик Хвостов, который удаляет хвостовые пробелы по всем файлам, кроме помеченных исключением. Затем включают Страж Последней Руны, который гарантирует концовку каждого документа одной и единственной пустой строкой. Затем вызывают Страж Гармонии YAML, чтобы убедиться, что YAML-скрипты без синтаксических ошибок. Затем активируют Страж-Привратник, который следит за размером добавляемых файлов и не пропускает большие артефакты выше заданного лимита. Далее следует блок с Кожей Каллиграфа Первого: Каллиграф Блэк, который перекраивает код согласно единому стилю и ограничению длины строк. Затем Инквизитор Флейк8 выполняет проверку соответствия кода установленным правилам и исключениям по файлам. После этого Хранитель Докстрингов форматирует докстринги и комментарии внутри файлов. В завершение призывается Хранитель Эстетики Приттер, который выравнивает стили Markdown и YAML в соответствии с общими правилами. Если все призывы проходят без Хаоса, коммит разрешается (успех), иначе возвращается неуспех и коммит блокируется.

### 4.2. Declarative Content (Для конфигураций и данных)

Репозитории и призывы на службу в текущей Скрижали:

- Репо: https://github.com/pre-commit/pre-commit-hooks
  rev: v6.0.0
  хуки:
  trailing-whitespace
  exclude: \.md$
  end-of-file-fixer
  check-yaml
  check-added-large-files
  maxkb: 800

- Репо: https://github.com/psf/black
  rev: 25.9.0
  хуки:
  black
  args: [--line-length=88]

- Репо: https://github.com/pycqa/flake8
  rev: 7.3.0
  хуки:
  flake8
  args: - --per-file-ignores=Part_4_Engineering/Scroll_26/Quest_1/spell_with_flaw.py:F401

- Репо: https://github.com/PyCQA/docformatter
  rev: v1.7.7
  хуки:
  docformatter
  args: [--in-place, --wrap-summaries=88, --wrap-descriptions=88]

- Репо: https://github.com/pre-commit/mirrors-prettier
  rev: v4.0.0-alpha.8
  хуки:
  prettier

## 5. Structural Decomposition (Декомпозиция структуры)

- Репозитории: список внешних кузниц, каждый с адресом и версией
- Хуки внутри каждого репо: идентификатор и дополнительные параметры
- Параметры исключений: путь или маски исключений для отдельных хуков
- Параметры форматирования и ограничений: длина строк, размеры файлов, оформление документации

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Sync
- Dependencies: внешний доступ к сетям для клонирования репозиториев и загрузки хуков из указанных источников

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap configuration files .yaml .json into scripts
- DO NOT change versions or paths during reconstruction

## 7. Verification & Testing (Верификация)

Геркин сценарии:

Feature: Pre-commit configuration execution
Scenario: Successful commit path
Given рабочее дерево чисто и изменяются только файлы проекта
When выполняется попытка коммита
Then все хуки пройдены и коммит завершается успешно

Scenario: Failure due to trailing whitespace
Given файл содержит хвостовые пробелы в отслеживаемых файлах
When выполняется попытка коммита
Then хук trailing-whitespace не проходит и коммит отклонён с соответствующим сообщением
