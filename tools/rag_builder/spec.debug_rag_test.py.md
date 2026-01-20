# debug_rag_test.py Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language: Python
- Frameworks: Standard Library (os)
- Context: ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

Легенда: В мире кодовых существ Ритуал Диагностики для Rag Builder — это хранитель зрения сквозь мрак путей. Этот Ритуал читает Эфир проекта в режиме только чтения, не изменяя боевой manifest.json и не создавая PDF в корне, и выносит ясный вердикт: причина слепоты кроется либо в путях, либо в правилах фильтрации. Зачем нужен этот файл: дать структурированное описание задачи и поведения Ритуала, чтобы любой Архимаг мог воспроизвести диагностику без риска повредить окружение.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args | STDIN | API Request | Kafka Topic | Smart Contract Call
- Format: JSON | Text | Binary | Stream
- Schema:
  interface InputData {
  directory: string;
  extensions: string[];
  filenames: string[];
  excludeDirs: string[];
  }

### 3.2. Outputs (Выходы)

- Destination: STDOUT | File | Database | API Response | Event Log
- Format: JSON | CSV | Text
- Success Criteria: Exit Code 0 | 200 OK | File Created
- Schema:
  interface OutputResult {
  totalFound: number;
  foundFiles: string[];
  pyFound: boolean;
  mdFound: boolean;
  messages: string[];
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Установить безопасную основу для Ритуала: задать корневой каталог PROJECT_DIR как текущую папку, собрать перечень директорий для Игнорирования EXCLUDE_DIRS, указать набор целевых расширений TARGET_EXTENSIONS и имена файлов TARGET_FILES, а также указать безопасную область вывода SAFE_OUTPUT_DIR и безопасное имя манифеста SAFE_MANIFEST.
2. Начать сканирование Эфира проекта через ритуал scan_project_files, передав PROJECT_DIR, TARGET_EXTENSIONS, TARGET_FILES и EXCLUDE_DIRS.
3. Вывести на Стражу начала Диагностики заголовок и текущее положение в Эфире (CWD и PROJECT_DIR).
4. Полученный список файлов отсортировать и зафиксировать в результации.
5. Определить статусы целевых файлов: проверить наличие quest_2_1.py и manifest.md в списке найденных файлов.
6. Показать сводку целевых файлов: виден ли каждый из них или НИЧЕГО не виден (Скрипт слеп).
7. Если ни один целевой файл не найден, вывести вывод о слепоте и возможной причине: путь не в том месте, или манифест был создан ранее в другой директории.
8. Если найден только один из файлов, вывести альтернативный вывод, указывая на возможную проблему в фильтрах или расширениях.
9. Если найдены оба файла, зафиксируй успешное окончание диагностики.
10. При запуске напрямую вызвать главный ритуал run_test.

### 4.2. Declarative Content (Для конфигураций и данных)

Скрижаль конфигурации и данные, что управляют Ритуалом, воплощены в следующих элементах:

- PROJECT_DIR — корневая папка для сканирования, установлен как текущая папка .
- EXCLUDE_DIRS — набор директорий, полностью исключаемых из обхода: .git, .dvc, **pycache**, .pytest_cache, build, dist, notebooklm_sources, debug_output.
- TARGET_EXTENSIONS — список расширений, которые учитываются: .py, .md, .txt, .json, .yml, .yaml, .spec, .sh, .mono, .html, .css, .js.
- TARGET_FILES — конкретные имена файлов без расширений: Dockerfile, .gitignore, .dvcignore, .flake8, LICENSE, NOTICE.
- SAFE_OUTPUT_DIR — имя безопасной папки вывода debug_output.
- SAFE_MANIFEST — имя безопасного файла манифеста debug_manifest.json.
- scan_project_files — ритуал сканирования, который принимает directory, extensions, filenames, exclude_dirs и возвращает отсортированный список найденных путей.
- run_test — главный ритуал запуска диагностики, который создаёт директорию вывода, вызывает сканирование и формирует итоговую карту найденного.
- Основной поток исполнения (точка входа) — главный охранный тестовый ритуал, который активируется при прямом запуске файла.

## 5. Structural Decomposition (Декомпозиция структуры)

- Функции
  - scan_project_files(directory, extensions, filenames, exclude_dirs) — сканирует директорию, рекурсивно обходя подпапки (за исключением EXCLUDE_DIRS); возвращает упорядоченный список путей к файлам, соответствующим расширениям или именам файлов.
  - run_test() — создаёт безопасную папку вывода, печатает статус диагностики, вызывает scan_project_files и выводит резюме и выводы по целевым файлам.
  - Точка входа if **name** == "**main**": вызывает run_test()

- Ключевые константы и переменные (как в исходном коде)
  - PROJECT_DIR
  - EXCLUDE_DIRS
  - TARGET_EXTENSIONS
  - TARGET_FILES
  - SAFE_OUTPUT_DIR
  - SAFE_MANIFEST

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Sync
- Dependencies: Только стандартная библиотека Python (модуль os)

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде; пользуйтесь .env там, где требуется секретная конфигурация.
- DO NOT печатать сырые данные в консоль в продуктивной эксплуатации.
- DO NOT использовать синхронные сетевые вызовы в основном цикле.
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты.
- DO NOT менять версии или пути во время реконструкции артефакта.

## 7. Verification & Testing (Верификация)

Геркин-сценарии
Feature: [Script Functionality]
Scenario: Successful execution
Given the diagnostic ritual has access to a project tree containing quest_2_1.py and manifest.md
When the ritual is executed
Then the tool reports total found files and confirms that both целевые файлы видимы

Scenario: Missing целевые файлы
Given the project tree lacks quest_2_1.py and manifest.md
When the ritual is executed
Then the tool reports blindness (не видит целевые файлы) and explains possible cause

ИССЛЕДУЕМЫЙ АРТЕФАКТ: debug_rag_test.py
