# main_linter.yml Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language: Python
- Frameworks: flake8
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Легенда: Этот эктоплант для Кодекса — Ритуал Линтинга. Он призван обеспечить чистоту и последовательность кода в репозитории Codex, автоматически подготавливая окружение Python 3.10, устанавливая инструмент линтинга flake8 и выполняя проверку по всем файлам проекта, с учётом предписанных исключений. В реальном использовании он функционирует как GitHub Action, активирующийся на события push и pull_request в ветке main и возвращающий статус в зависимости от наличия Хаоса в эфире кода.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: JSON
- Schema:
  интерфейс InputData {
  source: 'CLI' | 'STDIN' | 'API';
  payload?: string;
  repoPath?: string;
  excludePaths?: string[];
  configPath?: string;
  }

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: Text
- Success Criteria: Exit code 0
- Schema:
  интерфейс OutputResult {
  success: boolean;
  exitCode: number;
  issues: Array<{
  file: string;
  line?: number;
  code: string;
  message: string;
  }>;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic

Ритуал выполнения разыгрывается в последовательности шагов:

1. Подготовить колдунский круг: обеспечить среду выполнения с Python версии 3.10, используя соответствующий настройный компонент.
2. Принести Инструменты: установить линтер Flake8 через пакетный менеджер Python (pip).
3. Провести обследование Эфира: запустить Flake8 по всем файлам корневого каталога проекта, исключив заданные участки пути Part_4_Engineering/Scroll_26/Quest_1 и Part_4_Engineering/Scroll_26/Quest_2.
4. Собрать Хаос-отчёт: зафиксировать найденные проблемы в структуре, содержащей путь к файлу, номер строки, код проблемы и текстовое сообщение.
5. Завершить Ритуал: если Хаоса нет, завершиться с успешным кодом 0; иначе вернуть неуспех и перечислить найденные ошибки в выводе.

### 4.2. Declarative Content (Для конфигураций и данных)

Эфир данных и конфигураций, нужных для воссоздания 1-в-1:

- Имя Скрижали: Codex Linter
- Версии окружения: Python 3.10
- Инструмент линтинга: flake8
- Команда линтинга: flake8 . с исключениями по путям
- Исключаемые пути: Part_4_Engineering/Scroll_26/Quest_1, Part_4_Engineering/Scroll_26/Quest_2
- Действия на запуск: событие push к ветке main и событие pull_request к ветке main
- Право доступа: contents read

## 5. Structural Decomposition (Декомпозиция структуры)

- name: Codex Linter
- on: блоки триггеров (push к main, pull_request к main)
- permissions: contents: read
- jobs:
  - lint:
    - runs-on: ubuntu-latest
    - steps:
      - Checkout Code: uses: actions/checkout@v4
      - Set up Python: uses: actions/setup-python@v5, with: python-version: "3.10"
      - Install Dependencies: run: pip install flake8
      - Lint with flake8: run: flake8 . --exclude=Part_4_Engineering/Scroll_26/Quest_1,Part_4_Engineering/Scroll_26/Quest_2

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Sync
- Dependencies: Python 3.10, flake8

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде в артефакте или в логах.
- DO NOT выводить сырые данные в консоль в продакшн-режиме.
- DO NOT выполнять сетевые вызовы синхронно внутри длинных циклов линтинга.
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты (как часть кода).
- DO NOT менять версии или пути во время реконструкции артефакта.

## 7. Verification & Testing (Верификация)

Геркин сценарии

```gherkin
Feature: Codex Linter main_linter.yml Verification
  Scenario: Successful execution
    Given репозиторий без ошибок линтинга в ветке main
    When GitHub Action запускается на событие push или pull_request к main
    Then процесс завершается с кодом 0 и в логе не упоминается Хаос

  Scenario: Lint errors detected
    Given репозиторий содержит хотя бы одну линтин-ошибку
    When GitHub Action запускается на событие push или pull_request к main
    Then процесс завершается с ненулевым кодом и выводит список ошибок линтинга
```
