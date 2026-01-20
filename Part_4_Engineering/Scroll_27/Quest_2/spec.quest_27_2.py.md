# quest_27_2.py Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Low
- Language: Python
- Frameworks: os
- Context: https://docs.python.org/3/library/os.html

## 2. Goal & Purpose (Легенда)

Этот артефакт — Ритуал Фантома: он сотворяет иллюзорную модель размером 10 МБ, названную golem_mind.pth, в святилище models, чтобы продемонстрировать версионирование через ДВК. В ходе канона руна os создаёт директорию, чертит файл нужного размера байтами и провозглашает о завершении Чертог Сотворения. Легенда говорит, что такой фантом нужен для наглядного показа механики сохранения версий без реальных весомых данных.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: Text
- Schema:

```typescript
interface InputData {
  modelDir: string;
  modelName: string;
  sizeBytes: number;
}
```

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: Text
- Success Criteria: Exit 0
- Schema:

```typescript
interface OutputResult {
  modelPath: string;
  created: boolean;
  sizeBytes: number;
  message: string;
}
```

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Определяются константы артефакта: директория святилища для фантома, имя фантома и путь к файлу, а также запрошенный размер в мегабайтах и эквивалент в байтах.
- Совершается ритуал создания папки: вызывается функция makedirs для MODEL_DIR с флагом exist_ok=True, что означает руна мудрости — не выдаёт ошибки, если папка уже существует.
- В кристалле консоли объявляется начало ритуала: печатается сообщение о создании фантома размером SIZE_MB МБ по пути MODEL_PATH.
- Открывается портал в файл по MODEL_PATH в режимах записи байтов (wb). В контексте портала выполняется запись байтов нулей, равной SIZE_BYTES, чтобы собрать эфир нужного размера.
- В конце ритуала выводится сообщение об успешном завершении сотворения фантома.

### 4.2. Declarative Content (Для конфигураций и данных)

Данные конфигураций и параметры заклинания — это набор констант и манифестов:

- MODEL_DIR: имя папки-святилища
- MODEL_NAME: имя фантома модели
- MODEL_PATH: путь к фантомному файлу, соединение пути и имени
- SIZE_MB: размер фантома в мегабайтах
- SIZE*BYTES: размер фантома в байтах (SIZE_MB * 1024 \_ 1024)
- SIZE_BYTES управляет количеством заполняемых нулевых байтов в файле
- os: инструмент управления файловой системой, который приводит в движение создание папки и файлового портала

## 5. Structural Decomposition (Декомпозиция структуры)

- Константы артефакта: MODEL_DIR, MODEL_NAME, MODEL_PATH, SIZE_MB, SIZE_BYTES
- Инструменты: os
- Этапы ритуала: makedirs (с existential-магией), вывод стартового сообщения, открытие файла в режиме wb, запись SIZE_BYTES нулевых байтов, вывод успешного завершения

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: стандартный CPU, пара пустых гигабайтов свободной памяти
- Concurrency: синхронный запуск, без параллелизма
- Dependencies: только стандартная библиотека Python (os)
- Взаимодействие: работа с файловой системой через os

### 6.2. Prohibited Actions (Negative Constraints)

- НЕ хранить секреты в открытом виде; избегать хранилища секретов прямо в файле
- НЕ выводить большие объемы сырых данных в консоль в продакшн-режиме
- НЕ использовать синхронные сетевые вызовы в главном цикле (для этого артефакта не требуется)
- НЕ оборачивать конфигурационные файлы (.yaml, .json) в скрипты
- НЕ менять версии или пути во время реконструкции артефакта

## 7. Verification & Testing (Верификация)

```gherkin
Feature: Phantom Mind Creator
  Scenario: Successful execution
    Given the environment allows creating the models directory and writing files
    When the ritual is invoked
    Then the file models/golem_mind.pth exists
    And its size is 10485760 bytes
    And the console prints start and success messages

```

```gherkin
Feature: Phantom Mind Creator
  Scenario: Failure due to insufficient write permissions
    Given the destination directory is not writable
    When the ritual is invoked
    Then an error occurs (permission denied) and no file is created
    And the process reports an error message
```
