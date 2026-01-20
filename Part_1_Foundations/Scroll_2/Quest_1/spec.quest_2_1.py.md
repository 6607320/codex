# quest_2_1.py Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language: Python
- Frameworks: transformers, numpy
- Context: ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

Легенда: Этот Ритуал-скрипт проводит призыв духа Whisper-tiny через скрипт-подобный артефакт, чтобы проверить отклик на искусственную магическую тишину и продемонстрировать поведение модели на бессмысленных входных данных — первый урок в потенциальных галлюцинациях модели на абсурдном Эфире.

Instruction for AI: Этот раздел описывает WHY: зачем нужен этот файл и какую бизнес-задачу он иллюстрирует.

Описание на русском языке: Этот модуль служит демонстрационной сценой для тестирования отклика ASR-модели Whisper-tiny, используя искусственно созданный аудиоскоп — тишину — чтобы убедиться, что модель активна и возвращает результат в формате, пригодном для анализа. Он также демонстрирует базовую обработку результатов и логику вывода, полезную для дальнейших экспериментов с галлюцинациями и устойчивостью к негативному вводу.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: STDIN
- Format: Binary
- Schema: InputData
  - audioData: number[] — массив амплитуд аудиосигнала
  - sampleRate: number — частота дискретизации
  - requestId?: string — необязательный идентификатор запроса

Note: Это описание интерфейса представленo в духе TypeScript-интерфейсов, но сформулировано без копирования конкретного кода. Поля заданы так, чтобы воспроизводимо описать структуру Эфира входа для ритуала.

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: JSON
- Success Criteria: Exit Code 0
- Schema: OutputResult
  - transcript: string — текстовая расшифровка
  - confidence?: number — необязательная уверенность
  - tokens?: string[] — необязательные токены/слова
  - length?: number — длительность вывода (опционально)

Note: как в 3.1, вывод описывается в форме TypeScript-подобной структуры, но без вставки кода.

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Подготовка Ритуала: загрузка необходимых инструментов — империя библиотек и артефакт-амулета.
- Призыв духа Whisper-tiny: создание транслятора речи через аркан pipeline с задачей “automatic-speech-recognition” и моделью "openai/whisper-tiny", с умной руной device_map="auto".
- Создание Магической Тишины: генерация Эфира тишины — массив нулей длиной 16000, эквивалент 1 секунде аудио при частоте 16000 Гц.
- Ритуал Распознавания: подача магического тишинного Эфира транслятору и получение расшифровки.
- Анализ Результата: печать заголовка расшифровки и самого результата; вывод позволяет увидеть, как модель реагирует на “пустой” вход и какие артефакты могут появляться.
- Примечание: ритуал начинается с объявления призыва духа, затем выполняется собственно распознавание и анализ, после чего артефакт выводит результаты в консоль.

### 4.2. Declarative Content (Для конфигураций и данных)

- Модель: openai/whisper-tiny
- Задача: automatic-speech-recognition
- Окружение устройства: device_map="auto" — умная руна для распределения нагрузки между CPU/GPU
- Эфир входа: fake_speech — массив нулей длиной 16000
- Визуализация результата: печать заголовка и самого результата (объект/слово)

## 5. Structural Decomposition (Декомпозиция структуры)

- Акт 1: Подготовка Гримуаров
  - Импорт библиотек
  - Подготовка артефактов pipeline
- Акт 2: Призыв Whisper-tiny
  - Создание транслятора через pipeline
  - Установка параметров модели и распределения
- Акт 3: Создание Магической Тишины
  - Генерация нулевого аудиосигнала на 1 секунду
- Акт 4: Ритуал Распознавания
  - Передача тишины транслятору и получение расшифровки
- Акт 5: Анализ Результата
  - Печать вывода и его структуры
- Утилиты/Компоненты
  - Транслятор (pipeline)
  - Эфирные данные (fake_speech)
  - Логгер/Печать

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: стандартный CPU; GPU по возможности за счет device_map="auto"
- Concurrency: синхронная работа (Sync)
- Dependencies: transformers, numpy; модель: openai/whisper-tiny
- Контекст выполнения: работа через консоль/скрипт, вывод через print

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap конфигурационные файлы (.yaml, .json) в скрипты
- DO NOT change версии или пути во время реконструкции

## 7. Verification & Testing (Верификация)

### 7.2 Gherkin: happy path и один кейс ошибки

```gherkin
Feature: Script Functionality
  Scenario: Successful execution
    Given the Python environment has Python, transformers, numpy installed and quest_2_1.py is present
    When the script quest_2_1.py is executed
    Then the console prints "Результат расшифровки тишины:" followed by a result object (transcript, optional length/confidence)

  Scenario: Model loading failure
    Given the environment cannot load the model (no internet or missing weights)
    When attempting to initialize Whisper-tiny via the ritual
    Then the script reports an initialization error and exits with a non-zero code
```

Итоговые артифакты для Великих Хроник:

- Название артефакта: quest_2_1.py
- Ритуал подтверждает, что дух Whisper-tiny активен и корректно обрабатывает искусственную тишину, демонстрируя базовую устойчивость к бессмысленным входным данным и наблюдение за возможными галлюцинациями модели.
