# validate_model.py Specification

## 1. Meta Information

- Domain: ML/NLP
- Complexity: Medium
- Language: Python
- Frameworks: transformers, Click, torch, torch.profiler, validation_data
- Context: ../AGENTS.md; validation_data.VALIDATION_SET; transformers.pipeline; torch.profiler

## 2. Goal & Purpose (Легенда)

Этот файл реализует мощный CLI-ритуал для продвинутой валидации модели: загружает указанную модель из Hugging Face, профилирует процесс валидации с помощью torch.profiler, подсчитывает точность на заданном наборе примеров и выводит подробные отчеты и топ-15 CPU-профилей. Цель — перейти от догадок о узких местах к точной количественной картины производительности и корректности модели.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: Text
- Schema:
  interface InputData {
  modelName: string;
  }

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: Text
- Success Criteria: Exit Code 0
- Schema:
  interface OutputResult {
  accuracy: number;
  correctPredictions: number;
  totalSamples: number;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Пользователь вызывает команду CLI и передает параметр --model-name (с дефолтом на distilbert-base-uncased-finetuned-sst-2-english). Мастер-ритуал сообщает, что призывается дух модели с указанным именем.
2. Вступительный ритуал пытается создать pipeline для задачи sentiment-analysis, используя переданное имя модели. Если призыв удачен — продолжаем; если нет — печатаем ошибку и прерываем ритуал.
3. Подсчитываем общее число испытаний как длину VALIDATION_SET. Объявляем начало испытаний.
4. Вступаем в контекст Хронометра Истинного Времени, настраивая активити профайлера CPU и CUDA, запись форм, стек вызовов и именование профайлера.
5. Внутри профайлера запускаем ритуал испытаний: последовательно для каждого элемента VALIDATION_SET извлекаем текст и истинную метку, запрашиваем предсказание у sentiment_analyzer(text), сравниваем предсказанную метку с истинной и при совпадении увеличиваем счетчик верных предсказаний.
6. По завершении испытаний вычисляем точность как процент верных предсказаний от общего числа примеров.
7. Выводим итоговый отчет: общий статус, точность, количество верных ответов и общее число примеров. Затем выводим таблицу топ-15 CPU-профилей из Хронометра и завершаем ритуал.
8. При запуске через точку входа **name** == "**main**" запускаем главный ритуал как CLI-команду.

### 4.2. Declarative Content (Для конфигураций и данных)

Inventory описывает реальные данные и артефакты, которые задействованы в этом ритуале (не копируем код).

- VALIDATION_SET: набор примеров на входе для валидации, где каждый элемент содержит текст примера и истинную метку (например, текст и соответствующая положительная/отрицательная метка). Источник: модуль validation_data.
- model-name: строка, переданная через CLI (--model-name) или значение по умолчанию; используется для загрузки модели через huggingface pipeline.
- sentiment-analysis pipeline: готовый инструмент анализа сентимента, созданный через transformers.pipeline с использованием указанной модели.
- Хронометр Истинного Времени: контекст torch.profiler, который собирает данные об исполнении на CPU и CUDA, запоминает размеры тензоров и стек вызовов.
- progress visualization: индикатор прогресса через click.progressbar для наглядного отображения выполнения по элементам VALIDATION_SET.
- выводы и отчеты: текстовые сообщения в STDOUT и таблица профилей prof.key_averages().table(...) с топ-15 по cpu_time_total.

## 5. Structural Decomposition (Декомпозиция структуры)

- Функции/элементы кода:
  - run_validation(modelName: str): основная функция CLI-ритуала, реализующая всю логику валидации и профилирования.
  - sentiment_analyzer: объект pipeline("sentiment-analysis", model=modelName).
  - VALIDATION_SET: набор данных, импортируемый из validation_data.
  - click.command и click.option: парадная оболочка CLI.
  - torch.profiler.profile: контекст для сбора профилей.
  - click.progressbar: визуализация процесса выполнения.
  - main guard: if **name** == "**main**": run_validation().
- Вспомогательные элементы:
  - Выводы через click.secho, click.echo — для цветного уведомления и текстовых отчётов.
  - prof.key_averages().table(...) — вывод таблицы топовых операций по CPU.

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: стандартный CPU-режим, при наличии доступен CUDA-устройствам для профилирования.
- Concurrency: синхронный цикл обработки; одна волна выполнения, без явной параллелизации.
- Dependencies: transformers, click, torch, validation_data (модуль VALIDATION_SET).
- Требования к окружению: поддержка PyTorch с доступом к profiler; установленный Hugging Face Transformers.

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env).
- DO NOT print raw data to console in production mode.
- DO NOT use synchronous network calls in the main loop.
- DO NOT wrap configuration files (.yaml, .json) into scripts (like Python/Bash).
- DO NOT change versions or paths during reconstruction.

## 7. Verification & Testing (Верификация)

```gherkin
Feature: Script Functionality
  Scenario: Successful execution
    Given VALIDATION_SET is available and a valid model-name resolves to an existing Hugging Face model
    When the script validate_model.py is executed with --model-name <valid-model-name>
    Then the script prints startup messages, runs validation across all samples, and outputs:
      - final accuracy
      - number of correct predictions
      - a top-15 CPU profiler table
    And the process terminates with exit code 0

  Scenario: Model loading failure
    Given --model-name points to a non-existent or inaccessible model
    When the script is executed
    Then an error message is printed and the script terminates gracefully (no crash)
```
