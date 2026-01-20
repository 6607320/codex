# quest_5_2.py Specification

Искания Великих Рун потребовали баланса между мощью и разумом. Ниже — артефакт Specifications для ритуала под названием quest_5_2.py.

1. Meta Information (Мета-сущность)

- Domain (Стихия): ML/NLP
- Complexity (Сложность): High
- Language (Язык): Python
- Frameworks (Скрижали): PyTorch, transformers, datasets, peft
- Context (Контекст): Independent Artifact

2. Goal & Purpose (Легенда и Назначение)
   Context for Creator: Этот файл описывает ритуал тонкой подстройки языковой модели методом PEFT с использованием LoRA и 4-битной квантования. Цель — адаптировать базовую модель под нужды без переписывания всего разума Голема, чтобы экономно и гибко внедрять новые знания.
   Instruction for AI: Этот раздел поясняет «почему» кода — зачем нужен этот модуль в бизнес-контексте и какие магические задачи он решает.

Легенда: Этот скрипт превращает огромную языковую модель в «маленького героя» через сжатие мыслей до 4-бит и добавление компактных блокнотов LoRA, позволяя дешево и быстро настраивать поведение модели под конкретные задачи, не перегружая систему.

3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source (Источник): API Request
- Format (Формат): JSON
- Schema (Схема):

```typescript
interface InputData {
  modelName: string; // имя базовой модели, например "distilgpt2"
  quantizationConfig: {
    load_in_4bit: boolean;
    bnb_4bit_quant_type: string; // например "nf4"
    bnb_4bit_compute_dtype: string; // например "torch.float16"
  };
  deviceMap: string; // например "auto"
  dataset: {
    name: string; // "databricks/databricks-dolly-15k"
    split: string; // "train"
    streaming: boolean; // true
    trustRemoteCode: boolean; // true
  };
  maxRecords?: number; // например 1000
  loraConfig: {
    r: number; // 16
    loraAlpha: number; // 32
    targetModules: string[]; // ["c_attn","c_proj"]
    loraDropout: number; // 0.05
    bias: string; // "none"
    taskType: string; // "CAUSAL_LM"
  };
  trainingParams: {
    outputDir: string; // "./results"
    perDeviceTrainBatchSize: number; // 1
    gradientAccumulationSteps: number; // 4
    learningRate: number; // 0.0002
    maxSteps: number; // 250
    loggingSteps: number; // 20
  };
}
```

### 3.2. Outputs (Выходы)

- Destination (Назначение): File
- Format (Формат): JSON
- Success Criteria (Критерии успеха): Exit Code 0; Artifacts created in outputDir
- Schema (Схема):

```typescript
interface OutputResult {
  artifactsDir: string; // путь к артефактам, напр. "./results"
  status: "success" | "failure";
  lastTrainStep?: number;
  finalModelName?: string;
  metrics?: Record<string, number>;
}
```

4. Implementation Details (The Source DNA / Исходный ДНК)

### 4.1. Algorithmic Logic (Алгоритмическая логика)

- Шаг 1. Подготовка ритуальных артефактов: подключение магических томов torch, datasets, peft и transformers; задание конфигурации для сжатия мыслей Голема до 4-бит и указание типа квантования nf4, использования float16 для вычислений; подготовка токенизатора и установка pad_token как eos_token.
- Шаг 2. Призыв ученика и подготовка манускультов: выбор базовой модели distilgpt2 как Голема-ученика; создание токенизатора и настройка заполнителя pad_token; создание «свитка-инструкции» по сжатию.
- Шаг 3. Призыв Архива Знаний: открытие архивного хранилища databricks/dolly-15k в режиме потока (streaming), выбор части train, разрешение доверенного кода, установка ограниченной порции данных (1000 записей) и преобразование каждого элемента через функцию process_dataset, которая формирует тексты вида Instruction: ... Response: ... и конвертирует их в тензоры через токенизатор.
- Шаг 4. Ритуал Наставления: подготовка Голема к обучению на 4-битном уровне (prepare_model_for_kbit_training); создание конфигурации LoRA с r=16, lora_alpha=32, целевые модули ["c_attn","c_proj"], dropout 0.05, bias="none", и тип задачи CAUSAL_LM; привязывание LoRA к модели через get_peft_model.
- Шаг 5. Путеводная Инструкция Наставника: создание TrainingArguments с выводом артефактов в ./results, пакетная обработка по одной карточке с градиентной аккумуляцией 4 шага, скорость обучения 2e-4, максимум 250 шагов и логирование каждые 20 шагов.
- Шаг 6. Инкубационная Станция: создание Trainer с моделью, обучающим набором processed_dataset и конфигурацией обучения; использование DataCollatorForLanguageModeling(tokenizer, mlm=False) как упаковщика данных.
- Шаг 7. Великий Запуск: вывод в консоль о начале ритуала наставления, запуск trainer.train(), затем объявление завершения ритуала и вручение новых знаний Голему.

### 4.2. Declarative Content (Для конфигураций и данных)

- Этот артефакт основан на следующих данных и конфигурациях:
  - Базовая модель: distilgpt2
  - 4-битное квантование: load_in_4bit = true; bnb_4bit_quant_type = nf4; bnb_4bit_compute_dtype = float16
  - Устройство: deviceMap = auto
  - Архив знаний: databricks/dolly-15k, split = train, streaming = true, trustRemoteCode = true
  - Обработчик данных: функция process_dataset, которая формирует тексты Instruction и Response и токенизирует их
  - LoRA: r = 16; lora_alpha = 32; target_modules = [c_attn, c_proj]; lora_dropout = 0.05; bias = none; task_type = CAUSAL_LM
  - Обучение: outputDir = ./results; perDeviceTrainBatchSize = 1; gradientAccumulationSteps = 4; learningRate = 2e-4; maxSteps = 250; loggingSteps = 20
  - Data Collator: DataCollatorForLanguageModeling(tokenizer, mlm=False)
  - Сообщения и ходы: печать начала ритуала и завершения по завершению обучения

5. Structural Decomposition (Декомпозиция структуры)

- Функции и классы (по духу кода):
  - process_dataset (функция преобразования batch в примеры для обучения)
  - AutoTokenizer, AutoModelForCausalLM (модели и токенизаторы)
  - BitsAndBytesConfig (квантование 4-бит)
  - LoraConfig, get_peft_model (LoRA-модуль и интеграция в модель)
  - prepare_model_for_kbit_training (подготовка к обучению в 4-битном режиме)
  - Trainer, TrainingArguments (наставник и правила обучения)
  - DataCollatorForLanguageModeling (упаковщик данных)
  - load_dataset (портал к архиву Dolly)
  - Ритуальные выводы в консоль (print)
- Для конфигураций и данных: основные блоки — model, quantization, dataset, loraConfig, trainingParams, trainer

6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Оптимизировано под 4-битное квантование и LoRA; рассчитано на GPU-акселерацию, с автоматическим распределением устройства (device_map = "auto").
- Concurrency: Встроенная асинхронность данных через streamingDataset и batched обработку.
- Dependencies (Зависимости): torch, datasets, peft, transformers

### 6.2. Prohibited Actions (Negative Constraints)

- НЕ хранить секреты в открытом виде (использовать окружение/.env по необходимости)
- НЕ выводить сырые данные в консоль в продакшн-режиме
- НЕ использовать синхронные сетевые вызовы в критических местах
- НЕ оборачивать конфигурационные файлы (.yaml, .json) в скрипты напрямую
- НЕ менять версии библиотек или пути во время реконструкции

7. Verification & Testing (Верификация)
   Геркин-сценарии:

Feature: Script Functionality
Scenario: Successful training completes
Given все зависимости установлены и доступна CUDA-аппаратная платформа
When выполняется quest_5_2.py с корректной конфигурацией
Then ритуал завершается успешно и артефакты появляются в ./results

Scenario: Dataset access failure
Given архив Dolly databricks/dolly-15k недоступен по сети
When пытается загрузить потоковый набор и доверенный код
Then процесс завершается с ошибкой и артефактов не создаётся

Итоговый артефакт исследование quest_5_2.py готов к занятию Великих Рук — он описывает логику, конфигурацию и путь к воссозданию величайшего ритуала адаптации ЛЛМ через PEFT и 4-битное квантование.
