# quest_7_2.py Specification

## 1. Meta Information

- Domain: ML/NLP
- Complexity: High
- Language: Python
- Frameworks: PyTorch, datasets, diffusers, peft, torchvision, transformers, tqdm
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

- Context for Creator: Этот файл реализует ритуал дообучения духа-Демиурга в гигантской модели Stable Diffusion с использованием LoRA, чтобы вплести наш собственный художественный стиль и экономить VRAM. Скрипт выстраивает цепочку подготовки данных, сборки артефактов (кодировщик текста, вариации VAE и сердце модели UNet), настройку параметров и кульминацию в виде сохранения обученной стилевой печати.
- Instruction for AI: Высокий уровень намерения — показать, как через локальный набор палитр и минимальные ресурсы можно добиться внедрения нового визуального стиля в Diffusion-модель с минимальной нагрузкой на память.

Краткая легенда: Это руководство-практикум по созданию уникального стилизатора через элементарный ритуал обучения LoRA на модели Stable Diffusion, с акцентом на экономию маны VRAM и управляемую настройку обходных механизмов.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: JSON
- Schema:
  interface InputData {
  datasetDir: string; // путь к палитре (palette_dir)
  outputDir: string; // путь к папке результата (artist_seal)
  modelId: string; // имя базовой модели (CompVis/stable-diffusion-v1-4)
  resolution: number; // целевое разрешение (256)
  trainBatchSize: number; // размер батча (1)
  gradientAccumulationSteps: number; // накопление градиентов (4)
  learningRate: number; // скорость обучения (0.0001)
  numTrainEpochs: number; // число эпох (100)
  }

### 3.2. Outputs (Выходы)

- Destination: File
- Format: JSON
- Success Criteria: Exit code 0; файл/папка артефакта создан
- Schema:
  interface OutputResult {
  success: boolean;
  outputDir: string; // путь к сохранённой стилевой печати
  epochsCompleted: number;
  finalLoss?: number;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Инициализация ритуала: объявляются ключевые артефакты и параметры — имя базового духа-Демиурга, путь к палитре, путь вывода, размеры изображения и параметры обучения.
- Настройка магии параметров: фиксируются разрешение, размер порции, шаги накопления градиентов, скорость и число эпох, а также целевые компоненты модели и формат перемещений на Кристалл Маны (CUDA).
- Подготовка Учебника: загружаются картины из локального набора палитр, конструируется конвейер трансформаций (изменение размера, горизонтальное отражение, конвертация в тензор, нормализация) и применяется к каждой порции данных. Создается DataLoader с перемешиванием.
- Призыв Компонентов и Создание Блокнота: загружаются и инициализируются элементы Stable Diffusion — токенизатор и кодировщик текста, вариационный автоэнкодер (VAE) и UNet. Создается конфигурация LoRA и присоединяется к UNet через PEFT, превращая UNet в носителя стилевой печати.
- Ритуал Наставления: все действующие элементы переводятся на CUDA, UNet переводится в режим обучения, настраивается оптимизатор AdamW и планировщик шума DDPM.
- Подготовка стиля: определяется магическое слово-руна (train_prompt) и его текстовые руны (prompt_ids) переводятся на CUDA.
- Основной цикл обучения: для каждой эпохи и каждого батча:
  - В режиме без градиентов латентные представления чистых изображений получают через VAE.
  - Текстовый контекст кодируется текстовым Encoder-ом.
  - Генерируется случайный шум и временной шаг для данными латентами.
  - Шум добавляется к латентам, UNet предсказывает этот шум.
  - Вычисляется среднеквадратическая ошибка между предсказанным шумом и реальным шумом.
  - Ошибка обратно распространяется; накапливается градиент, обновление выполняется каждые несколько шагов (gradient_accumulation_steps).
  - По завершении каждой эпохи выводится текущая ошибка.
- Финализация: после завершения ритуала стираются старые следы и сохраняется обученная стилем печать (LoRA) в output_dir. Сообщение об успехе.

### 4.2. Declarative Content (Для конфигураций и данных)

- Базовый дух модели: CompVis/stable-diffusion-v1-4
- Палитра (учебный набор): ../Quest_1/generated_palette
- Папка вывода артефакта: artist_seal
- Разрешение: 256
- Бэтч: 1
- Gradient accumulation: 4
- Learning rate: 1e-4
- Эпохи: 100
- Пепел трансформаций: Resize(256x256), RandomHorizontalFlip, ToTensor, Normalize [-1, 1]
- Токенизатор: CLIPTokenizer
- Текстовый кодировщик: CLIPTextModel
- VAE: AutoencoderKL
- UNet: UNet2DConditionModel
- LoRA-конфигурация: r=16, lora_alpha=32, target_modules=[to_q, to_k, to_v, to_out.0], lora_dropout=0.05, bias="none"
- Ключевые параметры обучения и шумовых расписаний: DDPMScheduler

## 5. Structural Decomposition (Декомпозиция структуры)

- Файлы и артефакты
  - apply_transforms (функция)
  - dataset (объект) и train_dataloader (порции данных)
- Компоненты модели
  - AutoencoderKL (VAE)
  - CLIPTokenizer (Токенизатор)
  - CLIPTextModel (Текстовый кодировщик)
  - UNet2DConditionModel (UNet)
  - LoraConfig (класс конфигурации LoRA)
  - get_peft_model (функция присоединения LoRA к UNet)
- Обучение и утилиты
  - AdamW (оптимизатор)
  - DDPMScheduler (расписание шума)
  - torch (базовый пакет)
  - tqdm (индикатор прогресса)
  - transformers (инструменты для текста)
  - datasets (загрузка датасета)
  - torchvision.transforms (преобразования)
- Параметры и данные
  - палитра, модель, размер входа, шаги эпох, пр. параметры

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: настроено под GPU с поддержкой float16 и CUDA; размер батча мал (1) и 256x256 разрешение, чтобы экономить VRAM и ускорить обучение на слабых устройствах.
- Concurrency: синхронный обучающий цикл в рамках одного процесса; нет параллельной обработки данных на уровне ядра.
- Dependencies: PyTorch, datasets, diffusers, peft, torchvision, transformers, tqdm — должны быть доступны в окружении.

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде — использовать защищённые источники конфигурации (.env/secret vault).
- DO NOT печатать сырые данные в консоль в продакшене.
- DO NOT использовать синхронные сетевые вызовы в главном цикле.
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты.
- DO NOT менять версии библиотек или пути в процессе реконструкции артефакта.

## 7. Verification & Testing (Верификация)

### Герхин-Сценарии

Feature: Script Functionality
Scenario: Successful execution
Given набор палитры доступен по пути "../Quest_1/generated_palette" и папка вывода доступна для записи
When запускается quest_7_2.py
Then артефакт стилевой печати сохраняется в папке "artist_seal" и завершается с кодом 0

Scenario: Missing dataset path
Given указанный путь к палитре недоступен
When запускается quest_7_2.py
Then процесс завершается с ошибкой и артефакт не создаётся

ИНСТРУКЦИИ ПО РАЗДЕЛАМ (дабы Великий Хранитель не допускал искажений):

- В Заголовке: quest_7_2.py Specification
- 1. Meta Information: заполни Стихию (ML/NLP), контекст — Independent Artifact, сложность — High, язык — Python, Frameworks — перечислены выше.
- 2. Goal & Purpose: сразу легенда на русском и цель модуля.
- 3. Interface Contract: строго TypeScript интерфейсы для входов и выходов, поля без доп. пояснений.
- 4. Implementation Details: шаг за шагом, без квадратных скобок и без вставки кода.
- 5. Declarative Content: преобразуй данные в маркированный список с эмодзи; не копируй код.
- 6. System Context & Constraints: выпиши технические ограничения и запреты.
- 7. Verification & Testing: предложи 1–2 сценария на языке Геркин.

Артефакт: quest_7_2.py — воплощение ритуала стилизации через LoRA и дообучение UNet в Stable Diffusion.
