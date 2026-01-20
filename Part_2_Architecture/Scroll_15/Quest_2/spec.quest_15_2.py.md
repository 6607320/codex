# quest_15_2.py Specification

## 1. Meta Information

- **Domain:** Scripting
- **Complexity:** Medium
- **Language:** Python
- **Frameworks:** PyTorch, torchvision, Pillow
- **Context:** ../AGENTS.md

## 2. Goal & Purpose (Легенда)

Этот Ритуал призывает Духов-Помощников конвейера данных, чтобы наглядно доказать и измерить эффект от использования num_workers в DataLoader. Сначала проводится последовательный режим (num_workers=0), затем параллельный режим (num_workers=4). Временные задержки моделируют тяжёлую обработку данных и показывают, что параллелизм — не прихоть, а необходимость для эффективного использования GPU.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

```typescript
interface InputData {
  directory: string; // путь к директории с PNG-изображениями (Эфир)
  batchSize: number; // размер батча для конвейера (например, 4)
  shuffle: boolean; // перемешивать ли данные
  numWorkers: number; // число духов-помощников (потоков/процессов)
  transform?: {
    resize: [number, number]; // целевой размер изображений
    toTensor: boolean; // преобразовать в тензор
  };
}
```

### 3.2. Outputs (Выходы)

```typescript
interface OutputResult {
  executionTimes: {
    lazyLoading?: number; // время последовательной загрузки
    parallelLoading?: number; // время параллельной загрузки
  };
  logs?: string[]; // текстовые заметки ритуала
}
```

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- В начале собираются все необходимые артефакты из мира файлов: поиск PNG-файлов внутри указанной директории и фиксация их путей. Это выполняется через механизм, который называется “глобальный призыв” к источникам данных (glob).
- Затем создаётся заклинание-хранитель Свитков: класс CustomImageDataset, реализующий тройную логику:
  - Инициализация: сканирует директорию и формирует список путей к PNG-файлам; сохраняет опциональные линзы (transform).
  - Длина: возвращает число найденных файлов.
  - Получение элемента: по индексу выбирает путь к файлу, открывает изображение в RGB, накладывает искусственную задержку 0.05 секунды, применяет линзы (если заданы) и возвращает пару (образ, индекс Свитка).
- Подготовка лакмана линз: transform — Resize к 64 на 64 и конвертация в тензор (ToTensor).
- Подготовка архива данных: экземпляр CustomImageDataset получает directory="generated_palette" и transform=data_transform.
- Эксперимент 1: Ленивый Подносчик — конвейер DataLoader с numWorkers=0. Измеряется время полного пролистования через все батчи.
- Эксперимент 2: Эффективный Подносчик — конвейер DataLoader с numWorkers=4. Аналогично измеряется время пролистывания.
- В конце выводятся два временных результата и завершающее сообщение о параллелизме.

### 4.2. Declarative Content (Для конфигураций и данных)

- Эмблема: CustomImageDataset — Хранитель Свитков, собирает список PNG-изображений в указанной директории и подготавливает их для обработки.
- Эмблема: DataLoader — Подносчик Свитков, существующий в двух режимах: ленивый (numWorkers=0) и усиленный (numWorkers=4).
- Эмблема: Transforms — Линзы Восприятия, выполняющие Resize и конвертацию в тензор.
- Эмблема: generated_palette — Эфирная Скрижаль Данных, в которой лежат изображения.
- Эмблема: Хронометры — Хронометра, фиксирующие время выполнения каждого акта.
- Эмблема: Хаос задержки — искусственная задержка во время чтения изображения, имитирующая дорогостоящую обработку.

## 5. Structural Decomposition (Декомпозиция структуры)

- Класс: CustomImageDataset
  - Методы:
    - **init**(self, directory, transform=None)
    - **len**(self)
    - **getitem**(self, idx)
- Векторные компоненты ритуала:
  - data_transform: трансформации (Resize, ToTensor)
  - image_dataset: экземпляр CustomImageDataset
  - lazy_dataloader: DataLoader с num_workers=0
  - efficient_dataloader: DataLoader с num_workers=4
- Линия времени и выводы:
  - start_time, end_time для каждого эксперимента
  - Выводы на консоль: разница времени

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** Standard CPU, с возможной выгодой от параллельной загрузки на нескольких ядрах; базовые операции — в рамках PyTorch DataLoader.
- **Concurrency:** Async через DataLoader с многопоточностью/многопроцессностью (num_workers).
- **Dependencies:** PyTorch, torchvision, Pillow; стандартные модули Python: glob, os, time.

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде (.env без безопасного доступа).
- DO NOT печатать сырые данные в консоль в продакшене.
- DO NOT использовать синхронные сетевые вызовы в главном цикле ритуала.
- DO NOT оборачивать конфигурации (.yaml, .json) внутрь скриптов как статический код.
- DO NOT менять версии библиотек или пути конфигураций во время реконструкции.

## 7. Verification & Testing (Верификация)

Геркин-сценарии

Feature: [Script Functionality]
Scenario: Successful execution
Given директория "generated_palette" существует и содержит PNG-изображения
When ритуал quest_15_2.py выполняется
Then выводятся два временных значения — для ленивого и для эффективного поднощика
And завершающее сообщение указывает на очевидность параллелизма

Feature: [Script Functionality]
Scenario: Failure due to missing directory
Given директория отсутствует или недоступна
When ритуал quest_15_2.py выполняется
Then возникает ошибка доступа к файлу и выводится сообщение об ошибке
