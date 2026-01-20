# quest_15_1.py Specification

## 1. Meta Information

- Domain: ML/NLP
- Complexity: Medium
- Language: Python
- Frameworks: PyTorch, Torchvision, Pillow
- Context: ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

Легенда: Этот Ритуал создает собственный загрузчик эфира изображений: кастомный Dataset, обернутый в DataLoader, который лениво подает данные пакетами для больших наборов. Модуль демонстрирует ключевые навыки конвейера данных, умение работать с нестандартными коллекциями изображений и пакетной обработкой во время обучения моделей.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

Source: CLI Args  
Format: JSON  
Schema:

- InputData
  - directory: string
  - transform?: TransformSpec

- TransformSpec
  - resize?: [number, number]
  - toTensor?: boolean

### 3.2. Outputs (Выходы)

Destination: STDOUT  
Format: JSON  
Success Criteria: Exit code 0  
Schema:

- OutputResult
  - batch_shape: [number, number, number, number]
  - labels: number[]
  - status: string

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

0. Импортируются нужные библиотеки: glob, os, PIL.Image, torch.utils.data (Dataset, DataLoader), torchvision.transforms.
1. Определяется чертеж Хранителя Свитков (CustomImageDataset), наследующийся от Dataset.
2. Инициализация Хранителя принимает directory и optional transform; внутри собирается каталог изображений путем поиска файлов с расширением PNG в указанной директории и сохраняется в self.image_paths; transform сохраняется для применения.
3. Реализуется метод **len**, возвращающий количество найденных изображений.
4. Реализуется метод **getitem**(idx): выбирает путь к изображению по индексу, загружает изображение, конвертирует к RGB, применяет transform, возвращает пару (image, label) где label равен индексу.
5. Формируется конвейер линз data_transform как композиция Transform: Resize до 64x64 и ToTensor.
6. Создается экземпляр image_dataset как CustomImageDataset(directory="generated_palette", transform=data_transform).
7. Создается image_dataloader как DataLoader(image_dataset, batch_size=4, shuffle=True).
8. Производится демонстрационный запуск: извлекается первая пачка через next(iter(image_dataloader)).
9. Выводится статус и форма тензора первой пачки и меток, завершение ритуала.

### 4.2. Declarative Content (Для конфигураций и данных)

- Каталог изображений: generated_palette
  - Поиск: \*.png
  - Формат изображений в загрузчике: RGB (после конвертации)
- Конвейер линз: data_transform
  - Первая линза: Resize к 64x64
  - Вторая линза: ToTensor
- Хранитель Свитков: CustomImageDataset
  - directory: generated_palette
  - transform: data_transform
- Подносчик Свитков: DataLoader
  - dataset: image_dataset
  - batch_size: 4
  - shuffle: True
- Ритуал вывода
  - Печать начал работы и итоговой информации
  - Ожидаемая форма первой пачки: (4, 3, 64, 64)
  - Метки первой пачки: четверка значений индексов

## 5. Structural Decomposition (Декомпозиция структуры)

- Класс CustomImageDataset
  - **init**(directory, transform=None)
  - **len**()
  - **getitem**(idx)
- Объекты конвейера
  - data_transform: трансформации изображений (Resize, ToTensor)
  - image_dataset: экземпляр CustomImageDataset
  - image_dataloader: экземпляр DataLoader
- Фронтальная логика
  - Печать начала ритуала
  - Получение и печать первой пачки изображений и меток
  - Финальная ритуальная печать об успешном создании конвейера

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU (петля выполнения не полагается на CUDA; может работать на CPU с возможной безопасной поддержкой GPU, если доступно).
- Concurrency: Синхронная загрузка по умолчанию (DataLoader без явного указания num_workers).
- Dependencies: PyTorch, Torchvision, Pillow

### 6.2. Prohibited Actions (Negative Constraints)

- НЕ хранить секреты в открытом виде в конфигурациях.
- НЕ печатать сырые данные в консоль в продакшн-режиме.
- НЕ использовать синхронные сетевые вызовы в главном цикле загрузки.
- НЕ оборачивать конфигурационные файлы (.yaml, .json) внутрь скриптов.
- НЕ менять версии библиотек или пути файлов в процессе реконструкции артефакта без необходимости.

## 7. Verification & Testing (Верификация)

Геркин сценарии:

```gherkin
Feature: Data loading pipeline
  Scenario: Successful first batch retrieval
    Given directory "generated_palette" contains PNG images
    When the DataLoader is created with batch_size 4 and shuffle enabled
    And the first batch is retrieved
    Then the batch images have shape (4, 3, 64, 64)
    And the batch contains 4 labels
    And the system prints "Первая пачка успешно получена!"

  Scenario: Empty dataset directory raises an error on access
    Given directory "generated_palette" exists but contains no PNG images
    When the DataLoader is created and a batch is requested
    Then an error indicating an empty dataset or invalid batch is raised
```

Артефакт исследуемый: quest_15_1.py
ЭФИР ДАННЫХ: кастомный ImageDataset, DataLoader с батчем 4, перемешивание, линзы трансформаций Resize и ToTensor, директория generated_palette с файлами PNG, первая пачка формы (4, 3, 64, 64) и соответствующие метки. Ритуал завершается сообщением об успешном создании конвейера данных.
