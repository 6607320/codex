# quest_12_1.py Specification

1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language: Python
- Frameworks: PyTorch, torchvision, tqdm
- Context: ../AGENTS.md

2. Goal & Purpose (Легенда)
   Этой рукописью вычерчен путь к созданию Генератора Снов — первого артефакта, способного порождать новые изображения цифр из пустоты вариаций. Артефакт реализует Variational Autoencoder: Магический Пресс (Энкодер) сжимает рукописные цифры в пространство сновидений, а Проектор Снов (Декодер) выпускает из него новые образы. Мастерская скрипывает затемной баланс BCE и KLD, чтобы пространство было гладким и управляемым. Цель — продемонстрировать архитектуру автоэнкодеров и их вариационную магию в практической постановке: обучение на MNIST и последующая генерация сновидений в виде 28x28 изображений.

3. Interface Contract (Интерфейсный Контракт)

3.1. Inputs (Входы)

- Source: CLI Args
- Format: JSON | Text | Binary | Stream
- Schema:
  interface InputData {
  }

  3.2. Outputs (Выходы)

- Destination: File
- Format: PNG (изображение), а также статус выполнения
- Success Criteria: Exit с кодом 0 и созданный файл
- Schema:
  interface OutputResult {
  path: string;
  success: boolean;
  fileName?: string;
  mimeType?: string;
  message?: string;
  }

4. Implementation Details (The Source DNA / Исходный Код)

4.1. Algorithmic Logic (Для исполняемого кода)

- Подготовка заклинаний окружения: импортируются основные мантры маны и вспомогательные ингредиенты (Os, torch, nn, F, optim, datasets, transforms, save_image, tqdm).
- Настройка ритуала: задаются параметры пространства сновидений (latent_dim = 2), длительность обучения (epochs = 10) и сила пачки (batch_size = 128).
- Подготовка учебника-рукописей: загружается MNIST в трансформацииToTensor, создаётся подносчик (DataLoader) инициализирующий упаковку данных.
- Чертеж артефакта: создаётся двухчастный дизайн — Пресс (Энкодер) и Проектор (Декодер) внутри класса VAE, с линейными слоями, активациями и методами:
  - encode: пропускает вход через первый слой и возвращает две величины: mu и logvar.
  - reparameterize: осуществляет выборку из распределения через факторизационный трюк (mu, logvar) и случайный шум.
  - decode: восстанавливает изображение из точки в скрытом пространстве.
  - forward: связывает кодирование, репараметризацию и декодирование в единый цикл.
- Мера ошибок (The Balance of Chaos): loss_function сочетает бинарную кросс-энтропию (BCE) и регуляторную Kullback-Leibler дивергенцию (KLD).
- Руны наставления (Обучение): создаётся модель на CUDA, оптимизатор Adam, затем цикл эпох:
  - режим обучения включён, для каждого батча вычисляется реконструкция и хранилище mu/logvar, вычисляется общая потеря, выполняется обратное распространение и шаг оптимизации.
  - после эпохи печатается средняя ошибка.
- Магия творения (Сновидение): после обучения исполняется сражение с тенью — образцы из 64 латентных точек (0-центрированные) проходят через декодер и сохраняются как изображение:
  - создаётся каталог dreams
  - сохраняется файл dreams/dream_sample.png, в котором 64 изображения упакованы в сетку

    4.2. Declarative Content (Для конфигураций и данных)

- Инвентарь проекта: набор артефактов и параметров, необходимых для повторения
  - latent_dim: 2
  - epochs: 10
  - batch_size: 128
  - Уроки MNIST: трансформации ToTensor; путь данных "./data"
  - Архитектура: Encoder с fc1 (784 -> 400), fc21 (400 -> latent_dim), fc22 (400 -> latent_dim); Decoder с fc3 (latent_dim -> 400), fc4 (400 -> 784)
  - Фрагменты магии: ReLU, Sigmoid на финальном слое
  - Потери: BCE + KLD
  - Оптимизатор: Adam(lr=1e-3)
  - Умножители мощности: model.to("cuda"), data.to("cuda"), sample.decode на CPU для сохранения
  - Выход: dreams/dream_sample.png
  - Ожидаемая средняя ошибка: печатается после каждой эпохи
  - Верификация: no_grad контекст на финальной стадии и генерация выборки

5. Structural Decomposition (Декомпозиция структуры)

- Класс VAE
  - **init**: инициализация слоёв Encoder (fc1, fc21, fc22) и Decoder (fc3, fc4)
  - encode(x): проход через первый слой и ReLU, возврат mu и logvar
  - reparameterize(mu, logvar): сэмплинг z из нормального пространства через стандартное отклонение
  - decode(z): проход через декодер и сигмоид
  - forward(x): сборка полного цикла: сжатие -> репараметризация -> восстановление
- Функция loss_function(recon_x, x, mu, logvar): BCE для реконструкции и KLD для регуляризации, итоговая потеря — сумма
- Основной набор (скелет ритуала): создание модели, настройка оптимизатора, цикл эпох, обучение на DataLoader
- Этап генерации: выборка z из стандартного нормального распределения, декодирование, сохранение изображения
- Вспомогательные элементы: MNIST загрузчик, transform, save_image, tqdm прогресс-бар

6. System Context & Constraints (Системный контекст и Ограничения)

6.1. Technical Constraints

- Performance: Оптимизирован под графический процессор с CUDA; память под загрузку MNIST и размеры сети умеренные
- Concurrency: Синхронный обучающий цикл (нет асинхронных задач в ядре)
- Dependencies: PyTorch, torchvision, tqdm
- Hardware: Требуется доступ к CUDA-устройству (model.to("cuda"))

  6.2. Prohibited Actions (Negative Constraints)

- НЕ хранить секреты в открытом виде (использовать .env для секретов)
- НЕ выводить сырые данные в консоль в продакшн-режиме
- НЕ использовать синхронные сетевые вызовы в главном цикле
- НЕ оборачивать конфигурационные файлы (.yaml, .json) в скрипты
- НЕ менять версии библиотек или путей во время реконструкции

7. Verification & Testing (Верификация)

Геркин-сценарии

Feature: Dream Generator Script Operation
Scenario: Successful training and dream image generation
Given the MNIST dataset is loaded and the model is configured
When training completes for the defined epochs and a set of latent vectors is decoded
Then the file dreams/dream_sample.png is created and contains 64 generated digits

Scenario: Dataset loading failure
Given the MNIST dataset cannot be downloaded or loaded
When the script is executed
Then an error is raised indicating dataset loading failure and the process stops gracefully

Итоговый артефакт: quest_12_1.py

Independent Artifact
