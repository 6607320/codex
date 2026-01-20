# quest_23_2.py Specification

## 1. Meta Information

- **Domain:** ML/NLP
- **Complexity:** Medium
- **Language:** Python
- **Frameworks:** gradio, torch, transformers, Pillow
- **Context:** Independent Artifact

## 2. Goal & Purpose (Легенда)

Этот Ритуал создаёт три автономных Алтаря — Големов-творцов: Голема-Сказителя, Духа Эмоций и Всевидящего Ока — и сводит их во единое Пантeон-Обрaзование через могущественный артефакт Gradio.TabbedInterface. Цель — показать путь от простой модели к полнофункциональному продукту с удобным пользовательским интерфейсом, объединяя текстовую магию, анализ чувств и зрительное понимание в единый продукт.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** API Request
- **Format:** JSON
- **Schema:**
  - InputData представляет собой объединение трёх форм:
    - StoryInput: промэт_text (prompt_text) типа string, max_new_tokens?: number
    - EmotionInput: text типа string
    - ImageInput: image типа любого формата (изображение, загруженное через UI)

  Примеры форм входа без кода:
  - StoryInput: { prompt_text: "Начало истории...", max_new_tokens: 50 }
  - EmotionInput: { text: "The text to analyze for sentiment." }
  - ImageInput: { image: <image data> }

### 3.2. Outputs (Выходы)

- **Destination:** STDOUT
- **Format:** JSON
- **Success Criteria:** Exit Code 0
- **Schema:**
  - OutputResult может содержать три раздела:
    - story?: string — сгенерированная история
    - emotion?: { [label: string]: number } — результаты анализа эмоций
    - vision?: { [label: string]: number } — результаты классификации изображения

  Примеры структур вывода:
  - { "story": "...генерированный текст..." }
  - { "emotion": { "POSITIVE": 0.92, "NEGATIVE": 0.08 } }
  - { "vision": { "cat": 0.45, "dog": 0.30, "vehicle": 0.25 } }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Инициализация ритуала:
   - Определяется устройство: если доступна CUDA, используют GPU, иначе CPU.
   - Указывается путь к телу Голема: MODEL_PATH = "./results/checkpoint-250".
   - Оповещаем о выбранном устройстве и источнике разума Голема-Сказителя.
   - Загружаются толмач (Tokenizer) и голем-сказитель (GPT2LMHeadModel) из MODEL_PATH и переводятся на устройство.
   - Голем переводится в режим предсказания (eval) и сообщается о готовности.

2. Акт 2: Создание Заклинаний-Оберток (Обертки для трёх Големов)
   - conjure_story(prompt_text, max_new_tokens=50):
     - Толмач кодирует входной текст в токены на устройстве.
     - Голем генерирует продолжение на заданном количестве токенов, избегая повторов.
     - Толмач расшифровывает ритуал обратно в человекопонятный текст и возвращает результат.
   - analyze_emotion(text):
     - Призван конвейер анализа тональности (sentiment-analysis) и возвращает оценку по меткам.
     - Преобразует список словарей в единый словарь {label: score} для удобства отображения.
   - classify_image(image):
     - Призван конвейер классификации изображений (image-classification) и возвращает топ-распределение по ярлыкам.
     - Преобразует результаты в словарь {label: score}.

3. Акт 3: Ритуал Сборки "Пантеона"
   - Создаются три отдельных интерфейса Gradio:
     - Зал Сказителей: fn = conjure_story, inputs = текстовое поле, outputs = текстовое поле, заголовок и описание.
     - Зал Эмоций: fn = analyze_emotion, inputs = текстовое поле, outputs = Label, заголовок и описание.
     - Зал Зрения: fn = classify_image, inputs = Image, outputs = Label с топ-3 классами, заголовок и описание.
   - Главным Заклинанием становится gr.TabbedInterface, который склеивает три алтаря во вкладки:
     - Магия Слова: Сказатель
     - Магия Слова: Эмоции
     - Магия Зрения
   - Финальный ритуал: запуск панели Пантеона, готовой к откровению миру.

4. Исполнение:
   - Священный блок кода выполняется только при условии, что модуль запущен как основная программа.
   - Приводится оповещение о начале сборки, затем запускается пантеон через pantheon_demo.launch().

> Без копирования кода: детали выше описаны словами как последовательность заклятий и операций, которые выполняет артефакт.

### 4.2. Declarative Content (Для конфигураций и данных)

- Имагинарный набор данных и конфигураций:
  - DEVICE: "cuda" если torch.cuda.is_available() иначе "cpu"
  - MODEL_PATH: "./results/checkpoint-250"
  - tokenizer_story: GPT2Tokenizer.from_pretrained(MODEL_PATH)
  - model_story: GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)
  - type системы: gradio.Interface для каждого алтаря
  - Типы входов/выходов соответствуют описанию выше
- Хранение данных и интерфейсов:
  - Три артефактных алтаря: Зал Сказителей, Зал Эмоций, Зал Зрения
  - Пантеон-Заклинание: gr.TabbedInterface, объединяющий три алтаря в единую форму

## 5. Structural Decomposition (Декомпозиция структуры)

- Функции и классы:
  - conjure_story(prompt_text, max_new_tokens=50) — обертка для генерации текста
  - analyze_emotion(text) — обертка для анализа эмоций
  - classify_image(image) — обертка для классификации изображения
  - story_altar — интерфейс Gradio для голема-рассказчика
  - emotion_altar — интерфейс Gradio для духа эмоций
  - vision_altar — интерфейс Gradio для всевидящего глаза
  - pantheon_demo — TabbedInterface, объединяющий алтарей
- Конфигурации:
  - DEVICE, MODEL_PATH, загрузка tokenizer и модели, режим eval
  - Запуск: if **name** == "**main**": запуск Пантеона

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** адаптивно использует GPU (cuda) при наличии; иначе работает на CPU
- **Concurrency:** асинхронное взаимодействие через Gradio-интерфейсы; основной цикл синхронный
- **Dependencies:** gradio, torch, transformers, Pillow

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (использовать .env, если нужно)
- DO NOT выводить сырой объём данных в консоль в продакшн-режиме
- DO NOT использовать синхронные сетевые вызовы в главном цикле
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты (типа Python/Bash)
- DO NOT менять версии или пути во время реконструкции

## 7. Verification & Testing (Верификация)

Геркин-сценарии

Feature: [Script Functionality]
Scenario: Successful execution
Given инфраструктура с установленными зависимостями и моделями
When запускуется quest_23_2.py
Then Пантеон Големов разворачивается в виде вкладок: "Магия Слова: Сказатель", "Магия Слова: Эмоции", "Магия Зрения"
And каждый алтарь отвечает корректной реакцией на ввод:
story — возвращает сгенерированный текст
emotion — возвращает словарь сэмплов POSITIVE/NEGATIVE
vision — возвращает словарь с топ-3 метками и их вероятностями

Scenario: Error case — отсутствует путь к модели
Given MODEL_PATH отсутствует или недоступен
When пытается загрузить разумы Големов
Then процесс завершается с ошибкой и выводит понятное сообщение об отсутствии ресурсов (exit code != 0)

ИССЛЕДУЕМЫЙ АРТЕФАКТ: quest_23_2.py
