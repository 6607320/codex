# vision_spell.py Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Low
- Language: Python
- Frameworks: transformers, datasets
- Context: ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

Опишем легенду к Ритуалу. Этот модуль является минимальным демонстрационным примером упаковки Python-пакета и канонической структуры проекта arcane_wizardry. Он иллюстрирует, как можно импортировать и использовать внешние артефакты (библиотеки для загрузки данных и распознавания образов) внутри простого скрипта, при этом сохранять ясную канву проекта, Ready-to-pipe после установки. В основе—клик по порталу к CIFAR-10, воплощённый через артефакт-амулет и Всевидящее Око, чтобы продемонстрировать рабочий цикл: подготовка эфира, призыв духа и вывод вердикта.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args | STDIN | API Request | Kafka Topic | Smart Contract Call
- Format: JSON | Text | Binary | Stream
- Schema:

```typescript
interface InputData {
  // В текущем виде входы отсутствуют; скрипт не принимает внешних аргументов как части исполнения
}
```

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: JSON | CSV | Text
- Success Criteria: Exit Code 0
- Schema:

```typescript
interface OutputResult {
  message?: string;
  predictions?: { label: string; score: number }[];
  trueLabel?: string;
}
```

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Акт 1: Подготовка Гримуаров
  - Подключаем артефакт Библиотекаря для доступа к учебным эфирным данным и амулету распознавания.
  - Подключаем универсальный амулет для простого использования моделей.
- Акт 2: Призыв Магического Образа
  - Открываем портал к архиву CIFAR-10, используя потоковый доступ, чтобы не тащить весь эфир целиком.
  - Берём первый попавшийся образец из потока; внутри образа лежит изображение и его истинная субстанция, полученная через паспорт архива.
- Акт 3: Ритуал Распознавания
  - Призыв всевидящего духа через амулет image-classification с моделью google/mobilenet_v2_1.0_224.
  - Пропитываем образ в амулет и получаем список топ-5 вероятностей с их уверенностями.
- Акт 4: Демонстрация Результата
  - Выводим заголовок и печатаем вердикт духа — набор вероятностей и соответствующих предсказаний, сопоставляемых с истинной сутью образа.

### 4.2. Declarative Content (Для конфигураций и данных)

- Этот ритуал не хранит жестких конфигурационных файлов внутри кода; он черпает параметры из стандартной конфигурации библиотек hf/datasets и transformers. Основной эфир (Эфир) — это поток CIFAR-10 и модель MobileNetV2.

## 5. Structural Decomposition (Декомпозиция структуры)

- Функции: загрузка данных, извлечение образца, создание и использование пайплайна классификации, вывод результатов.
- Классы: нет явной пользовательской иерархии; используются готовые функции из библиотек.
- Для конфигов: основные логические блоки — загрузка эфира (dataset), призыв духа (vision_spirit pipeline), демонстрация вердикта (print).

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Async (за счёт внутреннего поточного API datasets) в рамках синхронного вызова скрипта
- Dependencies: datasets, transformers
- Объем данных: используется только первый образец из потока CIFAR-10; минимальная нагрузка памяти за счёт streaming=True

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap configuration files (.yaml, .json) into scripts (like Python/Bash)
- DO NOT change versions or paths during reconstruction

## 7. Verification & Testing (Верификация)

```gherkin
Feature: Vision Spell Script
  Scenario: Successful execution
    Given environment has Python and required libraries installed and CIFAR-10 accessible
    When vision_spell.py runs and processes the first sample from CIFAR-10
    Then a list of top-5 predicted labels with scores is printed to STDOUT and no errors occur

  Scenario: CIFAR-10 streaming failure
    Given network or streaming access to CIFAR-10 is unavailable
    When vision_spell.py attempts to load the dataset
    Then the script fails gracefully with informative logs and exits non-zero
```
