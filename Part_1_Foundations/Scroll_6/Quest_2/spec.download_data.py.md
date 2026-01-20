# download_data.py Specification

## 1. Meta Information

- Domain: ML/NLP
- Complexity: Medium
- Language: Python
- Frameworks: datasets
- Context: Independent Artifact

## 2. Goal & Purpose (Легенда)

Этот файл — канонический ритуал подготовки: он загружает и кеширует аудиоархив PolyAI/minds14 в локальную сокровищницу, чтобы основной квест мог мгновенно работать с данными, не тратя время на сеть. Задача — отделить тяжёлую загрузку от повторяющихся ритуалов обработки, чтобы будущие скрипты могли мгновенно черпать данные из готового кэша.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

Source: CLI Args
Format: JSON
Schema:
interface InputData {
dataset: string; // например: "PolyAI/minds14"
name: string; // например: "en-US"
split: string; // например: "train"
trustRemoteCode: boolean; // например: true
}

### 3.2. Outputs (Выходы)

Destination: STDOUT
Format: Text
Success Criteria: Exit Code 0
Schema:
interface OutputResult {
lines: string[]; // последовательность выводимых строк
exitCode: number; // завершение выполнения
}

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Преподнести зов подготовки: вывести в консоль сообщение о начале ритуала загрузки данных.
2. Призвать Великанa–гримуар load_dataset из Библиотеки Данцев (datasets). Это — единственный инструмент ритуала.
3. Вызвать load_dataset с параметрами:
   - dataset: "PolyAI/minds14"
   - name: "en-US"
   - split: "train"
   - trust_remote_code: True
     Говорят, что это заклинание связывает запрашиваемый архив с Великой Библиотекой Hugging Face и скачивает весь указанный архив, без потоковой передачи, сохраняя файлы в локальную сокровищницу (кэш) на диске.
4. Завершить ритуал: вывести финальное сообщение о завершении загрузки и доступности данных в кэше.
5. Концептуально — не использовать streaming=True, что означает скачивание всего архива целиком и сохранение его в кэш. Это обеспечивает мгновенный доступ последующих скриптов к уже скачанным данным.

### 4.2. Declarative Content (Для конфигураций и данных)

- Архив: PolyAI/minds14
- Язык набора: en-US
- Раздел набора: train
- Потоковая передача: отключена (streaming не используется)
- Резервное копирование: локальная сокровищница (кэш) на диске
- Флаги вызова: trust_remote_code = True

## 5. Structural Decomposition (Декомпозиция структуры)

- Внешняя зависимость: функция load_dataset из библиотеки datasets
- Главный оператор: вывод текстовых сообщений в консоль
- Действие: загрузка и кеширование набора
- Результат: данные доступны локально для повторного использования

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: обычная загрузка по сети с кешированием на диске; последующие запуски читают данные из локального кэша
- Concurrency: синхронный вызов (нет параллелизма внутри скрипта)
- Dependencies: datasets (и HuggingFace Hub как источник данных)
- Memory/CPU: стандартные ресурсы; загрузка может занимать сетевые и дисковые ресурсы, но не требует специализированной аппаратуры
- Streaming: отключённое, полномасштабное скачивание (без streaming=True)

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде (используйте .env, если есть секреты)
- DO NOT печатать сырые данные в консоль в боевом режимe
- DO NOT использовать синхронные сетевые вызовы внутри критических циклов сверх необходимого
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты
- DO NOT менять версии библиотек или пути реконструкции проекта без явной необходимости

## 7. Verification & Testing (Верификация)

Геркин-поводырь к тестам:

Feature: Script downloads and caches dataset
Scenario: Successful execution
Given Python environment with datasets library installed
When the script download_data.py is executed
Then the process exits with code 0
And the output contains: - "Начинаю ритуал предварительной загрузки данных..." - "Ритуал загрузки завершен. Все данные теперь в твоей локальной сокровищнице (кэше)."
And the PolyAI/minds14 en-US/train data is present in the local cache

Scenario: Network failure (error path)
Given network is unavailable
When the script download_data.py is executed
Then the process exits with a non-zero code
And an error trace is produced (due to unhandled exception in download)
And no data is cached locally

ИЗЯЩЕРСТВО АРТЕФАКТА: download_data.py

ИСХОДНЫЙ КОД (ОПИСАНИЕ, без копирования):  
Этот ритуал выполняет: подготовку кэширования архива PolyAI/minds14 через загрузку с Hugging Face с указанием en-US и split=train, без потоковой передачи, с выводами о начале и завершении ритуала. Ритуал сохраняет данные в локальную сокровищницу, чтобы последующие скрипты могли мгновенно использовать данные из кэша.
