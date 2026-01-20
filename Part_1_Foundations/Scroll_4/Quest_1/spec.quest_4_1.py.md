# quest_4_1.py Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Low
- Language: Python
- Frameworks: OpenCV (cv2)
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Легенда: Этот ритуал деконструкции кинопленки превращает движение в стаю неподвижных изображений. Монаха-исследователя учат видеть видео не как единое полотно, а как последовательность картинок, готовых к магическому анализу. Цель — автоматически открыть заданную кинопленку, извлечь кадры по одному и сохранить их в свитке файлов, чтобы последующие чародейства могли с ними играться.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: JSON
- Schema:
  interface InputData {
  video_path: string;
  output_folder: string;
  }

### 3.2. Outputs (Выходы)

- Destination: STDOUT | File
- Format: JSON | Text
- Success Criteria: Exit Code 0
- Schema:
  interface OutputResult {
  frames_extracted: number;
  output_folder: string;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Подхватить дух-скрипта: задать имя папки для извлечения кадров и создать её, если не существует.
2. Призвать Хранителя Эфира: открыть киноплёнку по заданному пути video_path через компонент OpenCV (cv2.VideoCapture).
3. Проверить доступность артефакта: если пленку открыть не удаётся, вывести сообщение об ошибке и направить взгляд в руководство по исправлению, затем остановить ритуал.
4. Поглотить поток: инициализировать счётчик кадров frame_count = 0 и в цикле читать кадры из видеопотока.
5. Пронумеровать и запечатлеть: для каждого прочитанного кадра сформировать имя файла frame\_<frame_count>.jpg внутри output_folder и сохранить кадр как изображение.
6. Пронумеровать прогрессы: увеличивать frame_count с каждым сохранённым кадром.
7. Завершить ритуал: освободить видеозахват и сообщить о завершении, указав сколько кадров извлечено и куда они помещены.

### 4.2. Declarative Content (Для конфигураций и данных)

- Видео источник: test_video.mp4
- Папка вывода кадров: frames/
- Формат кадров: .jpg
- Шаблон имени кадров: frame\_<число>.jpg
- Сообщение завершения: "Ритуал завершен. {frame_count} кадров извлечено в папку '{output_folder}'."

## 5. Structural Decomposition (Декомпозиция структуры)

- Функции: отсутствуют явные функции и классы. Логика реализована как линейная последовательность инструкций на верхнем уровне.
- Глобальные переменные/ресурсы:
  - output_folder: "frames"
  - video_path: "test_video.mp4"
  - video_capture: объект cv2.VideoCapture
  - frame_count: целое число, счётчик кадров
  - frame_filename: строка пути к сохраняемому кадру

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Sync
- Dependencies: Python 3, opencv-python (cv2)
- Platforms: кроссплатформенный сценарий (Windows, macOS, Linux)

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде (нет секретов в этом артефакте, но принципы соблюдения применимы).
- DO NOT печатать чувствительные данные в stdout в продуктивной среде (сообщения об ошибках и статусовые выводы держать в разумной форме).
- DO NOT использовать сетевые вызовы в основном цикле (поток чтения кадров локальный).
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты.
- DO NOT менять версии или пути во время реконструкции артефакта.

## 7. Verification & Testing (Верификация)

Геркин-сценарии:
Feature: Deconstruct Video into Frames

Scenario: Successful execution
Given a video file test_video.mp4 that can be opened
And the frames/ directory does not contain blocking artefacts
When the script quest_4_1.py is executed
Then a sequence of frames is saved as frames/frame_0.jpg, frames/frame_1.jpg, ... and the script prints the final message confirming N frames extracted and exits with code 0

Scenario: Failure to open video
Given the video_path test_video.mp4 cannot be opened
When the script quest_4_1.py is executed
Then an error message about opening the киноплёнку is printed and the script exits with a non-zero code

ИЗГОТОВЛЕННЫЙ АРТЕФАКТ: quest_4_1.py

Легендарные заметки:

- Скрипт-фокусник вызывает духи осмотра кадра за кадром и хранит их в виде отдельных изображений.
- Весь ритуал опирается на OpenCV и стандартный системный путь, не прибегая к иным источникам.
- Ритуал завершён, когда все кадры найдены и сохранены, либо при обнаружении неоткуда взяться кадру (хаос в начале).
