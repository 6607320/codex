# quest_18_1.py Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language: Python
- Frameworks: PyTorch, OpenCV, tqdm
- Context: ../AGENTS.md

## 2. Goal & Purpose (Легенда и Назначение)

Квест призывает духа Архивариуса Python и открывает Свиток Сторожевой ХимерЫ. Цель — вытащить из видеопотока эфирные координаты и уверенности обнаруженных объектов, накладывая на каждый кадр ограничивающие рамки и подписи, и сохранить итоговую визуальную хронику в виде трофеев на диске. Это не просто использование модели YOLOv5; задача — работать с результатами на низком уровне, превращая сырые отчеты в понятный визуальный отчет.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: JSON
- Schema:
  interface InputData {
  videoPath: string; // путь к входному видеоряду
  outputFolder?: string; // путь к папке вывода
  modelName?: string; // имя модели по умолчанию yolov5n
  confThreshold?: number; // порог уверенности
  }

### 3.2. Outputs (Выходы)

- Destination: File
- Format: JSON
- Success Criteria: Exit Code 0
- Schema:
  interface OutputResult {
  framesProcessed: number; // количество сохраненных кадров с аннотациями
  outputFolder: string; // папка вывода
  lastSavedFrame?: string; // путь к последнему сохраненному кадру (если есть)
  success: boolean; // флаг успешности
  error?: string; // описание хаоса при ошибке (если есть)
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Призыв духа модели: загрузить YOLOv5n через механизм PyTorch Hub, сообщить миру, что ритуал начался.
- Настроить характер духа: задать порог уверенности model.conf на 0.10 (чтобы ловить даже неуверенные находки).
- Проверить Кристалл Маны (Cuda): если доступен, перенести модель на GPU для ускорения ритуала.
- Подготовить киноплёнку: открыть видеопоток из файла test_video.mp4 и создать папку hunts_results для трофеев.
- Грань Слепой Охоты: пройти по всем кадрам видеопленки, для каждого кадра прогнать его через Химеру и получить вывод results.
- Явные Чернила: для каждого обнаруженного объекта взять координаты рамки, уверенность и класс; сформировать подпись вида "класс уверенность".
- Завести Руну Рисования: нарисовать прямоугольник вокруг каждого объекта на кадре и вписать подпись рядом с рамкой цветом некой гармонии.
- Трофеевые Доспехи: сохранить обработанный кадр как JPEG в папке вывода с именем result*frame*{frame_count}.jpg.
- Печать итогов: по завершении освободить ресурсы и сообщить, сколько кадров стало помечено и где они лежат.

### 4.2. Declarative Content (Для конфигураций и данных)

- Конфигурации и данные здесь не хранятся в явной скриптовой секции; основа — путь к файлу видео test_video.mp4, путь вывода hunt_results, порог уверенности 0.10, и выбор модели yolov5n через артефакт PyTorch Hub.

## 5. Structural Decomposition (Декомпозиция структуры)

- Основная сущность: модель YOLOv5n, загружаемая и используемая как объект-«дух».
- Виды действий: инициализация, конфигурация характера, инициация видеопотока, цикл обработки кадров, рисование рамок и подписи, сохранение кадров, завершение ритуала.
- Инструменты: файловая система (os), обработка видеопотока (cv2.VideoCapture), визуализация (cv2.rectangle, cv2.putText), прогресс-бар (tqdm).

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: оптимизировано под возможное использование CUDA; в противном случае работает на CPU.
- Concurrency: синхронный, последовательный цикл обработки кадров.
- Dependencies: PyTorch (для torch.hub.load и модели yolov5n), OpenCV (cv2), tqdm (индикатор прогресса).

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде в коде (используйте .env для секретов, если они нужны).
- DO NOT печатать сырые данные в консоль в продакшн-режиме.
- DO NOT использовать синхронные сетевые вызовы в главном цикле, если это не предусмотрено архитектурой.
- DO NOT оборачивать конфигурационные файлы (yaml, json) в скрипты.
- DO NOT менять версии библиотек или пути во время реконструкции артефакта.

## 7. Verification & Testing (Верификация)

1-2 сценария Gherkin (счастливый путь и один негативный)

Feature: Quest 18.1 Processing and Annotation
Scenario: Successful execution
Given video file "test*video.mp4" exists and is readable
When quest_18_1.py is executed
Then frames with bounding boxes are saved to "hunt_results" as "result_frame*\*.jpg"
And a summary message indicates the number of processed frames and the output folder

Scenario: Video file missing
Given video file "missing_video.mp4" does not exist
When quest_18_1.py is executed
Then no frames are saved and a clear error is reported without crashing

ИССЛЕДУЕМЫЙ АРТЕФАКТ: quest_18_1.py

ИСХОДНЫЙ КОД:

- Привлекать духа Архивариуса os и набор кинематографиста cv2 для работы с видеокадрами.
- Вызов PyTorch через torch.hub.load для загрузки yolov5n.
- Установка порога уверенности model.conf = 0.10 и миграция модели на CUDA, если доступна.
- Подключение видеоплёнки к порталу cv2.VideoCapture("test_video.mp4"), создание папки output hunt_results.
- Цикл обработки кадров с индикатором tqdm, чтение кадра, вычисление результатов детекции, рисование рамок и подписей на кадрах, сохранение финального кадра в hunt_results и увеличение счетчика.
- Освобождение ресурсов cap.release() и вывод итоговой сводки по кадрам.
