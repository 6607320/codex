# === quest_18_2.py (Финальная Версия с "Памятью" и Полными Комментариями) ===
# Квест: 18.2 - Подсчет Объектов
# Цель: Усовершенствовать наш скрипт, добавив простейшую "память" (трекинг),
# чтобы решить проблему "мерцающей" детекции и считать объекты на каждом кадре.

# Призываем помощника 'os' для работы с папками.
import os

# Призываем "Набор Кинематографиста" для работы с видео и рисования.
import cv2

# --- Акт 1: Подготовка Гримуаров ---
# Призываем наш силовой гримуар PyTorch.
import torch

# Призываем "индикатор прогресса" для наглядности.
from tqdm import tqdm

# --- Акт 2: Пробуждение и Настройка Химеры ---
# Сообщаем о начале ритуала.
print("Пробуждаю 'Сторожевую Химеру' (YOLOv5)...")
# torch.hub.load() - призывает артефакт 'yolov5n' из хранилища 'ultralytics/yolov5'.
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
# model.conf - это "характер" Химеры, ее "порог уверенности".
# Мы снижаем его до 0.10 (10%), чтобы она сообщала даже о неуверенных находках.
model.conf = 0.10
# if torch.cuda.is_available(): - Проверяем, доступен ли Кристалл Маны.
if torch.cuda.is_available():
    # model.to('cuda') - Если да, отправляем Химеру туда для ускорения.
    model = model.to("cuda")
# Сообщаем об успешном пробуждении.
print("Химера пробудилась и готова считать!")

# --- Акт 3: Подготовка к "Охоте и Учету" ---
# Сообщаем о подключении к источнику.
print("\nПодключаюсь к 'Магической Кинопленке'...")
# "Захватываем" наш видеофайл.
cap = cv2.VideoCapture("test_video.mp4")
# Готовим новую папку для результатов этого квеста.
output_folder = "counting_results"
# Создаем папку, если ее еще не существует.
os.makedirs(output_folder, exist_ok=True)
# Готовим счетчик для нумерации сохраненных кадров.
frame_count = 0
# Узнаем общее число кадров для нашего индикатора.
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- НОВАЯ МАГИЯ: Создание "Чаши Памяти" ---
# `last_known_results`: Создаем "магическую чашу", которая будет хранить
# "воспоминание" о последнем успешном отчете Химеры. Вначале она пуста (`None`).
last_known_results = None

# --- Акт 4: Ритуал "Охоты с Памятью" ---
# Сообщаем о начале основного ритуала.
print("Начинаю 'Охоту и Учет'...")

# Запускаем цикл по всем кадрам, обернутый в "индикатор прогресса".
for _ in tqdm(range(total_frames), desc="Подсчет на кинопленке"):
    # Читаем следующий кадр из кинопленки.
    success, frame = cap.read()
    # Если кадры закончились, прерываем ритуал.
    if not success:
        break

    # "Скармливаем" кадр Химере.
    results = model(frame)

    # --- Логика "Памяти" ---
    # `len(results.xyxy[0]) == 0`: "Если отчет Химеры для ЭТОГО кадра пуст..."
    # `... and last_known_results is not None`: "...И (`and`) при этом у нас есть что-то в 'памяти'..."
    if len(results.xyxy[0]) == 0 and last_known_results is not None:
        # `detections_to_draw = last_known_results`: "...то для рисования мы будем использовать 'воспоминание'".
        detections_to_draw = last_known_results
    else:
        # В противном случае (если Химера что-то нашла)...
        # `detections_to_draw = results.xyxy[0]`: "...мы используем свежий 'отчет'".
        detections_to_draw = results.xyxy[0]
        # И если этот свежий отчет не пустой...
        if len(detections_to_draw) > 0:
            # `last_known_results = detections_to_draw`: "...мы ОБНОВЛЯЕМ нашу 'память'".
            last_known_results = detections_to_draw

    # --- Ритуал Подсчета и Рисования ---
    # Готовим счетчик. Важно: мы обнуляем его для КАЖДОГО нового кадра.
    object_counter = 0
    # Наша цель для подсчета, основанная на результатах диагностики.
    target_class = "stop sign"

    # `if detections_to_draw is not None`: Защитная руна на случай, если и свежих, и старых результатов нет.
    if detections_to_draw is not None:
        # Перебираем объекты из 'detections_to_draw' (свежие или из памяти).
        for *box, conf, cls in detections_to_draw:
            # Узнаем имя найденного объекта по его номеру `cls`.
            detected_class = model.names[int(cls)]
            # Проверяем, совпадает ли он с нашей целью.
            if detected_class.strip() == target_class:
                # Если да, увеличиваем счетчик.
                object_counter += 1

            # --- Ритуал "Явных Чернил" (Нанесение отметок на кадр) ---

            # `label = ...`: Создаем текстовую "бирку" для нашего объекта.
            label = f"{detected_class} {conf:.2f}"

            # `x1, y1, x2, y2 = map(int, box)`: "Распаковываем" и округляем координаты рамки.
            x1, y1, x2, y2 = map(int, box)

            # `cv2.rectangle(...)`: Главное заклинание рисования рамки.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # `cv2.putText(...)`: Главное заклинание нанесения текста.
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Наносим на кадр итоговый счетчик.
    summary_text = f"Stop signs found: {object_counter}"
    # Наносим этот текст красным цветом (0,0,255) в углу кадра.
    cv2.putText(
        frame, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    # Сохраняем измененный кадр с рамками и счетчиком.
    save_path = os.path.join(output_folder, f"count_frame_{frame_count}.jpg")
    # `cv2.imwrite` - заклинание "записать образ".
    cv2.imwrite(save_path, frame)
    # Увеличиваем счетчик сохраненных кадров.
    frame_count += 1

# --- Акт 5: Завершение Ритуала ---
# "Освобождаем" кинопленку.
cap.release()
# Сообщаем финальный результат.
print(
    f"\nРитуал завершен. {frame_count} кадра с подсчетом сохранено в папку '{output_folder}'."
)
