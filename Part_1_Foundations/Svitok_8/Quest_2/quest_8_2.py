# === quest_8_2.py ===
# Квест: 8.2 - Создание классификатора действий
# Цель: Объединить все наши знания! Мы будем использовать Transfer Learning,
# чтобы обучить простую модель отличать одно видео-действие от другого.
# Мы будем вручную загружать видео, прореживать кадры, извлекать из них
# "визуальные эссенции" и обучать на них нашего Голема-Ученика.

# --- Акт 1: Подготовка Гримуаров ---

# Новый помощник 'glob' для "умного" поиска файлов.
import glob

# Помощник 'os' для работы с путями.
import os

# "Набор Кинематографиста".
import cv2

# Призываем все необходимые инструменты.
import torch
import torch.nn as nn
import torch.optim as optim

# Наш верный "индикатор прогресса".
from tqdm import tqdm

# Призываем чертежи для "Всевидящего Духа" и его "Настройщика Зрения".
from transformers import AutoImageProcessor, AutoModel

# --- Акт 2: Призыв "Всевидящего Духа" ---

print("Призываю 'Всевидящего Духа' (MobileNetV2) для извлечения эссенций...")
# Призываем "Настройщика Зрения", который подготовит кадры.
processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
# Призываем "Всевидящего Духа" как 'AutoModel', чтобы получить доступ
# к его "глубинным мыслям" (эмбеддингам), а не к финальному вердикту.
model_extractor = AutoModel.from_pretrained("google/mobilenet_v2_1.0_224").to("cuda")

# --- Акт 3: Извлечение "Эссенций" из Видео (Ручной режим) ---

print("\nСоздаю учебные материалы из видео-палитры...")
# Готовим "ящики" для наших учебных материалов.
embeddings = []
labels = []
# Устанавливаем "правило отбора": будем брать каждый 5-й кадр.
frame_skip = 5

# glob.glob(...) - это заклинание поиска.
# "video_palette/**/*.mp4" - означает "найди мне все файлы, заканчивающиеся на .mp4,
# во всех подпапках внутри 'video_palette'".
video_files = glob.glob("video_palette/**/*.mp4", recursive=True)
# Создаем "карту-справочник", чтобы переводить имена папок в номера классов.
label_map = {"action_A": 0, "action_B": 1}

# Запускаем цикл по каждому найденному видеофайлу.
for video_path in tqdm(video_files, desc="Извлечение видео-эссенций"):

    # Повторяем ритуал из Квеста 8.1: открываем видео и извлекаем
    # прореженные кадры в список 'video_frames'.
    cap = cv2.VideoCapture(video_path)
    video_frames = []
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % frame_skip == 0:
            video_frames.append(frame)
    cap.release()

    # Защитная руна: если видео было слишком коротким и мы не извлекли ни одного кадра, пропускаем его.
    if not video_frames:
        continue

    # os.path.dirname(video_path) - получает путь к папке ('video_palette/action_A').
    # os.path.basename(...) - из этого пути берет только последнее имя ('action_A').
    label_name = os.path.basename(os.path.dirname(video_path))
    # Заглядываем в наш "справочник", чтобы получить номер класса.
    label = label_map[label_name]

    # "Скармливаем" всю пачку кадров "Настройщику", а затем "Всевидящему Духу".
    inputs = processor(images=video_frames, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model_extractor(**inputs)

    # outputs.last_hidden_state - это "эссенции" КАЖДОГО кадра.
    # .mean(dim=[0, 2, 3]) - заклинание "глобального усреднения". Мы "схлопываем"
    # все эссенции в одну-единственную, которая представляет все видео целиком.
    video_essence = outputs.last_hidden_state.mean(dim=[0, 2, 3])

    # Складываем "эссенцию" и "метку" в наши ящики.
    embeddings.append(video_essence)
    labels.append(label)

# "Склеиваем" отдельные эссенции в единый тензор.
embeddings_tensor = torch.stack(embeddings)
# Превращаем список номеров-меток в тензор.
labels_tensor = torch.tensor(labels).to("cuda")

# --- Акт 4: Создание Голема-Ученика ---

# Создаем простейшего Голема.
# in_features=1280 - его "глаза" рассчитаны на эссенцию от MobileNetV2.
# out_features=2 - его "рот" выносит вердикт по двум классам (A или B).
classifier = nn.Linear(in_features=1280, out_features=2).to("cuda")

# --- Акт 5: Ритуал Наставления ---

print("\nНачинаю ритуал наставления Голема-Классификатора Действий...")
# Готовим "Рулетку" и "Волшебный Ключ".
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.01)

# Повторяем урок 50 раз.
for epoch in tqdm(range(50), desc="Обучение на действиях"):
    optimizer.zero_grad()
    outputs = classifier(embeddings_tensor)
    loss = criterion(outputs, labels_tensor)
    loss.backward()
    optimizer.step()

# --- Акт 6: Завершение ---

print(f"\nФинальная Ошибка (Loss): {loss.item():.4f}")
print("Ритуал завершен! Голем научился отличать 'движение вправо' от 'увеличения'.")
