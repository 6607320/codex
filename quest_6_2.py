# === quest_6_2.py ===
# Квест: 6.2 - Обучение классификатора
# Цель: Освоить Transfer Learning. Мы будем использовать "эссенции" (эмбеддинги),
# извлеченные мощной моделью, чтобы обучить НАШУ СОБСТВЕННУЮ, очень простую
# модель для решения конкретной задачи - классификации аудио.

# --- Акт 1: Подготовка Гримуаров ---

# Призываем все необходимые инструменты.
import torch
import torch.nn as nn  # Гримуар со "строительными блоками" для моделей.
import torch.optim as optim  # Гримуар с "методами исправления ошибок".
import torchaudio.transforms as T  # Гримуар для аудио-трансформаций.
from datasets import load_dataset
from tqdm import tqdm  # Наш "индикатор прогресса".
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# --- Акт 2: Призыв Инструментов и Данных ---

print("Призываю Духа-Эмпата и Настройщика Слуха...")
# Призываем пару для извлечения "эссенций".
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model_extractor.to("cuda")  # Отправляем на Кристалл Маны.

print("Загружаю данные из локальной сокровищницы...")
# Загружаем датасет 'PolyAI/minds14' из локального кэша (скачан в download_data.py).
# `trust_remote_code=True` все еще нужен, чтобы наша версия datasets согласилась
# запустить его "свиток-инструкцию".
full_dataset = load_dataset(
    "PolyAI/minds14", name="en-US", split="train", trust_remote_code=True
)

# --- Акт 3: Создание "Учебника" из Эссенций ---

print("\nСоздаю учебные материалы: 50 досье и меток...")
# Готовим два "ящика" для наших учебных материалов.
embeddings = []  # Для "эссенций"-досье.
labels = []  # Для "меток"-ответов.

# full_dataset.select(range(50)) - заклинание "выбери первые 50 записей".
# Мы итерируемся по этой маленькой выборке.
for sample in tqdm(full_dataset.select(range(50)), desc="Извлечение эссенций"):

    # Извлекаем сырые аудиоданные и их частоту.
    audio_data = sample["audio"]["array"]
    original_sampling_rate = sample["audio"]["sampling_rate"]

    # Ритуал "Передискретизации": если частота не 16000, мы ее исправляем.
    if original_sampling_rate != 16000:
        resampler = T.Resample(orig_freq=original_sampling_rate, new_freq=16000)
        audio_data = resampler(torch.tensor(audio_data, dtype=torch.float32)).numpy()

    # Повторяем ритуал извлечения "эссенции" из Квеста 6.1.
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt").to(
        "cuda"
    )
    with torch.no_grad():
        outputs = model_extractor(**inputs)
    essence = outputs.last_hidden_state.mean(dim=1)

    # Кладем полученную "эссенцию" в один ящик...
    embeddings.append(essence)
    # ...а ее "метку" (номер класса, например, "узнать баланс") - в другой.
    labels.append(sample["intent_class"])

# "Склеиваем" отдельные "эссенции" в один большой тензор для обучения.
embeddings_tensor = torch.cat(embeddings)
# Превращаем список номеров-меток в тензор.
labels_tensor = torch.tensor(labels).to("cuda")

# --- Акт 4: Создание Нашего Голема-Ученика ---

# nn.Linear - это чертеж простейшего "нейронного слоя".
# in_features=768 - его "глаза" рассчитаны на 768 чисел (размер нашей "эссенции").
# out_features=14 - его "рот" выдает 14 чисел (по числу классов в датасете minds14).
classifier = nn.Linear(in_features=768, out_features=14).to("cuda")

# --- Акт 5: Ритуал Наставления (Ручной режим) ---
print("\nНачинаю ритуал наставления...")
# "Рулетка" для измерения ошибки классификации.
criterion = nn.CrossEntropyLoss()
# "Волшебный Ключ" для исправления ошибок. Мы передаем ему "настройки"
# нашего ученика (`classifier.parameters()`) для "подкрутки".
optimizer = optim.Adam(classifier.parameters(), lr=0.01)

# Повторяем урок 100 раз.
for epoch in tqdm(range(100), desc="Обучение классификатора"):
    optimizer.zero_grad()  # 1. Стираем старые ошибки.
    outputs = classifier(embeddings_tensor)  # 2. Ученик делает предсказание.
    loss = criterion(outputs, labels_tensor)  # 3. Измеряем ошибку.
    loss.backward()  # 4. Вычисляем, как исправиться.
    optimizer.step()  # 5. Исправляемся.

    # Каждые 10 эпох печатаем отчет о том, как уменьшается ошибка.
    if (epoch + 1) % 10 == 0:
        print(f"  Эпоха {epoch + 1}/100, Ошибка (Loss): {loss.item():.4f}")

print("Ритуал завершен! Наш Голем-Классификатор обучен.")

# --- Акт 6: Ритуал Сохранения Знаний ---

save_path = "voice_classifier_knowledge.pth"
# torch.save - главное заклинание для сохранения.
# classifier.state_dict() - это "слепок разума" нашего ученика, его обученные "настройки"-веса.
torch.save(classifier.state_dict(), save_path)
print(f"\nЗнания Голема-Классификатора запечатаны в артефакт: '{save_path}'")
