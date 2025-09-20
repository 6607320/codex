# === quest_14_1.py ===
# Квест: 14.1 - Ритуал "Квантизации"
# Цель: Освоить Post-Training Static Quantization. Мы "сожмем" нашу обученную
# модель CNN, превратив ее "мысли" из float32 в int8.

# Призываем помощника 'os' для работы с файлами (измерения размера).
import os

# --- Акт 1: Подготовка Гримуаров ---
# Призываем наш силовой гримуар PyTorch.
import torch

# Призываем "строительные блоки" для моделей (Conv2d, Linear...).
import torch.nn as nn

# Призываем гримуар с "функциональными" заклинаниями (relu, max_pool2d...).
import torch.nn.functional as F

# Призываем "инструменты для исправления ошибок" (оптимизаторы).
import torch.optim as optim

# Призываем "Библиотеку" с "учебником" MNIST и гримуар трансформаций.
from torchvision import datasets, transforms

# Призываем наш "индикатор прогресса".
from tqdm import tqdm

# --- Акт 2: Призыв и Обучение "Тяжелого" Голема ---


# --- Чертеж нашей MiniCNN из Квеста 10.2 ---
# `class`: Объявляем "чертеж" для нашего Голема.
class MiniCNN(nn.Module):
    # `def __init__`: Заклинание Инициализации, создает "внутренности" Голема.
    def __init__(self):
        # `super().__init__()`: Пробуждаем магию родительского чертежа `nn.Module`.
        super().__init__()
        # `self.conv1`: Создаем первый "этаж" свертки.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # `self.conv2`: Создаем второй "этаж" свертки.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # `self.fc1`: Создаем "зал раздумий" (линейный слой).
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    # `def forward`: Главное заклинание, описывающее путь "мысли".
    def forward(self, x):
        # `x = F.relu(...)`: Прогоняем мысль через "этаж", "переключатель" и "уменьшитель".
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # `x = F.relu(...)`: Повторяем для второго этажа.
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # `x = x.view(...)`: "Сплющиваем" карты в один длинный вектор.
        x = x.view(-1, 32 * 7 * 7)
        # `x = self.fc1(x)`: Отправляем вектор в "зал раздумий".
        x = self.fc1(x)
        # `return F.log_softmax(...)`: Возвращаем логарифм вероятностей.
        return F.log_softmax(x, dim=1)


# --- Загрузка "учебника" MNIST ---
# `transform = ...`: Создаем конвейер трансформаций.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
# `train_dataset = ...`: Загружаем "учебник".
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
# `train_loader = ...`: Создаем "подносчик" данных.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- Быстрое Обучение ---
# `model_fp32 = MiniCNN()`: Сотворяем "Золотого" Голема (FP32 - float32).
model_fp32 = MiniCNN()
print("Быстро обучаю 'Золотого' Голема (FP32)...")
# `optimizer = ...`: Готовим "Волшебный Ключ" для обучения.
optimizer = optim.Adam(model_fp32.parameters(), lr=0.01)
# `criterion = ...`: Готовим "Рулетку" для измерения ошибок.
criterion = nn.CrossEntropyLoss()
# `model_fp32.train()`: Переводим Голема в режим обучения.
model_fp32.train()
# `for ...`: Начинаем цикл обучения (только на 100 пачках для скорости).
for i, (data, target) in enumerate(tqdm(train_loader, desc="Обучение FP32", total=100)):
    # `if i >= 100: break`: Ставим ограничитель на 100 шагов.
    if i >= 100:
        break
    # `data, target = ...`: Отправляем "учебник" и ответы на верстак.
    data, target = data, target
    # `optimizer.zero_grad()`: Стираем старые ошибки.
    optimizer.zero_grad()
    # `output = model_fp32(data)`: Голем делает предсказание.
    output = model_fp32(data)
    # `loss = criterion(...)`: Измеряем ошибку.
    loss = criterion(output, target)
    # `loss.backward()`: Вычисляем "шепот исправления".
    loss.backward()
    # `optimizer.step()`: "Подкручиваем руны".
    optimizer.step()

# --- Акт 3: Ритуал "Трансмутации" (Квантизация) ---
print("\nНачинаю ритуал 'Трансмутации'...")
# `model_to_quantize = MiniCNN()`: Создаем новый, "чистый" экземпляр Голема.
model_to_quantize = MiniCNN()
# `model_to_quantize.load_state_dict(...)`: Копируем "разум" из обученного Голема в новый.
model_to_quantize.load_state_dict(model_fp32.state_dict())
# `model_to_quantize.eval()`: Переводим Голема в режим "экзамена".
model_to_quantize.eval()
# `model_to_quantize.qconfig = ...`: "Ставим на верстак 'Алхимический Стол'" (применяем конфиг).
model_to_quantize.qconfig = torch.quantization.get_default_qconfig("fbgemm")
# `torch.quantization.prepare(...)`: "Расставляем магических наблюдателей" внутри Голема.
torch.quantization.prepare(model_to_quantize, inplace=True)

# --- Калибровка ---
print("  -> Провожу калибровку на 10 пачках данных...")
# `for ...`: Начинаем цикл "калибровки".
for i, (data, _) in enumerate(train_loader):
    # `if i >= 10: break`: Ограничиваем калибровку 10-ю пачками.
    if i >= 10:
        break
    # `model_to_quantize(data)`: "Прогоняем" данные через Голема, чтобы "наблюдатели" собрали статистику.
    model_to_quantize(data)

# --- Трансмутация ---
# `torch.quantization.convert(...)`: Главное заклинание! Превращаем "золото" в "дерево".
torch.quantization.convert(model_to_quantize, inplace=True)
print("Трансмутация завершена!")

# --- Акт 4: Сравнение Веса Артефактов ---
# `torch.save(...)`: Временно сохраняем "разум" первого Голема.
torch.save(model_fp32.state_dict(), "temp_fp32_model.pth")
# `torch.save(...)`: Временно сохраняем "разум" второго, "сжатого" Голема.
torch.save(model_to_quantize.state_dict(), "temp_int8_model.pth")
# `size_fp32 = ...`: Измеряем размер первого файла.
size_fp32 = os.path.getsize("temp_fp32_model.pth") / 1024
# `size_int8 = ...`: Измеряем размер второго файла.
size_int8 = os.path.getsize("temp_int8_model.pth") / 1024

print("\n--- Сравнение Веса Артефактов ---")
# `print(...)`: Выводим результат.
print(f"  Вес 'Золотого' Голема (FP32): {size_fp32:.2f} КБ")
# `print(...)`: Выводим результат.
print(f"  Вес 'Деревянного' Голема (INT8): {size_int8:.2f} КБ")
# `print(...)`: Выводим результат.
print(f"  Модель стала в {size_fp32 / size_int8:.1f} раз легче!")

# `os.remove(...)`: "Прибираемся", удаляем временный файл.
os.remove("temp_fp32_model.pth")
# `os.remove(...)`: Удаляем второй временный файл.
os.remove("temp_int8_model.pth")
