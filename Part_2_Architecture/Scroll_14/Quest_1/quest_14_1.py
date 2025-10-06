# === quest_14_1.py ===
# Имя этого пергамента, хранящего ритуал "Трансмутации".
# Квест: 14.1 - Ритуал "Квантизации"
# Каноническое имя Квеста, как оно записано в Великом Кодексе.
# Цель: Освоить Post-Training Static Quantization. Мы "сожмем" нашу обученную
# Священная цель нашего ритуала.
# модель CNN, превратив ее "мысли" из float32 в int8.
# Детальное описание цели.

# Мы призываем "Духа-Архивариуса" `os` для работы с файлами (измерения размера).
import os

# --- Акт 1: Подготовка Гримуаров ---
# Начинается первый акт: мы призываем все необходимые знания и инструменты.
# Мы призываем наш главный силовой гримуар `PyTorch`.
import torch

# Мы призываем `torch.nn` (с псевдонимом `nn`) — главу с чертежами базовых
# блоков для моделей.
import torch.nn as nn

# Мы призываем `torch.nn.functional` (с псевдонимом `F`) — гримуар с
# "функциональными" заклинаниями.
import torch.nn.functional as F

# Мы призываем `torch.optim` (с псевдонимом `optim`) — гримуар с
# "инструментами для исправления ошибок".
import torch.optim as optim

# Мы призываем "Библиотеку" с "учебником" MNIST и гримуар трансформаций.
from torchvision import datasets, transforms

# Мы призываем наш верный "индикатор прогресса".
from tqdm import tqdm

# --- Акт 2: Призыв и Обучение "Тяжелого" Голема ---
# Начинается второй акт: мы создаем и обучаем нашего "Золотого" Голема.


# --- Чертеж нашей MiniCNN из Квеста 10.2 ---
# `class`: Мы объявляем "чертеж" для нашего Голема.
class MiniCNN(nn.Module):
    # `def __init__`: Мы определяем Заклинание Инициализации, которое создает "внутренности" Голема.
    def __init__(self):
        # `super().__init__()`: Мы пробуждаем магию родительского чертежа `nn.Module`.
        super().__init__()
        # `self.conv1`: Мы создаем первый "этаж" свертки.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # `self.conv2`: Мы создаем второй "этаж" свертки.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # `self.fc1`: Мы создаем "зал раздумий" (линейный слой).
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    # `def forward`: Мы определяем Главное заклинание, описывающее путь "мысли".
    def forward(self, x):
        # `x = F.relu(...)`: Мы прогоняем мысль через "этаж", "переключатель" и "уменьшитель".
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # `x = F.relu(...)`: Мы повторяем для второго этажа.
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # `x = x.view(...)`: Мы "сплющиваем" карты в один длинный вектор.
        x = x.view(-1, 32 * 7 * 7)
        # `x = self.fc1(x)`: Мы отправляем вектор в "зал раздумий".
        x = self.fc1(x)
        # `return F.log_softmax(...)`: Мы возвращаем логарифм вероятностей.
        return F.log_softmax(x, dim=1)


# --- Загрузка "учебника" MNIST ---
# `transform = ...`: Мы создаем конвейер трансформаций.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
# `train_dataset = ...`: Мы загружаем "учебник".
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
# `train_loader = ...`: Мы создаем "подносчик" данных.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- Быстрое Обучение ---
# `model_fp32 = MiniCNN()`: Мы сотворяем "Золотого" Голема (FP32 — float32).
model_fp32 = MiniCNN()
# Мы оглашаем на кристалл о начале обучения.
print("Быстро обучаю 'Золотого' Голема (FP32)...")
# `optimizer = ...`: Мы готовим "Волшебный Ключ" для обучения.
optimizer = optim.Adam(model_fp32.parameters(), lr=0.01)
# `criterion = ...`: Мы готовим "Рулетку" для измерения ошибок.
criterion = nn.CrossEntropyLoss()
# `model_fp32.train()`: Мы переводим Голема в режим обучения.
model_fp32.train()
# `for ...`: Мы начинаем цикл обучения (только на 100 пачках для скорости).
for i, (data, target) in enumerate(tqdm(train_loader, desc="Обучение FP32", total=100)):
    # `if i >= 100: break`: Мы ставим ограничитель на 100 шагов.
    if i >= 100:
        # Мы прерываем цикл.
        break
    # `data, target = ...`: Мы отправляем "учебник" и ответы на верстак.
    data, target = data, target
    # `optimizer.zero_grad()`: Мы стираем старые ошибки.
    optimizer.zero_grad()
    # `output = model_fp32(data)`: Наш Голем делает предсказание.
    output = model_fp32(data)
    # `loss = criterion(...)`: Мы измеряем ошибку.
    loss = criterion(output, target)
    # `loss.backward()`: Мы вычисляем "шепот исправления".
    loss.backward()
    # `optimizer.step()`: Мы "подкручиваем руны".
    optimizer.step()

# --- Акт 3: Ритуал "Трансмутации" (Квантизация) ---
# Начинается третий, самый магический акт: мы "сжимаем" нашего Голема.
# Мы оглашаем на кристалл о начале ритуала.
print("\nНачинаю ритуал 'Трансмутации'...")
# `model_to_quantize = MiniCNN()`: Мы создаем новый, "чистый" экземпляр Голема для трансмутации.
model_to_quantize = MiniCNN()
# `model_to_quantize.load_state_dict(...)`: Мы копируем "разум" из обученного Голема в новый.
model_to_quantize.load_state_dict(model_fp32.state_dict())
# `model_to_quantize.eval()`: Мы переводим Голема в режим "экзамена".
model_to_quantize.eval()
# `model_to_quantize.qconfig = ...`: Мы "ставим на верстак 'Алхимический Стол'" (применяем конфиг).
model_to_quantize.qconfig = torch.quantization.get_default_qconfig("fbgemm")
# `torch.quantization.prepare(...)`: Мы "расставляем магических наблюдателей" внутри Голема.
torch.quantization.prepare(model_to_quantize, inplace=True)

# --- Калибровка ---
# Мы оглашаем о начале калибровки.
print("  -> Провожу калибровку на 10 пачках данных...")
# `for ...`: Мы начинаем цикл "калибровки".
for i, (data, _) in enumerate(train_loader):
    # `if i >= 10: break`: Мы ограничиваем калибровку 10-ю пачками.
    if i >= 10:
        # Мы прерываем цикл.
        break
    # `model_to_quantize(data)`: Мы "прогоняем" данные через Голема, чтобы "наблюдатели" собрали статистику.
    model_to_quantize(data)

# --- Трансмутация ---
# `torch.quantization.convert(...)`: Мы произносим Главное заклинание! Превращаем "золото" в "дерево".
torch.quantization.convert(model_to_quantize, inplace=True)
# Мы оглашаем, что трансмутация завершена.
print("Трансмутация завершена!")

# --- Акт 4: Сравнение Веса Артефактов ---
# Начинается финальный акт: мы сравниваем вес наших Големов.
# `torch.save(...)`: Мы временно сохраняем "разум" первого Голема.
torch.save(model_fp32.state_dict(), "temp_fp32_model.pth")
# `torch.save(...)`: Мы временно сохраняем "разум" второго, "сжатого" Голема.
torch.save(model_to_quantize.state_dict(), "temp_int8_model.pth")
# `size_fp32 = ...`: Мы измеряем размер первого файла.
size_fp32 = os.path.getsize("temp_fp32_model.pth") / 1024
# `size_int8 = ...`: Мы измеряем размер второго файла.
size_int8 = os.path.getsize("temp_int8_model.pth") / 1024

# Мы оглашаем заголовок нашего вердикта.
print("\n--- Сравнение Веса Артефактов ---")
# `print(...)`: Мы выводим результат.
print(f"  Вес 'Золотого' Голема (FP32): {size_fp32:.2f} КБ")
# `print(...)`: Мы выводим результат.
print(f"  Вес 'Деревянного' Голема (INT8): {size_int8:.2f} КБ")
# `print(...)`: Мы выводим результат.
print(f"  Модель стала в {size_fp32 / size_int8:.1f} раз легче!")

# `os.remove(...)`: Мы "прибираемся", удаляя временный файл.
os.remove("temp_fp32_model.pth")
# `os.remove(...)`: Мы удаляем второй временный файл.
os.remove("temp_int8_model.pth")
