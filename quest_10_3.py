# === quest_10_3.py ===
# Квест: 10.3 - Сборка Рунного Камня
# Цель: Собрать свою первую Сверточную Нейронную Сеть (CNN) с нуля,
# используя готовые "строительные блоки" из PyTorch (nn.Conv2d, nn.Linear).
# Мы обучим ее на классической задаче распознавания рукописных цифр MNIST.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# --- Акт 1: Подготовка "Учебника" (Данные MNIST) ---
print("Готовлю 'учебник' с рукописными цифрами (MNIST)...")
# Создаем конвейер трансформаций для изображений
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Превращаем картинку в тензор
        transforms.Normalize((0.1307,), (0.3081,)),  # Нормализуем яркость
    ]
)
# Скачиваем и готовим "учебник"
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# --- Акт 2: Чертеж Нашей "Башни Прозрения" (Модель CNN) ---
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Первый этаж: 1 входной канал (ч/б), 16 "гномов с фонарями", трафарет 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # Второй этаж: 16 входных карт, 32 "гнома", трафарет 3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # "Магистр-Оценщик": принимает "сплющенный" атлас и выдает вердикт по 10 классам (цифры 0-9)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # Прогоняем образ через 1-й этаж, "магический переключатель" ReLU и "уменьшитель" MaxPool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Прогоняем через 2-й этаж
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # "Сплющиваем" финальный атлас в длинный вектор
        x = x.view(-1, 32 * 7 * 7)
        # Отдаем вектор "Магистру-Оценщику"
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)  # Возвращаем логарифм вероятностей


# --- Акт 3: Ритуал Наставления ---
print("Сотворяю 'Башню Прозрения' и начинаю ритуал наставления...")
model = MiniCNN().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Проводим один полный "учебный год" (эпоху)
model.train()  # Переводим модель в режим обучения
for batch_idx, (data, target) in tqdm(
    enumerate(train_loader), total=len(train_loader), desc="Обучение 'Башни'"
):
    data, target = data.to("cuda"), target.to("cuda")
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

print(f"\nРитуал завершен! Финальная Ошибка (Loss): {loss.item():.4f}")

# --- Акт 4: Экзамен (Проверка на одной цифре) ---
print("\nПровожу экзамен: показываю 'Башне' одну случайную цифру...")
# Берем один тестовый образец
test_dataset = datasets.MNIST("./data", train=False, transform=transform)
sample_data, sample_label = test_dataset[0]

model.eval()  # Переводим модель в режим "экзамена"
with torch.no_grad():
    output = model(sample_data.unsqueeze(0).to("cuda"))

prediction = output.argmax(dim=1, keepdim=True).item()
print(f"  -> 'Башня' увидела цифру: {prediction}")
print(f"  -> Правильный ответ был: {sample_label}")
if prediction == sample_label:
    print("  Вердикт: Успех! 'Башня Прозрения' видит!")
else:
    print("  Вердикт: Ошибка! 'Башня' еще требует тренировок.")
