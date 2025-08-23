# === quest_13_2.py ===
# Квест: 13.2 - Перегонка знаний
# Цель: Реализовать Knowledge Distillation. Мы обучим маленькую "модель-ученика"
# имитировать "мысли" (soft labels) большой "модели-учителя", передавая
# ей знания более эффективно, чем при обучении на "жестких" метках.

# --- Акт 1: Подготовка Гримуаров ---
# Призываем наш силовой гримуар PyTorch.
import torch

# Призываем "строительные блоки" для моделей (Conv2d, Linear, ReLU...).
import torch.nn as nn

# Призываем гримуар с "функциональными" заклинаниями (softmax, max_pool2d...).
import torch.nn.functional as F

# Призываем "инструменты для исправления ошибок" (оптимизаторы).
import torch.optim as optim

# Призываем "Библиотеку" с нашим "учебником" MNIST и гримуар трансформаций.
from torchvision import datasets, transforms

# Призываем наш "индикатор прогресса".
from tqdm import tqdm

# --- Акт 2: Подготовка "Учебника" и Чертежей ---

# Создаем конвейер трансформаций для подготовки изображений.
transform = transforms.Compose(
    [
        # 1. transforms.ToTensor() - превращает картинку из формата PIL/numpy в Тензор PyTorch.
        transforms.ToTensor(),
        # 2. transforms.Normalize(...) - "нормализует" яркость пикселей, приводя их
        #    к стандартному распределению. Это помогает модели учиться быстрее.
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
# Загружаем "учебник" MNIST. download=True скачает его, если нужно.
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
# Создаем "подносчик" данных, который будет подавать нам "учебник" пачками по 64 картинки.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# --- Чертеж "Магистра" (Teacher): Большая и сложная CNN ---
class TeacherModel(nn.Module):
    def __init__(self):  # Ритуал сотворения Магистра.
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Первый этаж: 32 "гнома".
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Второй этаж: 64 "гнома".
        self.fc1 = nn.Linear(9216, 128)  # Первый "зал раздумий".
        self.fc2 = nn.Linear(128, 10)  # Второй, финальный "зал".

    def forward(self, x):  # Путь мысли через разум Магистра.
        x = F.relu(self.conv1(x))  # Проход через 1-й этаж и "переключатель" ReLU.
        x = F.relu(
            F.max_pool2d(self.conv2(x), 2)
        )  # Проход через 2-й этаж, "переключатель" и "уменьшитель".
        x = torch.flatten(x, 1)  # "Сплющиваем" карты признаков в один длинный вектор.
        x = F.relu(self.fc1(x))  # Проход через 1-й "зал".
        x = self.fc2(x)  # Получение финальных "мыслей"-логитов.
        return x


# --- Чертеж "Подмастерья" (Student): Маленькая и простая CNN ---
class StudentModel(nn.Module):
    def __init__(self):  # Ритуал сотворения Подмастерья.
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # Всего один этаж с 16-ю "гномами".
        self.fc1 = nn.Linear(2704, 10)  # И всего один "зал раздумий".

    def forward(self, x):  # Путь мысли через разум Подмастерья.
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


# --- Акт 3: Обучение "Магистра" ---
print("Обучаю 'Магистра', чтобы он стал мудрым...")
teacher = TeacherModel().to("cuda")  # Сотворяем Магистра и отправляем на Кристалл Маны.
optimizer_teacher = optim.Adam(
    teacher.parameters(), lr=0.01
)  # Готовим "Ключ" для его обучения.
criterion = nn.CrossEntropyLoss()  # Готовим "Рулетку" для измерения ошибок.

teacher.train()  # Переводим Магистра в режим обучения.
# Проводим один быстрый "учебный год" (эпоху).
for data, target in tqdm(train_loader, desc="Обучение Магистра (1 эпоха)"):
    data, target = data.to("cuda"), target.to("cuda")
    optimizer_teacher.zero_grad()
    output = teacher(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer_teacher.step()
print("Магистр обучен и готов делиться мудростью.")

# --- Акт 4: Ритуал "Перегонки Знаний" ---
print("\nНачинаю ритуал 'Перегонки Знаний' для 'Подмастерья'...")
student = StudentModel().to("cuda")  # Сотворяем Подмастерья.
optimizer_student = optim.Adam(
    student.parameters(), lr=0.01
)  # Готовим "Ключ" для него.

# --- Настройка Магии "Перегонки" ---
distillation_temp = 5.0  # "Температура", смягчающая "шепот" Магистра.
alpha = 0.5  # "Баланс" между мудростью Магистра и сухими фактами из учебника.

student.train()  # Переводим Ученика в режим обучения.
teacher.eval()  # Переводим Магистра в режим "экзаменатора".

# Начинаем урок для Подмастерья.
for data, target in tqdm(train_loader, desc="Обучение Подмастерья (1 эпоха)"):
    data, target = data.to("cuda"), target.to("cuda")
    optimizer_student.zero_grad()

    student_logits = student(data)  # Получаем "мысли" Подмастерья.
    with torch.no_grad():  # Магистр не учится, поэтому отключаем градиенты.
        teacher_logits = teacher(data)  # Получаем "мудрый шепот" от Магистра.

    # --- Вычисление Составной Ошибки ---
    # Ошибка №1 (Distillation Loss): Насколько "мысли" ученика отличаются от "шепота" учителя.
    distillation_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(
            student_logits / distillation_temp, dim=1
        ),  # Смягчаем и логируем "мысли" ученика.
        F.softmax(
            teacher_logits / distillation_temp, dim=1
        ),  # Смягчаем "шепот" учителя.
    )

    # Ошибка №2 (Student Loss): Насколько ответы ученика соответствуют реальным "правильным ответам".
    student_loss = criterion(student_logits, target)

    # Финальная, взвешенная Ошибка.
    loss = (
        alpha * distillation_loss * (distillation_temp**2)
        + (1.0 - alpha) * student_loss
    )

    # Стандартный ритуал исправления ошибок.
    loss.backward()
    optimizer_student.step()

print(f"\nРитуал завершен! Финальная Ошибка Подмастерья (Loss): {loss.item():.4f}")
print("Подмастерье впитал мудрость Магистра.")
