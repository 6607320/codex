# === quest_16_2.py ===
# Квест: 16.2 - Запись в Хроники
# Цель: Интегрировать MLflow в реальный процесс обучения. Мы обучим нашу
# MiniCNN и будем автоматически записывать все параметры и метрики в "Хронограф".

# Призываем наш главный гримуар "Летописец".
import mlflow

# --- Акт 1: Подготовка Гримуаров ---
# Призываем наш силовой гримуар PyTorch и его "строительные блоки".
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Призываем "Библиотеку" с "учебником" MNIST и гримуар трансформаций.
from torchvision import datasets, transforms

# Наш верный "индикатор прогресса".
from tqdm import tqdm

# --- Акт 2: Подготовка "Учебника" и Чертежа ---
# (Этот акт тебе полностью знаком из Квеста 10.2)

# Создаем конвейер трансформаций для подготовки изображений.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
# Загружаем "учебник" MNIST.
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
# Создаем "подносчик" данных.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# Чертеж нашей "Башни Прозрения".
class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


# --- Акт 3: Настройка и Начало Записи в Хроники ---
# Выбираем "книгу", в которую будем делать запись.
mlflow.set_experiment("Первый Свиток Хроник")

# `with mlflow.start_run(...)`: Начинаем новую "страницу" в нашем журнале.
with mlflow.start_run(run_name="Обучение MiniCNN"):

    # --- Запись "Ингредиентов" (Гиперпараметров) ---
    # Создаем переменные для наших настроек, чтобы было удобно их менять и записывать.
    learning_rate = 0.01
    batch_size = 64
    epochs = 1
    # `mlflow.log_param()`: Записываем каждую нашу "настройку" на "страницу" журнала.
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    # --- Акт 4: Ритуал Наставления под Надзором Летописца ---
    print("Начинаю ритуал наставления 'Башни Прозрения' под запись...")
    # Сотворяем Голема и отправляем на Кристалл Маны.
    model = MiniCNN().to("cuda")
    # Готовим "Волшебный Ключ", используя нашу переменную `learning_rate`.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Готовим "Рулетку" для измерения ошибок.
    criterion = nn.CrossEntropyLoss()

    # Переводим модель в режим обучения.
    model.train()
    # Начинаем цикл по "учебным годам" (эпохам).
    for epoch in range(epochs):
        # Переменная для хранения самой последней ошибки в эпохе.
        final_loss = 0.0
        # `enumerate(train_loader)`: Берем "пачки" данных и их порядковые номера (`step`).
        for step, (data, target) in tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Эпоха {epoch+1}"
        ):
            # Отправляем "пачку" данных на Кристалл Маны.
            data, target = data.to("cuda"), target.to("cuda")
            # Стираем старые ошибки.
            optimizer.zero_grad()
            # Голем делает предсказание.
            output = model(data)
            # Измеряем ошибку.
            loss = criterion(output, target)
            # Вычисляем "шепот исправления".
            loss.backward()
            # "Подкручиваем руны".
            optimizer.step()

            # --- Запись "Результатов" (Метрик) на каждом шагу ---
            # `if step % 100 == 0`: "Если номер шага делится на 100 без остатка..."
            if step % 100 == 0:
                # `mlflow.log_metric()`: "...то записать текущее значение ошибки (`loss.item()`)
                #    в журнал на шаге `step`". Это создаст точки для нашего графика.
                mlflow.log_metric("loss", loss.item(), step=step)
            # Запоминаем ошибку этого шага.
            final_loss = loss.item()

    # После завершения всех шагов, записываем финальную ошибку.
    mlflow.log_metric("final_loss", final_loss)

    # Сообщаем о завершении обучения.
    print(f"\nРитуал завершен! Финальная Ошибка (Loss): {final_loss:.4f}")

    # --- Акт 5: Сохранение Артефакта-Модели в Хроники ---
    print("Сохраняю обученного Голема как артефакт...")
    # `mlflow.pytorch.log_model`: Это специальное, мощное заклинание.
    # Оно берет нашу обученную модель PyTorch (`model`) и сохраняет ее
    # целиком (вместе со всеми необходимыми файлами) в подпапку "model"
    # внутри "страницы" нашего журнала.
    mlflow.pytorch.log_model(model, "model")
    print("Голем сохранен в 'Хрониках'.")

# Сообщаем, что запись в "Хроники" завершена и даем инструкцию.
print("\nЗапись в 'Хроники' завершена. Запусти 'mlflow ui', чтобы увидеть результат.")
