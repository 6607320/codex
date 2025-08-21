# === quest_12_2.py ===
# Квест: 12.2 - Создание "Туманного Образа"
# Цель: Реализовать с нуля Denoising Diffusion Probabilistic Model (DDPM).
# Это фундаментальный принцип, лежащий в основе всех современных
# генераторов изображений, таких как Stable Diffusion.

# --- Легенда Квеста: Лепка из Первозданного Хаоса ---
# 1. Прямой процесс: Мы учим "подмастерье" (UNet), показывая ему, как
#    идеальная статуя (картинка) постепенно покрывается "пылью" (шумом).
# 2. Обратный процесс: Мы даем обученному "подмастерье" бесформенный
#    кусок хаоса (случайный шум) и просим его шаг за шагом "счищать пыль",
#    пока не проявится новая, уникальная статуя.

# --- Акт 1: Подготовка Гримуаров ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os

# --- Акт 2: Настройка Ритуала ---
# Задаем размер наших будущих творений. 32x32 - хороший компромисс для скорости.
image_size = 32
# Количество "учебных страниц" в одной пачке.
batch_size = 128
# Количество "учебных лет". Диффузия требует больше времени на обучение.
epochs = 20

# --- Акт 3: Подготовка "Учебника" ---
print("Готовлю 'учебник' с рукописными цифрами (MNIST)...")
# Создаем конвейер трансформаций.
transform = transforms.Compose([
    # Увеличиваем размер цифр MNIST (28x28) до нашего рабочего размера (32x32).
    transforms.Resize((image_size, image_size)),
    # Превращаем картинку в Тензор PyTorch.
    transforms.ToTensor(),
    # Нормализуем пиксели к диапазону [-1, 1]. Это стандарт для диффузионных моделей.
    transforms.Normalize([0.5], [0.5])
])
# Загружаем наш учебник.
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# Создаем "подносчик" для данных.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- Акт 4: Чертеж "Скульптора Хаоса" (простой UNet) ---
# UNet - это архитектура, похожая на "песочные часы". Она сначала сжимает
# образ, чтобы понять "суть", а потом разжимает его, восстанавливая детали.
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # --- Путь Вниз (Сжатие) ---
        # Первый этаж: 1 цветовой канал -> 32 "карты признаков".
        self.down1 = nn.Conv2d(1, 32, 3, padding=1)
        # Второй этаж: 32 карты -> 64 карты.
        self.down2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # --- Путь Наверх (Восстановление) ---
        # Первый "восстанавливающий" этаж: 64 карты -> 32 карты.
        self.up1 = nn.Conv2d(64, 32, 3, padding=1)
        # Второй "восстанавливающий" этаж: 32 карты -> 1 финальная карта (предсказанный шум).
        self.up2 = nn.Conv2d(32, 1, 3, padding=1)
        
        # Инструмент "Уменьшения" (сжимает карту в 2 раза).
        self.pool = nn.MaxPool2d(2)
        # Инструмент "Увеличения" (растягивает карту в 2 раза).
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    # Главное заклинание, описывающее путь образа через "часы".
    def forward(self, x):
        # Путь вниз...
        x1 = F.relu(self.down1(x)) # Этаж 1
        x2 = F.relu(self.down2(self.pool(x1))) # Этаж 2 (сначала уменьшаем, потом анализируем)
        # Путь наверх...
        x3 = F.relu(self.up1(self.upsample(x2))) # Этаж 3 (сначала увеличиваем, потом анализируем)
        # "Магический мостик" (Skip connection): мы добавляем информацию с первого этажа (x1)
        # к третьему (x3). Это помогает не потерять мелкие детали при сжатии.
        x3 = x3 + x1 
        # Финальный этаж.
        output = self.up2(x3)
        return output

# --- Акт 5: Настройка Магии Шума ---
# Настраиваем "прямой процесс" - как именно мы будем добавлять "пыль".
timesteps = 300 # У нас будет 300 шагов от чистой статуи до полного хаоса.
beta_start, beta_end = 0.0001, 0.02 # Сила "запыления" на первом и последнем шаге.
betas = torch.linspace(beta_start, beta_end, timesteps) # Создаем плавный переход силы шума.
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0) # Накопленное произведение альф, ключевой параметр.

# --- НОВОЕ ЗАКЛИНАНИЕ ТЕЛЕПОРТАЦИИ ---
# Отправляем все наши "расписания" на Кристалл Маны ОДИН РАЗ в самом начале.
betas = betas.to("cuda")
alphas = alphas.to("cuda")
alphas_cumprod = alphas_cumprod.to("cuda")

# Заклинание, которое может "испортить" любую картинку до любого шага 't' за один раз.
def get_noisy_image(x_start, t):
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - alphas_cumprod[t])[:, None, None, None]
    noise = torch.randn_like(x_start) # Создаем случайный шум.
    # Формула "зашумления": (корень_альфы * чистый_образ) + (корень_не_альфы * шум)
    noisy_image = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
    return noisy_image, noise

# --- Акт 6: Ритуал Наставления "Скульптора" ---
model = SimpleUNet().to("cuda") # Сотворяем и отправляем на Кристалл.
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Готовим "Волшебный Ключ".
criterion = nn.MSELoss() # "Рулетка", измеряющая разницу между двумя образами.

print("\nНачинаю ритуал наставления 'Скульптора Хаоса'...")
for epoch in range(epochs):
    # Цикл по "учебнику".
    for step, (images, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Эпоха {epoch+1}"):
        optimizer.zero_grad() # Стираем старые ошибки.
        
        # Выбираем случайный "шаг зашумления" 't' для каждой картинки в пачке.
        t = torch.randint(0, timesteps, (images.size(0),)).long()
        images = images.to("cuda")
        
        # "Портим" чистые образы до состояния 't' и запоминаем, какой шум мы добавили.
        noisy_images, noise = get_noisy_image(images, t)
        noisy_images = noisy_images.to("cuda")
        noise = noise.to("cuda")
        
        # Просим "Скульптора" предсказать, какой шум был добавлен.
        predicted_noise = model(noisy_images)
        # Измеряем ошибку: насколько предсказанный шум отличается от реального.
        loss = criterion(predicted_noise, noise)
        
        # Стандартный ритуал исправления ошибок.
        loss.backward()
        optimizer.step()
    
    print(f"  Эпоха {epoch+1} завершена. Ошибка (Loss): {loss.item():.4f}")

# --- Акт 7: Магия Творения (Обратная Диффузия) ---
print("\nРитуал завершен! Прошу 'Скульптора' сотворить образ из хаоса...")
os.makedirs("chaos_sculptures", exist_ok=True)

# Начинаем с куска "первозданного хаоса" - тензора со случайным шумом.
generated_images = torch.randn(64, 1, image_size, image_size).to("cuda")

# Запускаем цикл "очищения" в ОБРАТНОМ порядке - от самого шумного шага к чистому.
for t in tqdm(range(timesteps - 1, -1, -1), desc="Творение из хаоса"):
    with torch.no_grad(): # Отключаем отслеживание ошибок.
        # "Скульптор" смотрит на текущий "кусок мрамора" и предсказывает, какая "пыль" на нем лишняя.
        predicted_noise = model(generated_images)
        
        # --- Математическая магия "очищения" ---
        # Это обратная формула к той, что была в 'get_noisy_image'.
        # Мы берем текущий образ и "вычитаем" из него предсказанный шум,
        # но с определенными коэффициентами, чтобы не "очистить" слишком сильно.
        alpha_t = alphas[t]
        alpha_t_cumprod = alphas_cumprod[t]
        generated_images = (1 / torch.sqrt(alpha_t)) * \
            (generated_images - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise)
        
        # Добавляем немного случайного шума (кроме последнего шага).
        # Это делает процесс более стабильным.
        if t > 0:
            noise = torch.randn_like(generated_images)
            generated_images += torch.sqrt(betas[t]) * noise

# "Приводим в порядок" финальные образы, чтобы их можно было сохранить.
generated_images = (generated_images.clamp(-1, 1) + 1) / 2
# Сохраняем сетку из 64 сотворенных шедевров.
save_image(generated_images, 'chaos_sculptures/new_creation.png')
print("Образы сотворены! Открой 'chaos_sculptures/new_creation.png'.")