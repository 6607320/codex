# === quest_7_2.py ===
# Квест: 7.2 - Ритуал "Стилевой Печати"
# Цель: Обучить "магический блокнот" (LoRA) на нашей палитре изображений,
# чтобы запечатать в него суть нашего художественного стиля.
# Это самый сложный ритуал Наставления на данный момент.

# --- Акт 1: Подготовка Гримуаров ---

# Призываем все необходимые инструменты.
import os

import torch
import torch.nn.functional as F  # Гримуар с "мерами ошибок" (loss functions)
from datasets import load_dataset

# Призываем "строительные блоки" Демиурга по отдельности.
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

# Призываем магию "блокнотов".
from peft import LoraConfig, get_peft_model

# Призываем гримуар трансформаций для образов.
from torchvision import transforms

# Наш верный индикатор прогресса.
from tqdm import tqdm

# Призываем Переводчика и Кодировщик Текста.
from transformers import CLIPTextModel, CLIPTokenizer

# --- Акт 2: Настройка Ритуала ---
print("Акт 1: Настройка Ритуала...")
# Имя базового Духа-Демиурга.
model_id = "CompVis/stable-diffusion-v1-4"
# Папка с нашими "учебными картинами".
palette_dir = "generated_palette"
# Папка, куда мы сохраним нашу обученную "Печать".
output_dir = "artist_seal"
os.makedirs(output_dir, exist_ok=True)

# Параметры обучения, адаптированные для нашего Кристалла Маны.
resolution = 256
train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 1e-4
num_train_epochs = 100

# --- Акт 3: Подготовка "Учебника" ---
print("Акт 2: Подготовка Учебника...")
# Загружаем наши картины из локальной папки.
dataset = load_dataset(palette_dir, trust_remote_code=True)

# Создаем конвейер трансформаций для аугментации и подготовки.
preprocess = transforms.Compose(
    [
        transforms.Resize((resolution, resolution)),  # Уменьшаем до 256x256
        transforms.RandomHorizontalFlip(),  # Случайно отражаем
        transforms.ToTensor(),  # Превращаем в тензор
        transforms.Normalize([0.5], [0.5]),  # Нормализуем цвета
    ]
)


# Создаем функцию-помощника, которая будет применять этот конвейер.
def apply_transforms(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}


# "Прикрепляем" нашего помощника к датасету.
dataset.set_transform(apply_transforms)
# Создаем "подносчик", который будет подавать нам данные на урок.
train_dataloader = torch.utils.data.DataLoader(
    dataset["train"], batch_size=train_batch_size, shuffle=True
)

# --- Акт 4: Призыв Компонентов и Создание "Блокнота" ---
print("Акт 3: Призыв компонентов...")
# Призываем всех помощников Демиурга по отдельности.
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=torch.float16
)
# Призываем "Проявителя" (VAE) и "Духа-Скульптора" (UNet).
vae = AutoencoderKL.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float16
)
unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet", torch_dtype=torch.float16
)

# Создаем чертеж "магического блокнота".
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.05,
    bias="none",
)
# "Прикрепляем" блокнот к сердцу Демиурга - UNet.
unet = get_peft_model(unet, lora_config)

# --- Акт 5: Ритуал Наставления ---
print("Акт 4: Начинаю Великий Ритуал Наставления...")

# Перемещаем всех духов на Кристалл Маны.
vae.to("cuda")
text_encoder.to("cuda")
unet.to("cuda")

# Приказываем UNet войти в "режим обучения".
unet.train()

# Готовим "инструмент для исправления ошибок" (оптимизатор).
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
# Призываем "Духа Шума", который будет портить наши картины для урока.
noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Создаем "магическое слово-активатор", с которым будет ассоциироваться наш стиль.
train_prompt = "a beautiful painting in sks style"
# Сразу переводим его в числовые руны, чтобы не делать это в цикле.
prompt_ids = tokenizer(
    train_prompt,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=tokenizer.model_max_length,
).input_ids.to("cuda")

# Начинаем урок, который повторится 100 раз (эпох).
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f"Эпоха {epoch+1}/{num_train_epochs}",
    ):
        with torch.no_grad():
            # Шаг 1: Переносим чистую картину в "латентное пространство" с помощью VAE.
            clean_images = batch["input"].to("cuda", dtype=torch.float16)
            latents = (
                vae.encode(clean_images).latent_dist.sample()
                * vae.config.scaling_factor
            )
            # Шаг 2: Получаем текстовый контекст.
            encoder_hidden_states = text_encoder(prompt_ids)[0]

        # Шаг 3: Добавляем случайный шум к латенту.
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Шаг 4: Просим Демиурга (UNet) предсказать, какой шум мы добавили.
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Шаг 5: Считаем ошибку (насколько предсказанный шум отличается от реального).
        loss = F.mse_loss(noise_pred, noise)
        # Шаг 6: Вычисляем и применяем "исправления" для нашего "блокнота".
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Сообщаем об ошибке в конце каждой эпохи.
    print(f"  Эпоха {epoch+1} завершена. Ошибка (Loss): {loss.item():.4f}")

# --- Акт 6: Сохранение "Стилевой Печати" ---
print("\nРитуал завершен! Сохраняю 'Стилевую Печать'...")
# Сохраняем не весь UNet, а только обученный "блокнот".
unet.save_pretrained(output_dir)
print(f"\nТвоя 'Стилевая Печать' сохранена в папке '{output_dir}'.")
