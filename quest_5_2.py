# === quest_5_2.py ===
# Квест: 5.2 - Ритуал "Эффективной Адаптации"
# Цель: Провести первое дообучение (fine-tuning) языковой модели, используя
# продвинутую магию QLoRA, чтобы ритуал был возможен на нашем Кристалле Маны.

# --- Акт 1: Подготовка Всех Гримуаров ---

# Призываем все необходимые инструменты для этого сложного ритуала.
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Акт 2: Призыв Ученика и Подготовка Учебников ---

print("Подготовка к ритуалу Наставления...")
# Имя нашего Голема-ученика.
model_name = "distilgpt2"
# Призываем Переводчика, который говорит на его диалекте.
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Техническая руна: говорим Переводчику использовать символ "конца текста"
# в качестве "заполнителя" для выравнивания предложений.
tokenizer.pad_token = tokenizer.eos_token

# Создаем "свиток-инструкцию" по "магическому сжатию" (квантизации).
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # Приказываем сжимать до 4-бит.
    bnb_4bit_quant_type="nf4", # Используем эффективный тип сжатия "Normal Float 4".
    bnb_4bit_compute_dtype=torch.float16, # Во время вычислений используем 16-битную точность.
)

# Призываем Голема-Сказителя (AutoModelForCausalLM).
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config, # Передаем ему инструкцию по сжатию.
    device_map="auto" # Доверяем "Ускорителю" распределить Голема по ресурсам.
)

# Создаем "Магический Преобразователь" - функцию, которая будет "на лету"
# готовить наши учебные материалы.
def process_dataset(batch):
    # Собираем инструкцию и ответ в одну строку, чтобы имитировать диалог.
    texts = [f"Instruction: {instr}\n\nResponse: {resp}" for instr, resp in zip(batch['instruction'], batch['response'])]
    # Переводим этот текст в числовые руны-тензоры.
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

# Открываем "портал" к архиву 'dolly-15k'.
streaming_dataset = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True, trust_remote_code=True)
# Создаем "умный портал":
# 1. .take(1000) - Берем только первые 1000 записей.
# 2. .map(...) - "Прикрепляем" к порталу наш "Магический Преобразователь".
processed_dataset = streaming_dataset.take(1000).map(process_dataset, batched=True)

# --- Акт 3: Создание "Магического Блокнота" (LoRA) ---

# Произносим над сжатым Големом подготовительное заклинание.
model = prepare_model_for_kbit_training(model)

# Создаем "чертеж" для нашего блокнота с указанием всех параметров.
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
# "Прикрепляем" блокнот к Голему.
model = get_peft_model(model, lora_config)

# --- Акт 4: Ритуал Наставления ---

# Создаем "инструкцию для Наставника".
training_args = TrainingArguments(
    output_dir="./results",          # Куда складывать артефакты.
    per_device_train_batch_size=1,   # Учим по одной карточке за раз.
    gradient_accumulation_steps=4,   # Накапливаем опыт с 4-х карточек.
    learning_rate=2e-4,              # Скорость обучения.
    max_steps=250,                   # Главная руна: говорим, что урок длится ровно 250 шагов.
    logging_steps=20,                # Каждые 20 шагов докладывать о прогрессе.
)

# Призываем самого Наставника (Trainer).
trainer = Trainer(
    model=model,                       # Кого учить.
    train_dataset=processed_dataset,   # По какому "умному порталу" учить.
    args=training_args,                # По какой инструкции учить.
    # Используем профессионального помощника для упаковки данных.
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Сообщаем о начале Великого Ритуала.
print("\nНачинаю Великий Ритуал Наставления...")
# Запускаем урок.
trainer.train()

# Сообщаем о завершении.
print("\nРитуал завершен! Голем получил новые знания.")