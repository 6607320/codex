# === quest_5_3.py ===
# Квест: 5.3 - Проверка новых знаний
# Цель: Проверить, действительно ли наш Голем чему-то научился.
# Мы загрузим "чистого" Голема, "прикрепим" к нему наш обученный
# "магический блокнот" (LoRA) и зададим ему вопрос, чтобы оценить результат.

# --- Акт 1: Подготовка Гримуаров ---

# Призываем PyTorch.
import torch
# Призываем чертежи для Голема, Переводчика и инструкции по сжатию.
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# Призываем специальный чертеж PeftModel для работы с "блокнотами".
from peft import PeftModel

# --- Акт 2: Призыв и "Просветление" Голема ---

# Сообщаем о начале ритуала.
print("Призываю 'чистого' Голема в сжатой 4-битной форме...")
# Имя нашего базового Голема.
model_name = "distilgpt2"

# Создаем ту же самую инструкцию по сжатию, что и при обучении.
# Это ВАЖНО: Голем и его "блокнот" должны быть в одной "системе магии".
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Призываем "чистого" Голема, применяя к нему сжатие на лету.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Призываем соответствующего Переводчика.
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Устанавливаем 'pad_token', как мы делали при обучении.
tokenizer.pad_token = tokenizer.eos_token

# Указываем путь к нашему артефакту - папке с "магическим блокнотом".
adapter_path = "results/checkpoint-250"
print(f"\nНахожу магический блокнот в '{adapter_path}'...")

# КЛЮЧЕВОЕ ЗАКЛИНАНИЕ: PeftModel.from_pretrained
# Оно берет "чистого" Голема (model) и "надевает" на него "блокнот" (adapter_path).
# Внутри оно читает adapter_config.json, находит нужные "отделы мозга"
# и прикрепляет к ним обученные "страницы" из adapter_model.safetensors.
model = PeftModel.from_pretrained(model, adapter_path)
print("Голем прочел блокнот и получил новые знания!")

# --- Акт 3: Экзамен ---

# Создаем "экзаменационный билет" (промпт).
# Мы используем тот же формат "Instruction: ... Response:", которому учили Голема.
prompt = "Instruction: Which genre is the hobbit?\n\nResponse:"
print(f"\nЗадаю Голему вопрос: {prompt}")

# Переводим наш вопрос в числовые руны-тензоры и отправляем на Кристалл Маны.
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Запускаем генерацию ответа.
# model.generate - это главное заклинание для получения ответа от Голема-Сказителя.
# 1. input_ids - передаем сами руны.
# 2. attention_mask - передаем "карту внимания".
# 3. max_new_tokens=20 - ограничиваем длину ответа 20-ю новыми токенами.
outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=20)

# --- Акт 4: Оценка Ответа ---

# Голем вернул нам ответ в виде числовых рун.
# tokenizer.decode - это заклинание для обратного перевода чисел в текст.
# outputs[0] - берем первый (и единственный) вариант ответа.
# skip_special_tokens=True - просим не печатать служебные руны вроде "<|endoftext|>".
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Печатаем финальный, расшифрованный ответ нашего "просветленного" ученика.
print("\nОтвет 'Просветленного' Голема:")
print(response)