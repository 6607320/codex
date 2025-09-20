# === quest_11_2_visual.py ===
# Квест: 11.2 - Наглядная Версия "Многогранного Фокуса"
# Цель: Визуализировать, как разные "мудрецы" (головы) в Multi-Head Attention
# находят РАЗНЫЕ смысловые связи в одном и том же предложении.

# Призываем "математический" гримуар для вычисления корня.
import math

# Призываем "Художника" для рисования карт.
import matplotlib.pyplot as plt

# Призываем 'numpy' для удобной работы с массивами для рисования.
import numpy as np

# --- Акт 1: Подготовка Гримуаров ---
# Призываем PyTorch и его "строительные блоки".
import torch
import torch.nn as nn


# --- Акт 2: Чертеж "Совета Мудрецов" ---
# Создаем чертеж нашего артефакта.
class MultiHeadAttention(nn.Module):
    # Заклинание Инициализации: создает все внутренние механизмы.
    def __init__(self, d_model, num_heads):
        super().__init__()  # Пробуждаем магию родительского класса.
        # Проверяем, что общую "глубину смысла" можно поровну разделить между "мудрецами".
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Сохраняем ключевые параметры.
        self.d_model = d_model  # Общая "глубина смысла".
        self.num_heads = num_heads  # Количество "мудрецов".
        self.d_k = d_model // num_heads  # "Глубина смысла" для каждого мудреца.

        # Создаем "Рунные Камни" (Linear слои) для создания Q, K, V.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # Финальный "Рунный Камень" для объединения выводов.
        self.W_o = nn.Linear(d_model, d_model)

    # Внутреннее заклинание "Фокуса Мысли".
    def scaled_dot_product_attention(self, Q, K, V):
        # Вычисляем "симпатию" и масштабируем ее.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Превращаем "симпатию" в "проценты внимания" (вероятности).
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # СОХРАНЯЕМ ВЕСА ДЛЯ ВИЗУАЛИЗАЦИИ: "запоминаем" карту внимания.
        self.last_attn_weights = attn_probs
        # Смешиваем "коктейль" из Значений (V), взвешенных на проценты внимания.
        output = torch.matmul(attn_probs, V)
        return output

    # Внутреннее заклинание "Разделения Смысла".
    def split_heads(self, x):
        # Получаем размеры входящей ауры.
        batch_size, seq_length, d_model = x.size()
        # "Нарезаем" и "переставляем" оси, чтобы у каждого мудреца была своя стопка смыслов.
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    # Внутреннее заклинание "Великого Единения".
    def combine_heads(self, x):
        # Получаем размеры.
        batch_size, _, seq_length, d_k = x.size()
        # "Склеиваем" выводы мудрецов обратно в единую, целостную ауру.
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    # Главное заклинание артефакта.
    def forward(self, Q, K, V):
        # 1. Создаем Q, K, V.
        Q, K, V = self.W_q(Q), self.W_k(K), self.W_v(V)
        # 2. "Нарезаем" их для каждого мудреца.
        Q, K, V = self.split_heads(Q), self.split_heads(K), self.split_heads(V)
        # 3. Каждый мудрец проводит свой ритуал "Фокуса".
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        # 4. "Склеиваем" их выводы.
        output = self.combine_heads(attention_output)
        # 5. Пропускаем через финальный "Камень".
        output = self.W_o(output)
        return output


# --- Акт 3: Испытание на "Осмысленном" Предложении ---

# Задаем параметры: общая "глубина смысла" - 128, количество "мудрецов" - 8.
d_model, num_heads = 128, 8
# Создаем "ауры" для 5 условных слов.
x = torch.randn(1, 5, d_model)

print("--- Испытание 'Многогранного Фокуса' ---")
# Сотворяем наш артефакт по чертежу.
multi_head_attention = MultiHeadAttention(d_model, num_heads)
# Запускаем ритуал.
output = multi_head_attention(x, x, x)

# --- Акт 4: Визуализация Мыслей "Совета Мудрецов" ---
print("\nСоздаю визуализацию мыслей каждого 'мудреца'...")

# Извлекаем "Карты Внимания", которые мы предусмотрительно сохранили.
attention_maps = multi_head_attention.last_attn_weights.squeeze(0).detach().numpy()

# Создаем подписи для осей нашей карты.
labels = ["Старый", "король", "дал", "корону", "принцу"]

# Создаем большую "картину", разделенную на 8 "холстов" (2 строки, 4 столбца).
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

# Запускаем цикл, который пройдет по каждому из 8 "холстов".
# `axes.flat` - "расплющивает" нашу сетку 2x4 в простой список из 8 холстов.
for i, ax in enumerate(axes.flat):
    # Берем карту внимания для i-го мудреца.
    heatmap_data = attention_maps[i]
    # Рисуем ее на текущем холсте 'ax'.
    im = ax.imshow(heatmap_data, cmap="viridis")

    # Настраиваем подписи на осях текущего холста.
    ax.set_xticks(np.arange(len(labels)), labels=labels, fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    # Поворачиваем подписи на оси X для читаемости.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Даем каждому "портрету" свое имя.
    ax.set_title(f"Мысли Мудреца #{i+1}")

# Даем название всей нашей большой картине.
fig.suptitle("Эмпатическое Поле 'Совета Мудрецов'")
# Автоматически подгоняем расположение, чтобы ничего не накладывалось.
fig.tight_layout()
# Сохраняем финальный артефакт.
plt.savefig("multi_head_attention_map.png")

print("Магия визуализирована! Открой файл 'multi_head_attention_map.png'.")

# --- Акт 5: Расшифровка Визуального Результата ---
print("\n--- Что означает эта карта? ---")
# Это финальные инструкции для ученика, объясняющие, как интерпретировать результат.
print(
    "Открой картинку. Ты видишь 8 разных 'Карт Внимания' - по одной от каждого 'мудреца'."
)
print(
    "Они все разные! Это потому, что каждый мудрец 'смотрит' на мир через свою призму."
)
print(
    "- Мудрец #1 мог научиться находить связи 'кто-кому'. Он бы показал яркую связь между 'король' и 'принцу'."
)
print(
    "- Мудрец #2 мог научиться находить глаголы. Он бы 'смотрел' только на слово 'дал'."
)
print(
    "- Мудрец #3 мог научиться связывать прилагательные с существительными. Он бы 'смотрел' от 'Старый' на 'король'."
)
print(
    "Вместе, выводы всех восьми мудрецов дают невероятно богатое и полное понимание предложения."
)
