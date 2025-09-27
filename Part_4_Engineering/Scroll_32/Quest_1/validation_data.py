# --- Свиток "Камень Истины" (validation_data.py) ---

# Мы создаем нерушимый эталон, "Набор Испытаний", который будет служить мерилом мудрости наших духов.
VALIDATION_SET = [
    # Первое испытание: текст, наполненный радостью, и истинный ответ, выгравированный рядом - "ПОЗИТИВ".
    {"text": "I love this product, it is absolutely amazing!", "label": "POSITIVE"},
    # Второе испытание: слова, пропитанные горечью, и эталонный ответ - "НЕГАТИВ".
    {"text": "This is the worst service I have ever received in my life.", "label": "NEGATIVE"},
    # Третье испытание: хвалебная песнь шедевру и ключ к загадке - "ПОЗИТИВ".
    {"text": "The movie was fantastic, a true masterpiece.", "label": "POSITIVE"},
    # Четвертое испытание: жалоба на разочарование и истинный вердикт - "НЕГАТИВ".
    {"text": "I am so disappointed with the quality, it broke after one day.", "label": "NEGATIVE"},
    # Пятое испытание: описание чудесного опыта и его истинная суть - "ПОЗИТИВ".
    {"text": "What a wonderful experience, I would recommend it to everyone.", "label": "POSITIVE"},
    # Шестое испытание: слова сожаления и их эталонная эмоция - "НЕГАТИВ".
    {"text": "A complete waste of time and money, I regret buying this.", "label": "NEGATIVE"},
    # Седьмое испытание: описание дружелюбной помощи и ее светлая сторона - "ПОЗИТИВ".
    {"text": "The team was very helpful and friendly.", "label": "POSITIVE"},
    # Восьмое испытание: рассказ об ужасной еде и грубости, и его темная суть - "НЕГАТИВ".
    {"text": "The food was terrible and the waiter was rude.", "label": "NEGATIVE"},
    # Девятое испытание: восхищение выдающимся выступлением и его позитивный заряд - "ПОЗИТИВ".
    {"text": "An outstanding performance by the entire cast.", "label": "POSITIVE"},
    # Десятое, финальное испытание: клятва никогда не возвращаться и ее неоспоримый вердикт - "НЕГАТИВ".
    {"text": "I will never come back to this place again.", "label": "NEGATIVE"},
# Руна ']' запечатывает этот свиток, завершая перечень испытаний.
]