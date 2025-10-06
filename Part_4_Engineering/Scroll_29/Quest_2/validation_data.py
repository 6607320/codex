# === validation_data.py ===
# Имя этого пергамента, в котором хранится наш "Кристалл Истины".

# Мы начинаем ритуал сотворения нашего "Кристалла Истины"
# (`VALIDATION_SET`) — это священный перечень, хранящий неоспоримые
# эталоны.
VALIDATION_SET = [
    # Первый эталонный свиток: руническая надпись (`text`) и ее истинная суть
    # (`label`).
    {
        "text": "I love this product, it is absolutely amazing!",
        "label": "POSITIVE",
    },
    # Второй эталон: руническая надпись с явно негативной "истинной сутью".
    {
        "text": "This is the worst service I have ever received in my life.",
        "label": "NEGATIVE",
    },
    # Третий эталон, подтверждающий позитивную магию.
    {
        "text": "The movie was fantastic, a true masterpiece.",
        "label": "POSITIVE",
    },
    # Четвертый эталон, содержащий негативную магию разочарования.
    {
        "text": "I am so disappointed with the quality, it broke after one day.",
        "label": "NEGATIVE",
    },
    # Пятый эталон: еще одна руническая надпись с позитивной "истинной сутью".
    {
        "text": "What a wonderful experience, I would recommend it to everyone.",
        "label": "POSITIVE",
    },
    # Шестой эталон, описывающий бесполезность артефакта.
    {
        "text": "A complete waste of time and money, I regret buying this.",
        "label": "NEGATIVE",
    },
    # Седьмой эталон, хранящий позитивное воспоминание о помощи.
    {"text": "The team was very helpful and friendly.", "label": "POSITIVE"},
    # Восьмой эталон, запечатывающий негативный опыт.
    {
        "text": "The food was terrible and the waiter was rude.",
        "label": "NEGATIVE",
    },
    # Девятый эталон: позитивная оценка мастерства.
    {
        "text": "An outstanding performance by the entire cast.",
        "label": "POSITIVE",
    },
    # Десятый, финальный эталон, выражающий твердое негативное намерение.
    {
        "text": "I will never come back to this place again.",
        "label": "NEGATIVE",
    },
    # Закрывающая скобка `]` — это руна, которая "запечатывает" наш "Кристалл
    # Истины", завершая его сотворение.
]
