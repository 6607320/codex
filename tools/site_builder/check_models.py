import requests


def get_free_models():
    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()["data"]
        print(f"Всего моделей доступно: {len(data)}")
        print("-" * 40)
        print("💎 БЕСПЛАТНЫЕ МОДЕЛИ ( :free ):")
        print("-" * 40)

        free_models = []
        for model in data:
            # Ищем те, у которых в ID есть ':free'
            if ":free" in model["id"]:
                free_models.append(model)
                # Выводим ID и размер контекста (сколько текста влезает)
                context = model.get("context_length", "Unknown")
                print(f"• ID: {model['id']}")
                print(f"  Context: {context}")
                print("-" * 20)

        return free_models
    else:
        print("Ошибка подключения к OpenRouter")
        return []


if __name__ == "__main__":
    get_free_models()
