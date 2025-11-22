import os
import json
import re
import glob
from openai import OpenAI
from dotenv import load_dotenv  # <-- Добавь этот импорт, если его нет

# === НАСТРОЙКИ ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
SCENARIOS_DIR = os.path.join(BASE_DIR, "static", "scenarios")
CODEX_FILE = os.path.join(PROJECT_ROOT, "CODEX.md")

# --- ВОТ ИЗМЕНЕНИЯ ---
# 1. Загружаем .env из папки tools/site_builder
load_dotenv(os.path.join(BASE_DIR, ".env"))

# 2. Читаем ключ из окружения
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "❌ ОШИБКА: Не найден ключ. Создай .env в папке tools/site_builder/"
    )
# --- КОНЕЦ ИЗМЕНЕНИЙ ---

# Выбранная нами модель: Grok 4.1 Fast (Бесплатная, 2M контекст)
MODEL_ID = "x-ai/grok-4.1-fast:free"

if not os.path.exists(SCENARIOS_DIR):
    os.makedirs(SCENARIOS_DIR)

# Инициализация клиента (OpenRouter использует стандартный протокол OpenAI)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def parse_codex_legends():
    """Читает CODEX.md и вытаскивает тексты заданий."""
    if not os.path.exists(CODEX_FILE):
        print(f"❌ Не найден {CODEX_FILE}")
        return {}

    with open(CODEX_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    legends = {}
    # Разбиваем файл по заголовкам квестов
    quest_blocks = re.split(r"-\s+(?:\[.?\]\s+)?Квест\s+(\d+\.\d+)", content)

    for i in range(1, len(quest_blocks), 2):
        q_id = quest_blocks[i]
        block_content = quest_blocks[i + 1]

        # Ищем текст внутри комментария <!-- ... -->
        match = re.search(r"<!--(.*?)-->", block_content, re.DOTALL)
        if match:
            legends[q_id] = match.group(1).strip()

    return legends


def get_quest_code(q_id):
    """Находит папку квеста и собирает весь код."""
    target_path = None
    try:
        s_num, q_num = q_id.split(".")
        # Используем (?!\d) чтобы убедиться, что после номера нет другой цифры (1 vs 10)
        pattern = re.compile(rf"Scroll_{s_num}(?!\d).*Quest_{q_num}(?!\d)")
    except Exception:
        return ""

    for root, _, _ in os.walk(PROJECT_ROOT):
        if "site_builder" in root:
            continue
        if pattern.search(root):
            target_path = root
            break

    if not target_path:
        return ""

    code_text = ""
    # Читаем ВСЕ важные файлы (у Grok много памяти, можно не экономить)
    extensions = [
        "*.py",
        "*.sh",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
    ]
    for ext in extensions:
        for file_path in glob.glob(os.path.join(target_path, ext)):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_text += (
                        f"\n=== FILE: {os.path.basename(file_path)} ===\n{f.read()}\n"
                    )
            except Exception:
                pass
    return code_text


def generate_scenario(q_id, legend, code):
    print(f"☁️ [Grok] Генерация для {q_id}...")

    prompt = f"""
    TASK: You are a Linux Terminal Simulator generator.
    Based on the INSTRUCTION (what the user should do) and the REAL CODE
    (what is actually in the files), generate a JSON scenario.

    INSTRUCTION:
    {legend}

    REAL CODE FILES:
    {code}

    REQUIREMENTS:
    1. Output ONLY valid JSON. No markdown, no comments.
    2. Format: Array of objects.
    - "command": The exact command user should type (e.g. "python quest.py"
        or "pip install..."). Infer this from the instruction and file names.
    - "output": Realistic terminal output.
        * If the python code has print("Hello"), the output MUST contain "Hello".
        * If it's a training script, generate fake but realistic logs (Epoch 1..).
        * If it's pip install, generate pip logs.
    - "is_final": boolean. Set to true ONLY for the very last command
        in the sequence.

    EXAMPLE JSON:
    [
    {{"command": "conda activate env", "output": "(env) user@host:~$ "}},
    {{"command": "python main.py", "output": "Starting...\\nDone.", "is_final": true}}
    ]
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Низкая температура = строгий JSON
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Ошибка API: {e}")
        return None


def clean_json(text):
    """Чистит ответ от markdown-обертки (```json ...

    ```)
    """
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        # Если не нашли массив, пробуем распарсить как есть
        return json.loads(text)
    except Exception:
        return None


def main():
    legends = parse_codex_legends()
    print(f"📜 Квестов в работе: {len(legends)}")

    # Проходим по ВСЕМ квестам
    for q_id, legend in legends.items():
        target_file = os.path.join(SCENARIOS_DIR, f"quest_{q_id}.json")

        # Если файл уже есть — пропускаем (чтобы не тратить токены зря)
        if os.path.exists(target_file):
            print(f"⏩ Скип: {q_id} (уже готов)")
            continue

        code = get_quest_code(q_id)
        if not code:
            # Если кода нет, генерируем заглушку (разбито для E501)
            code = (
                "(No code files found in directory. "
                "Generate generic logs based on instruction.)"
            )

        response = generate_scenario(q_id, legend, code)

        if response:
            data = clean_json(response)
            if data:
                with open(target_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✅ Готово: {target_file}")
            else:
                print(f"❌ Grok вернул невалидный JSON для {q_id}")


if __name__ == "__main__":
    main()
