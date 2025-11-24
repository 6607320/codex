import os
import json
import re
import markdown
import textwrap
import hashlib

# === НАСТРОЙКИ ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "static")
QUESTS_DIR = os.path.join(OUTPUT_DIR, "quests")
INDEX_FILE = os.path.join(OUTPUT_DIR, "index.json")
CACHE_FILE = os.path.join(OUTPUT_DIR, "build_cache.json")
SCENARIOS_DIR = os.path.join(SCRIPT_DIR, "static", "scenarios")

print(f"📂 Корень проекта: {PROJECT_ROOT}")


def calculate_hash(content):
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def parse_codex():
    codex_path = os.path.join(PROJECT_ROOT, "CODEX.md")
    if not os.path.exists(codex_path):
        return [], ""
    with open(codex_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    readme_html = ""
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_html = markdown.markdown(f.read(), extensions=["fenced_code"])

    quests = []
    current_quest, capture_comment, comment_buffer = None, False, []
    current_part_num, current_part_title, current_scroll_title = (
        "0",
        "Неизвестная Часть",
        "Неизвестный Свиток",
    )

    part_pattern = re.compile(r"##\s+\*\*Часть\s+(\d+):\s*(.*?)\*\*")
    subheader_pattern = re.compile(r"^\s*###\s+")
    quest_pattern = re.compile(r"-\s+(?:\[.?\]\s+)?Квест\s+(\d+\.\d+):\s*(.*)")

    for line in lines:
        part_match = part_pattern.search(line)
        if part_match:
            current_part_num = part_match.group(1)
            current_part_title = (
                f"Часть {current_part_num}: {part_match.group(2).strip()}"
            )
            continue

        if subheader_pattern.search(line):
            raw_title = line.strip().replace("###", "").replace("**", "").strip()
            current_scroll_title = re.sub(r"\s*\(Статус:.*?\)", "", raw_title).strip()
            continue

        quest_match = quest_pattern.search(line)
        if quest_match:
            if current_quest:
                quests.append(current_quest)
            q_id = quest_match.group(1)
            q_title = re.sub(r"\s*`?\[.*?\]`?$", "", quest_match.group(2)).strip()
            current_quest = {
                "id": q_id,
                "title": q_title,
                "partNumber": current_part_num,
                "partTitle": current_part_title,
                "scrollTitle": current_scroll_title,
                "legend_md": "",
                "legend_html": "",
                "manifest_html": "<em>Свиток не найден...</em>",
                "status": "locked",
            }
            capture_comment, comment_buffer = False, []
            continue

        if current_quest:
            if "<!--" in line:
                capture_comment = True
            if capture_comment:
                comment_buffer.append(line)
            if "-->" in line and capture_comment:
                capture_comment = False
                full = "".join(comment_buffer)
                match = re.search(r"<!--(.*?)-->", full, re.DOTALL)
                if match:
                    raw = textwrap.dedent(match.group(1))
                    current_quest["legend_md"] = raw
                    current_quest["legend_html"] = markdown.markdown(raw)
                comment_buffer = []
    if current_quest:
        quests.append(current_quest)
    return quests, readme_html


def build():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(QUESTS_DIR):
        os.makedirs(QUESTS_DIR)
    if not os.path.exists(SCENARIOS_DIR):
        os.makedirs(SCENARIOS_DIR)

    cache, new_cache = load_cache(), {}
    quests_list, readme_html = parse_codex()
    index_data, updated_count = [], 0

    quest_fs_map = {}
    for root, _, _ in os.walk(PROJECT_ROOT):
        if any(d in root for d in ["site_builder", ".git", ".dvc"]):
            continue
        match = re.search(r"Scroll_(\d+)[\\/]Quest_(\d+)", root)
        if match and "manifest.md" in os.listdir(root):
            q_id = f"{int(match.group(1))}.{int(match.group(2))}"
            quest_fs_map[q_id] = root

    for quest in quests_list:
        q_id = quest["id"]
        quest_folder = quest_fs_map.get(q_id)
        manifest_content, scenario_data = "", []

        if quest_folder:
            manifest_path = os.path.join(quest_folder, "manifest.md")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_content = f.read()
                quest["manifest_html"] = markdown.markdown(
                    textwrap.dedent(manifest_content),
                    extensions=["fenced_code", "tables"],
                )
                quest["status"] = "active"

        # === ИЗМЕНЕНИЕ: Поддержка нового формата сценариев с _meta ===
        scenario_file = os.path.join(SCENARIOS_DIR, f"quest_{q_id}.json")
        if os.path.exists(scenario_file):
            try:
                with open(scenario_file, "r", encoding="utf-8") as f:
                    raw_scenario = json.load(f)

                    # Если это новый формат с метаданными
                    if isinstance(raw_scenario, dict) and "scenario" in raw_scenario:
                        scenario_data = raw_scenario["scenario"]
                    # Если это старый формат (просто список)
                    elif isinstance(raw_scenario, list):
                        scenario_data = raw_scenario
            except Exception:
                print(f"⚠️ Ошибка чтения сценария для {q_id}")
                scenario_data = []
        # =============================================================

        data_to_hash = (
            quest["title"]
            + quest["legend_md"]
            + manifest_content
            + json.dumps(scenario_data)
        )
        current_hash = calculate_hash(data_to_hash)
        new_cache[q_id] = current_hash
        json_path = os.path.join(QUESTS_DIR, f"quest_{q_id}.json")

        if cache.get(q_id) != current_hash or not os.path.exists(json_path):
            full_quest_data = {
                "id": quest["id"],
                "title": quest["title"],
                "legend": quest["legend_html"],
                "manifest": quest["manifest_html"],
                "status": quest["status"],
                "scenario": scenario_data,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(full_quest_data, f, ensure_ascii=False, indent=2)
            updated_count += 1

        index_data.append(
            {
                "id": quest["id"],
                "title": quest["title"],
                "partNumber": quest["partNumber"],
                "partTitle": quest["partTitle"],
                "scrollTitle": quest["scrollTitle"],
                "status": quest["status"],
            }
        )

    final_index = {"readme": readme_html, "quests": index_data}
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(final_index, f, ensure_ascii=False, indent=2)
    save_cache(new_cache)
    print(f"✨ Сборка завершена! Обновлено квестов: {updated_count}")


if __name__ == "__main__":
    build()
