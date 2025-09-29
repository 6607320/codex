import os
import hashlib
import json
from fpdf import FPDF
from typing import List, Dict, Any, Set

# --- КОНСТАНТЫ ---
PROJECT_DIR = '../../'
EXCLUDE_DIRS = {'.git', '.dvc', '__pycache__', '.pytest_cache', 'build', 'dist', 'notebooklm_sources', '.venv', 'venv'}
TARGET_EXTENSIONS = ('.py', '.md', '.txt', '.json', '.yml', '.yaml', '.spec')
TARGET_FILES = ('Dockerfile', '.gitignore', '.dvcignore', '.flake8')
MANIFEST_FILE = '../../manifest.json'
OUTPUT_DIR = '../../notebooklm_sources'
WORD_LIMIT = 475000
LIVING_CHUNK_FILENAME = 'living_chronicle.pdf' ### ИЗМЕНЕНИЕ: Постоянное имя для активного тома ###

# --- ОСНОВНЫЕ ФУНКЦИИ (остаются без изменений) ---
def calculate_sha256(filepath: str) -> str:
    # ... (код без изменений)
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError:
        return ""

def scan_project_files(directory: str, extensions: tuple, filenames: tuple, exclude_dirs: set) -> List[str]:
    # ... (код без изменений)
    all_files = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(extensions) or file in filenames:
                filepath = os.path.join(root, file)
                all_files.append(filepath)
    try:
        script_path = os.path.abspath(__file__)
        all_files = [f for f in all_files if os.path.abspath(f) != script_path]
    except NameError:
        pass
    return sorted(all_files)

def load_manifest(manifest_path: str) -> Dict[str, Any]:
    # ... (код без изменений)
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"files": {}, "chunks": []}

def save_manifest(manifest_path: str, data: Dict[str, Any]):
    # ... (код без изменений)
    # Важно: сохраняем manifest в корне проекта, а не рядом со скриптом
    manifest_root_path = os.path.join(PROJECT_DIR, manifest_path)
    os.makedirs(os.path.dirname(manifest_root_path) or '.', exist_ok=True)
    with open(manifest_root_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def create_pdf_from_text(text_content: str, output_path: str):
    # ... (код без изменений, но с исправленным DeprecationWarning)
    print(f"Создание PDF по пути {output_path}...")
    try:
        pdf = FPDF()
        # uni=True убран, чтобы избежать предупреждения
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf')
        pdf.set_font('DejaVu', '', size=10)
        pdf.add_page()
        pdf.multi_cell(0, 5, text_content)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        pdf.output(output_path)
        print(f"   PDF-том успешно создан: {output_path}")
    except Exception as e:
        print(f"   !!! ОШИБКА при создании PDF {output_path}: {e}")

def main():
    """Главная функция, управляющая всем процессом."""
    print("--- Запуск строителя «Живой Летописи» ---")

    # Пути теперь строятся относительно корня проекта
    manifest_path = os.path.join(PROJECT_DIR, MANIFEST_FILE)
    output_dir_path = os.path.join(PROJECT_DIR, OUTPUT_DIR)
    os.makedirs(output_dir_path, exist_ok=True)

    manifest_data = load_manifest(manifest_path)
    
    print("1. Сканирование файлов проекта...")
    project_files = scan_project_files(PROJECT_DIR, TARGET_EXTENSIONS, TARGET_FILES, EXCLUDE_DIRS)
    current_files_hashes = {f: calculate_sha256(f) for f in project_files}
    print(f"   Найдено {len(project_files)} релевантных файлов.")

    print("2. Обнаружение изменений...")
    changed_or_new_files = [
        filepath for filepath, current_hash in current_files_hashes.items()
        if manifest_data["files"].get(filepath) != current_hash
    ]

    if not changed_or_new_files:
        print("   Изменений не найдено. Работа завершена.")
        return
    print(f"   Обнаружено {len(changed_or_new_files)} новых/измененных файлов.")

    print("3. Подготовка к инкрементальному обновлению (стратегия отката)...")
    files_to_reprocess: Set[str] = set(changed_or_new_files)

    ### ИЗМЕНЕНИЕ: Логика отката стала проще ###
    if manifest_data["chunks"]:
        last_chunk = manifest_data["chunks"][-1]
        # Откатываем только если последний чанк - это "живой" чанк. Архивные не трогаем.
        if last_chunk['file'] == LIVING_CHUNK_FILENAME:
            print(f"   Найден активный том '{LIVING_CHUNK_FILENAME}'. Производится откат.")
            manifest_data["chunks"].pop() # Удаляем его из манифеста в памяти
            
            living_chunk_path = os.path.join(output_dir_path, LIVING_CHUNK_FILENAME)
            try:
                if os.path.exists(living_chunk_path):
                    os.remove(living_chunk_path)
                    print(f"   Удален старый файл: {living_chunk_path}")
            except OSError as e:
                print(f"   Ошибка при удалении файла {living_chunk_path}: {e}")
            
            for source_file in last_chunk["sources"]:
                manifest_data["files"].pop(source_file, None)
            
            files_to_reprocess.update(last_chunk["sources"])
        else:
             print("   Последний том является архивным. Начинается новый активный том.")
    else:
        print("   Существующие тома не найдены. Начинается создание с нуля.")
    
    sorted_files_to_process = sorted(list(files_to_reprocess))
    print(f"   Всего файлов для обработки: {len(sorted_files_to_process)}")

    print("4. Наполнение новых томов...")
    
    ### ИЗМЕНЕНИЕ: Считаем только архивные тома для нумерации ###
    archived_chunks_count = len([c for c in manifest_data["chunks"] if c['file'] != LIVING_CHUNK_FILENAME])
    next_part_number = archived_chunks_count + 1
    
    current_chunk_content = ""
    current_chunk_sources = []
    current_chunk_word_count = 0

    for filepath in sorted_files_to_process:
        # ... (внутренняя логика чтения файла и подсчета слов остается той же)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except IOError:
            print(f"   ПРЕДУПРЕЖДЕНИЕ: Не удалось прочитать файл {filepath}. Пропускаем.")
            continue
        word_count = len(content.split())

        ### ИЗМЕНЕНИЕ: При достижении лимита, создаем АРХИВНЫЙ том ###
        if current_chunk_word_count > 0 and (current_chunk_word_count + word_count) > WORD_LIMIT:
            print(f"   Достигнут лимит слов. Архивируем том part_{next_part_number}.pdf...")
            part_filename = f"part_{next_part_number}.pdf"
            output_path = os.path.join(output_dir_path, part_filename)
            create_pdf_from_text(current_chunk_content, output_path)
            
            manifest_data["chunks"].append({
                "file": part_filename, "word_count": current_chunk_word_count, "sources": current_chunk_sources
            })
            
            next_part_number += 1
            current_chunk_content, current_chunk_sources, current_chunk_word_count = "", [], 0

        # Добавляем файл в текущий (пока еще безымянный) кусок
        file_header = f"\n\n--- НАЧАЛО: {filepath} ---\n\n"
        file_footer = f"\n\n--- КОНЕЦ: {filepath} ---\n\n"
        current_chunk_content += file_header + content + file_footer
        current_chunk_word_count += word_count
        current_chunk_sources.append(filepath)
        
        if filepath in current_files_hashes:
            manifest_data["files"][filepath] = current_files_hashes[filepath]

    ### ИЗМЕНЕНИЕ: Все, что осталось, сохраняем в "живой" том ###
    if current_chunk_word_count > 0:
        print(f"   Сохранение активного тома '{LIVING_CHUNK_FILENAME}' ({current_chunk_word_count} слов)...")
        output_path = os.path.join(output_dir_path, LIVING_CHUNK_FILENAME)
        create_pdf_from_text(current_chunk_content, output_path)
        
        manifest_data["chunks"].append({
            "file": LIVING_CHUNK_FILENAME, "word_count": current_chunk_word_count, "sources": current_chunk_sources
        })
        
    print("5. Обновление и сохранение файла летописи manifest.json...")
    save_manifest(MANIFEST_FILE, manifest_data)
    print("--- Работа успешно завершена! ---")

if __name__ == "__main__":
    main()