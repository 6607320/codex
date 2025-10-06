# Призываем духа операционной системы `os` для взаимодействия с файлами и путями.
import os

# Призываем духа криптографии `hashlib` для сотворения нерушимых печатей (хешей).
import hashlib

# Призываем духа формата JSON для чтения и записи нашей Летописи.
import json

# Из великой библиотеки `fpdf` призываем сущность `FPDF`, нашего Духа-Каллиграфа.
from fpdf import FPDF

# Призываем сущности из `typing` для наложения на ритуалы рун ясности (аннотаций типов).
from typing import List, Dict, Any, Set

# --- Незыблемые Законы (Константы) ---
# Устанавливаем Путь к корню Великого Кодекса, где начнется Великий Обход.
PROJECT_DIR = "../../"
# Начертаем Черный Список Директорий, которые ритуал сканирования должен игнорировать.
EXCLUDE_DIRS = {
    ".git",
    ".dvc",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "notebooklm_sources",
}
# Определяем Истинные Имена Расширений тех свитков, что достойны нашего внимания.
TARGET_EXTENSIONS = (".py", ".md", ".txt", ".json", ".yml", ".yaml", ".spec")
# Определяем Истинные Имена особых свитков, что не имеют расширений, но также важны.
TARGET_FILES = ("Dockerfile", ".gitignore", ".dvcignore", ".flake8")
# Устанавливаем Путь к главному свитку Летописи, `manifest.json`.
MANIFEST_FILE = "../../manifest.json"
# Устанавливаем Путь к Хранилищу Томов, где будут материализованы PDF-свитки.
OUTPUT_DIR = "../../notebooklm_sources"
# Устанавливаем магический Лимит Слов для одного архивного PDF-тома.
WORD_LIMIT = 475000
# Даем Истинное Имя нашему вечно меняющемуся «Живому Тому».
LIVING_CHUNK_FILENAME = (
    "living_chronicle.pdf"  # ИЗМЕНЕНИЕ: Постоянное имя для активного тома ###
)


# --- Малые Ритуалы (Функции) ---
# Определяем ритуал `calculate_sha256` для Наложения Печати.
def calculate_sha256(filepath: str) -> str:
    # Создаем пустой магический котел для вычисления нерушимой печати SHA256.
    sha256_hash = hashlib.sha256()
    # Начинаем ритуал под защитой охранных рун на случай ошибок чтения.
    try:
        # Открываем свиток в бинарном режиме (`"rb"`) для чтения его плоти как есть.
        with open(filepath, "rb") as f:
            # Читаем свиток по частям (блоками по 4096 байт), чтобы не перегружать
            # память.
            for byte_block in iter(lambda: f.read(4096), b""):
                # Каждую прочитанную часть бросаем в котел, постепенно формируя печать.
                sha256_hash.update(byte_block)
        # Возвращаем готовую печать в виде шестнадцатеричной текстовой строки.
        return sha256_hash.hexdigest()
    # Если ритуал чтения свитка был прерван ошибкой...
    except IOError:
        # ...возвращаем пустоту как знак неудачи.
        return ""


# Определяем ритуал `scan_project_files` для проведения Великого Обхода Кодекса.
def scan_project_files(
    directory: str, extensions: tuple, filenames: tuple, exclude_dirs: set
) -> List[str]:
    # Готовим пустой магический мешок для сбора путей ко всем достойным свиткам.
    all_files = []
    # Начинаем Великий Обход Кодекса с помощью духа `os.walk`.
    for root, dirs, files in os.walk(directory, topdown=True):
        # Изгоняем папки из Черного Списка из списка `dirs` на лету, чтобы
        # `os.walk` в них не заходил.
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        # В каждой посещенной папке перебираем все найденные в ней свитки.
        for file in files:
            # Проверяем, соответствует ли имя свитка нашим Истинным Именами.
            if file.endswith(extensions) or file in filenames:
                # Соединяем путь к папке (`root`) и имя свитка (`file`), создавая полный
                # путь.
                filepath = os.path.join(root, file)
                # Помещаем найденный канонический путь в наш магический мешок.
                all_files.append(filepath)
    # Начинаем ритуал под защитой охранных рун.
    try:
        # Определяем абсолютный путь к самому этому пергаменту (`rag_builder.py`).
        script_path = os.path.abspath(__file__)
        # Изгоняем путь к самому себе из списка, ибо ритуал не должен
        # анализировать сам себя.
        all_files = [f for f in all_files if os.path.abspath(f) != script_path]
    # Если ритуал запущен в среде, где `__file__` не определен...
    except NameError:
        # ...просто продолжаем, ничего не делая.
        pass
    # Возвращаем отсортированный по алфавиту список путей для порядка и предсказуемости.
    return sorted(all_files)


# Определяем ритуал `load_manifest` для чтения древней Летописи.
def load_manifest(manifest_path: str) -> Dict[str, Any]:
    # Начинаем ритуал под защитой охранных рун на случай, если Летопись еще не
    # сотворена.
    try:
        # Открываем свиток Летописи для чтения с кодировкой UTF-8.
        with open(manifest_path, "r", encoding="utf-8") as f:
            # Преобразуем текст свитка из JSON в структуру данных и возвращаем её.
            return json.load(f)
    # Если Летопись не найдена или искажена...
    except (FileNotFoundError, json.JSONDecodeError):
        # ...сотворяем и возвращаем новую, пустую Летопись изначальной структуры.
        return {"files": {}, "chunks": []}


# Определяем ритуал `save_manifest` для запечатывания обновленной Летописи.
def save_manifest(manifest_path: str, data: Dict[str, Any]):
    # Собираем канонический путь к Летописи в корне Кодекса.
    manifest_root_path = os.path.join(PROJECT_DIR, manifest_path)
    # Сотворяем нужную директорию для Летописи, если она еще не существует.
    os.makedirs(os.path.dirname(manifest_root_path) or ".", exist_ok=True)
    # Открываем свиток Летописи для записи с кодировкой UTF-8.
    with open(manifest_root_path, "w", encoding="utf-8") as f:
        # Записываем наши знания в свиток в формате JSON с отступами.
        json.dump(data, f, indent=4, ensure_ascii=False)


# Определяем ритуал `create_pdf_from_text` для сотворения PDF-тома.
def create_pdf_from_text(text_content: str, output_path: str):
    # Оповещаем Мага-Техноманта о начале сотворения нового тома.
    print(f"Создание PDF по пути {output_path}...")
    # Начинаем ритуал под защитой охранных рун на случай ошибок сотворения.
    try:
        # Призываем в мир новую, пустую сущность PDF-свитка.
        pdf = FPDF()
        # Обучаем Духа-Каллиграфа руническому шрифту `DejaVuSans.ttf`.
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf")
        # Приказываем Духу-Каллиграфу использовать этот шрифт размером 10.
        pdf.set_font("DejaVu", "", size=10)
        # Добавляем в наш PDF-свиток первый чистый пергаментный лист.
        pdf.add_page()
        # Вписываем все текстовое содержимое на лист, с автоматическим переносом строк.
        pdf.multi_cell(0, 5, text_content)
        # Убеждаемся, что существует папка, в которую мы хотим сохранить наш том.
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Приказываем Духу-Каллиграфу запечатать и материализовать PDF-свиток.
        pdf.output(output_path)
        # Сообщаем об успешном завершении ритуала сотворения тома.
        print(f"   PDF-том успешно создан: {output_path}")
    # Если охранные руны сработали во время сотворения...
    except Exception as e:
        # ...мы сообщаем о провале ритуала и причине неудачи.
        print(f"   !!! ОШИБКА при создании PDF {output_path}: {e}")


# Определяем Великий Ритуал `main`, который управляет всем процессом.
def main():
    # Этот docstring описывает назначение функции `main`.
    """Главная функция, управляющая всем процессом."""
    # Объявляем о начале Великого Ритуала.
    print("--- Запуск строителя «Живой Летописи» ---")
    # Собираем полный путь к Летописи, соединяя корень Кодекса и имя файла Летописи.
    manifest_path = os.path.join(PROJECT_DIR, MANIFEST_FILE)
    # Собираем полный путь к Хранилищу Томов.
    output_dir_path = os.path.join(PROJECT_DIR, OUTPUT_DIR)
    # Убеждаемся, что Хранилище Томов существует, создавая его при необходимости.
    os.makedirs(output_dir_path, exist_ok=True)
    # Проводим ритуал чтения Летописи, чтобы узнать о предыдущем состоянии Кодекса.
    manifest_data = load_manifest(manifest_path)
    # Объявляем о начале фазы инвентаризации.
    print("1. Сканирование файлов проекта...")
    # Запускаем Великий Обход Кодекса, чтобы получить список всех релевантных свитков.
    project_files = scan_project_files(
        PROJECT_DIR, TARGET_EXTENSIONS, TARGET_FILES, EXCLUDE_DIRS
    )
    # Для каждого найденного свитка вычисляем его отпечаток души (хеш).
    current_files_hashes = {f: calculate_sha256(f) for f in project_files}
    # Сообщаем, сколько всего свитков было найдено в Кодексе.
    print(f"   Найдено {len(project_files)} релевантных файлов.")
    # Объявляем о начале фазы сравнения прошлого и настоящего.
    print("2. Обнаружение изменений...")
    # Создаем список свитков, которые либо являются новыми, либо их печать изменилась.
    changed_or_new_files = [
        # Перебираем все текущие свитки и их печати.
        filepath
        for filepath, current_hash in current_files_hashes.items()
        # Сравниваем текущую печать с той, что записана в Летописи.
        if manifest_data["files"].get(filepath) != current_hash
        # Завершаем создание списка.
    ]
    # Если список измененных свитков пуст...
    if not changed_or_new_files:
        # ...сообщаем, что Кодекс в гармонии, и завершаем ритуал.
        print("   Изменений не найдено. Работа завершена.")
        # Прерываем Великий Ритуал.
        return
    # Сообщаем, сколько свитков требуют нашего внимания.
    print(f"   Обнаружено {len(changed_or_new_files)} новых/измененных файлов.")
    # Объявляем о начале фазы отката для подготовки к обновлению.
    print("3. Подготовка к инкрементальному обновлению (стратегия отката)...")
    # Готовим котел для переплавки, помещая в него все новые и измененные свитки.
    files_to_reprocess: Set[str] = set(changed_or_new_files)
    # Если в Летописи уже есть записи о сотворенных томах...
    if manifest_data["chunks"]:
        # ...изучаем запись о самом последнем сотворенном томе.
        last_chunk = manifest_data["chunks"][-1]
        # Если последний том — это наш «Живой Том»...
        if last_chunk["file"] == LIVING_CHUNK_FILENAME:
            # ...сообщаем о начале ритуала отката.
            print(
                f"   Найден активный том '{LIVING_CHUNK_FILENAME}'. Производится откат."
            )
            # Изгоняем запись о «Живом Томе» из Летописи в памяти.
            manifest_data["chunks"].pop()
            # Собираем полный путь к материальной плоти «Живого Тома».
            living_chunk_path = os.path.join(output_dir_path, LIVING_CHUNK_FILENAME)
            # Начинаем ритуал под защитой охранных рун.
            try:
                # Проверяем, существует ли еще материальная плоть тома.
                if os.path.exists(living_chunk_path):
                    # Уничтожаем старый PDF-файл.
                    os.remove(living_chunk_path)
                    # Сообщаем об успехе.
                    print(f"   Удален старый файл: {living_chunk_path}")
            # Если ритуал уничтожения был прерван ошибкой...
            except OSError as e:
                # ...сообщаем о неудаче.
                print(f"   Ошибка при удалении файла {living_chunk_path}: {e}")
            # Начинаем цикл по всем свиткам, которые были запечатаны в уничтоженном
            # томе.
            for source_file in last_chunk["sources"]:
                # Изгоняем из Летописи в памяти запись о каждом из этих свитков.
                manifest_data["files"].pop(source_file, None)
            # Бросаем все «освобожденные» свитки обратно в котел на переплавку.
            files_to_reprocess.update(last_chunk["sources"])
        # Если последний том был архивным...
        else:
            # ...мы его не трогаем, а сообщаем, что будем создавать новый «Живой Том».
            print("   Последний том является архивным. Начинается новый активный том.")
    # Если в Летописи вообще не было томов...
    else:
        # ...сообщаем, что начинаем Великое Сотворение с самого начала.
        print("   Существующие тома не найдены. Начинается создание с нуля.")
    # Превращаем наш котел (множество) в отсортированный список для
    # предсказуемого порядка.
    sorted_files_to_process = sorted(list(files_to_reprocess))
    # Сообщаем итоговое количество свитков, которые будут переработаны.
    print(f"   Всего файлов для обработки: {len(sorted_files_to_process)}")
    # Объявляем о начале главной фазы — сотворения новых томов.
    print("4. Наполнение новых томов...")
    # Подсчитываем, сколько у нас уже есть запечатанных АРХИВНЫХ томов.
    archived_chunks_count = len(
        [c for c in manifest_data["chunks"] if c["file"] != LIVING_CHUNK_FILENAME]
    )
    # Определяем номер для следующего АРХИВНОГО тома.
    next_part_number = archived_chunks_count + 1
    # Готовим пустой астральный сосуд для сбора текстового содержимого.
    current_chunk_content = ""
    # Готовим пустой список для имен свитков в текущем томе.
    current_chunk_sources = []
    # Обнуляем счетчик слов для текущего тома.
    current_chunk_word_count = 0
    # Начинаем магический круг, перебирая каждый свиток для обработки.
    for filepath in sorted_files_to_process:
        # Начинаем ритуал под защитой охранных рун.
        try:
            # Открываем свиток для чтения в текстовом режиме.
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                # Извлекаем все текстовое содержимое свитка.
                content = f.read()
        # Если ритуал чтения был прерван ошибкой...
        except IOError:
            # ...сообщаем о проблеме и пропускаем этот свиток.
            print(
                f"   ПРЕДУПРЕЖДЕНИЕ: Не удалось прочитать файл {filepath}. Пропускаем."
            )
            # Переходим к следующему свитку в магическом круге.
            continue
        # Подсчитываем количество слов в извлеченном содержимом.
        word_count = len(content.split())
        # Если текущий том не пуст и добавление нового свитка превысит Лимит Слов...
        if (
            current_chunk_word_count > 0
            and (current_chunk_word_count + word_count) > WORD_LIMIT
        ):
            # ...объявляем о запечатывании текущего тома как АРХИВНОГО.
            print(
                f"   Достигнут лимит слов. Архивируем том part_{next_part_number}.pdf..."
            )
            # Даем имя новому архивному тому.
            part_filename = f"part_{next_part_number}.pdf"
            # Собираем полный путь для его материализации.
            output_path = os.path.join(output_dir_path, part_filename)
            # Запускаем ритуал сотворения PDF-тома.
            create_pdf_from_text(current_chunk_content, output_path)
            # Добавляем в Летопись в памяти новую запись об этом архивном томе.
            manifest_data["chunks"].append(
                {
                    # Записываем его имя, количество слов и список свитков-источников.
                    "file": part_filename,
                    "word_count": current_chunk_word_count,
                    "sources": current_chunk_sources,
                    # Завершаем создание записи.
                }
            )
            # Увеличиваем номер для следующего возможного архивного тома.
            next_part_number += 1
            # Очищаем сосуды и счетчики, чтобы начать собирать следующий том с чистого
            # листа.
            current_chunk_content, current_chunk_sources, current_chunk_word_count = (
                "",
                [],
                0,
            )
        # Создаем магический заголовок, указывающий происхождение содержимого.
        file_header = f"\n\n--- НАЧАЛО: {filepath} ---\n\n"
        # Создаем магическое завершение, чтобы отделить свитки друг от друга.
        file_footer = f"\n\n--- КОНЕЦ: {filepath} ---\n\n"
        # Добавляем содержимое свитка, обрамленное метаданными, в астральный сосуд.
        current_chunk_content += file_header + content + file_footer
        # Увеличиваем счетчик слов текущего тома.
        current_chunk_word_count += word_count
        # Добавляем имя свитка в список источников для текущего тома.
        current_chunk_sources.append(filepath)
        # Если для данного файла у нас есть свежая печать...
        if filepath in current_files_hashes:
            # ...обновляем (или добавляем) запись о файле и его новой печати в Летопись в памяти.
            manifest_data["files"][filepath] = current_files_hashes[filepath]
    # После завершения круга, проверяем, осталось ли что-то в нашем сосуде.
    if current_chunk_word_count > 0:
        # Объявляем о сотворении финального, «Живого Тома».
        print(
            f"   Сохранение активного тома '{LIVING_CHUNK_FILENAME}' ({current_chunk_word_count} слов)..."
        )
        # Собираем путь для его материализации, используя его постоянное Имя.
        output_path = os.path.join(output_dir_path, LIVING_CHUNK_FILENAME)
        # Запускаем ритуал сотворения PDF.
        create_pdf_from_text(current_chunk_content, output_path)
        # Добавляем в Летопись запись о «Живом Томе».
        manifest_data["chunks"].append(
            {
                # Записываем его постоянное имя, количество слов и источники.
                "file": LIVING_CHUNK_FILENAME,
                "word_count": current_chunk_word_count,
                "sources": current_chunk_sources,
                # Завершаем создание записи.
            }
        )
    # Объявляем о финальном акте — запечатывании Летописи.
    print("5. Обновление и сохранение файла летописи manifest.json...")
    # Запускаем ритуал сохранения, передавая ему обновленные знания из памяти.
    save_manifest(MANIFEST_FILE, manifest_data)
    # Объявляем об успешном завершении Великого Ритуала.
    print("--- Работа успешно завершена! ---")


# Эта руна проверяет, был ли пергамент призван напрямую.
if __name__ == "__main__":
    # Призываем наш Великий Ритуал `main` к исполнению.
    main()
