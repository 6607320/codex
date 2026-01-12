# **Квест 24.2: Ковка "Самодостаточного Артефакта" — Отчет об Исследовании**

## **Оглавление**

1. **Введение: Природа Самодостаточного Артефакта**
2. **Глава I: Архитектурные Основы PyInstaller и Проблема Размера**
   - Механизм Замораживания Интерпретатора
   - Анализ Зависимостей и Графы Импорта
   - Анатомия "Hello World" на 3 Гигабайта
3. **Глава II: Экосистема PyTorch — Разбор Монолита**
   - Бинарная Структура: Python против C++
   - Роль Аппаратных Ускорителей (CUDA и cuDNN)
   - Проблема Динамической Компиляции (JIT)
4. **Глава III: Стратегия "Чистой Комнаты" (Clean Room Strategy)**
   - Изоляция Окружения: Venv против Conda
   - Управление Версиями и Индексами Пакетов
   - Протокол Установки CPU-версий
5. **Глава IV: Инженерная Работа с Файлами Спецификаций (.spec)**
   - Программируемая Сборка: Класс Analysis
   - Алгоритмическая Фильтрация Бинарных Файлов
   - Управление Скрытыми Импортами и Хуками
6. **Глава V: Техники Глубокой Оптимизации и Исключения**
   - Хирургическое Удаление Подмодулей
   - Кейс: torch.distributions и Тестовые Наборы
   - Проблема Исходных Кодов для TorchScript
7. **Глава VI: Сжатие и Альтернативные Компиляторы**
   - UPX: Преимущества и Риски Стабильности
   - Nuitka: Трансляция в C++ против Упаковки
   - Сравнительный Анализ Производительности и Размера
8. **Глава VII: Диагностика и Разрешение Конфликтов**
   - Отладка Загрузчика и Ошибки Импорта
   - Кросс-платформенные Нюансы и Драйверы
9. **Заключение и Итоговые Рекомендации**

## ---

**Введение: Природа Самодостаточного Артефакта**

В современной разработке программного обеспечения на языке Python, особенно в сфере машинного обучения (ML), существует фундаментальный разрыв между средой разработки и средой эксплуатации. Разработчик оперирует в экосистеме, богатой инструментами отладки, пакетными менеджерами и динамическими связями. Однако конечный пользователь или целевой сервер зачастую требуют того, что в рамках Квеста 24.2 именуется "Самодостаточным Артефактом" — единого исполняемого файла, не требующего предварительной установки интерпретатора Python, библиотек или настройки переменных окружения.  
Задача трансформации динамического скрипта PyTorch в статический бинарный файл является нетривиальной инженерной проблемой. PyTorch, по своей сути, представляет собой массивную обертку над библиотеками C++ (LibTorch), CUDA (вычисления на GPU) и cuDNN (нейросетевые примитивы). Стандартные инструменты упаковки, такие как PyInstaller, при наивном использовании создают дистрибутивы, размер которых часто превышает 3 гигабайта.1 Это явление, известное как "раздувание бинарного файла" (binary bloating), делает развертывание приложений на граничных устройствах (Edge devices) или пользовательских машинах практически невозможным.  
Данный отчет представляет собой исчерпывающее руководство по преодолению этих ограничений. Базируясь на анализе технической документации, обсуждений в сообществе разработчиков и эмпирических данных, мы деконструируем процесс сборки, выявляем причины избыточности и предлагаем многоуровневую стратегию оптимизации. Цель отчета — не просто предоставить набор команд, а сформировать глубокое понимание механизмов, лежащих в основе создания минималистичных и производительных исполняемых файлов.

## ---

**Глава I: Архитектурные Основы PyInstaller и Проблема Размера**

Чтобы эффективно бороться с размером артефакта, необходимо детально понимать, как именно PyInstaller преобразует скрипт в исполняемый файл. Это не компиляция в традиционном смысле (как в C или Go), а сложный процесс упаковки и виртуализации файловой системы.

### **Механизм Замораживания Интерпретатора**

PyInstaller функционирует как система статического анализа и упаковки. В режиме \--onefile он создает архив, который содержит:

1. **Загрузчик (Bootloader):** Небольшая программа на C, которая запускается операционной системой.
2. **Интерпретатор Python:** Обычно в виде динамической библиотеки (python3.dll или libpython.so).
3. **Скрипты и Модули:** Упакованные в архивный формат PYZ (Python Zip).
4. **Бинарные Зависимости:** Файлы расширений (.pyd, .so) и динамические библиотеки (.dll), необходимые для работы модулей.4

При запуске такого "самодостаточного" файла происходит процесс, невидимый для пользователя, но критичный для понимания производительности и дискового пространства:

- Загрузчик создает временную директорию (обычно в /tmp/\_MEIxxxx на Linux или %TEMP%/\_MEIxxxx на Windows).
- Все библиотеки и зависимости распаковываются в эту директорию.
- Загрузчик модифицирует sys.path встроенного интерпретатора, указывая на эту временную папку.
- Запускается входной скрипт пользователя.4

**Инсайт:** Это означает, что приложение размером 3 ГБ требует не только 3 ГБ для хранения самого .exe файла, но и дополнительные 3 ГБ свободного места во временной директории для каждого запуска, а также значительное время на распаковку, что создает задержку при старте ("cold start latency").6

### **Анализ Зависимостей и Графы Импорта**

Сердцем PyInstaller является фаза анализа (Analysis). Инструмент сканирует абстрактное синтаксическое дерево (AST) скрипта, выявляет все инструкции import и рекурсивно ищет соответствующие файлы на диске.7  
Здесь кроется корень проблемы размера. PyInstaller спроектирован с философией "лучше перебдеть, чем недобдеть". Если ваш скрипт делает import torch, PyInstaller находит пакет torch в site-packages. Поскольку torch — это сложный пакет, содержащий биндинги к нативным библиотекам, PyInstaller включает в сборку все найденные бинарные файлы, связанные с пакетом. В стандартной установке PyTorch это включает в себя полную поддержку CUDA, даже если код пользователя не использует GPU.5  
Кроме того, механизм анализа плохо справляется с динамическими импортами (например, \_\_import\_\_('name')), которые часто используются в плагинных системах. Чтобы избежать ошибок ModuleNotFoundError в рантайме, PyInstaller использует систему "хуков" (hooks) — предопределенных скриптов, которые принудительно указывают, какие скрытые импорты нужно включить. Хуки для популярных библиотек (numpy, scipy, torch) часто включают огромные массивы данных и бинарников "на всякий случай".9

### **Анатомия "Hello World" на 3 Гигабайта**

Многочисленные отчеты разработчиков свидетельствуют о том, что даже простейший скрипт, импортирующий PyTorch, разрастается до гигантских размеров.

- _Случай из практики:_ Пользователь сообщает, что простое приложение с использованием Selenium и PyTorch достигает 1.5 ГБ, а с полным набором CUDA библиотек — более 3 ГБ.5
- _Причина:_ Библиотеки NVIDIA (CUDA Toolkit). Стандартный пакет torch включает в себя статические копии библиотек для линейной алгебры (cublas), нейросетевых примитивов (cudnn), и преобразований Фурье (cufft).

Таблица 1.1 иллюстрирует примерный вклад различных компонентов в размер стандартного дистрибутива PyTorch.

| Компонент                 | Примерный Размер (сжатый/распакованный) | Функция                      |
| :------------------------ | :-------------------------------------- | :--------------------------- |
| **CUDA Runtime (cudart)** | \~500 KB / 1 MB                         | Базовое взаимодействие с GPU |
| **cuDNN (DLLs)**          | \~300 MB / 700 MB                       | Примитивы глубокого обучения |
| **cuBLAS (DLLs)**         | \~150 MB / 400 MB                       | Линейная алгебра на GPU      |
| **LibTorch (C++ Core)**   | \~200 MB / 500 MB                       | Основной движок тензоров     |
| **Python Interface**      | \~50 MB / 100 MB                        | Python-обертки               |
| **MKL / OpenBLAS**        | \~100 MB / 200 MB                       | Оптимизация для CPU          |

Как видно из таблицы, львиную долю объема занимают библиотеки поддержки GPU. Если PyInstaller слепо копирует их в сборку, размер неизбежно достигает нескольких гигабайт.

## ---

**Глава II: Экосистема PyTorch — Разбор Монолита**

Чтобы оптимизировать сборку, необходимо понимать, что PyTorch — это не просто библиотека Python, а полноценная платформа, встроенная в Python.

### **Бинарная Структура: Python против C++**

PyTorch построен на базе библиотеки LibTorch, написанной на C++. Python-код в пакете torch служит высокоуровневым интерфейсом. Это создает двойную зависимость при сборке:

1. PyInstaller должен собрать все .py файлы для логики.
2. PyInstaller должен найти и сохранить связи с .dll (Windows) или .so (Linux) файлами, которые реализуют низкоуровневые операции.

Проблема усугубляется тем, что PyTorch использует RPATH (на Linux) или поиск в директории приложения (на Windows) для загрузки этих библиотек. При перемещении в "замороженный" бандл пути меняются. PyInstaller пытается решить это, собирая все найденные библиотеки в одну папку (или корень архива), что иногда приводит к конфликтам имен или дублированию библиотек, если разные пакеты зависят от разных версий одной и той же C-библиотеки (например, libomp или mkl).12

### **Роль Аппаратных Ускорителей (CUDA и cuDNN)**

CUDA (Compute Unified Device Architecture) — это проприетарная платформа NVIDIA. Библиотеки CUDA являются внешними зависимостями. Однако, для удобства пользователей (user experience), PyTorch распространяется через pip в виде wheel-файлов, которые **уже содержат** необходимые версии библиотек CUDA. Это называется "statical linking" или "vendoring".  
Это решение разработчиков PyTorch, идеальное для простоты установки (pip install torch и всё работает), является катастрофой для PyInstaller. Инструмент видит эти библиотеки внутри пакета torch и считает их неотъемлемой частью приложения. Он не может знать, будет ли пользователь использовать GPU, поэтому он пакует всё.  
Более того, различные версии PyTorch могут поставляться с поддержкой разных архитектур CUDA (v11.8, v12.1), каждая из которых имеет свой набор тяжелых библиотек.13  
**Критический Инсайт:** PyInstaller не анализирует _исполнение_ кода. Он анализирует _наличие_ файлов. Если в папке site-packages/torch/lib лежат гигабайты библиотек CUDA, они попадут в сборку, даже если ваш код — это print(torch.tensor()), выполняющийся на CPU.

### **Проблема Динамической Компиляции (JIT)**

PyTorch обладает встроенным JIT-компилятором (TorchScript), который позволяет сериализовать модели и оптимизировать их исполнение. Механизм JIT часто требует доступа к **исходному коду** (.py) моделей для их инспекции и компиляции в промежуточное представление.15  
Стандартное поведение PyInstaller — компилировать .py файлы в байт-код .pyc и удалять исходники для экономии места и обфускации. Это приводит к специфическим ошибкам вида OSError: Can't get source for \<function...\> при попытке использовать torch.jit.script или при загрузке некоторых моделей из библиотеки timm или torchvision.  
Для "Самодостаточного Артефакта" это означает необходимость явного включения исходных файлов определенных модулей в сборку, что противоречит интуитивному желанию удалить всё лишнее. Это тонкий баланс между размером и функциональностью, который решается через настройку файла спецификации (о чем будет сказано в Главе IV).

## ---

**Глава III: Стратегия "Чистой Комнаты" (Clean Room Strategy)**

Самый эффективный метод борьбы с размером артефакта лежит не в настройках сжатия, а в управлении входящими данными. Принцип "Garbage In, Garbage Out" (Мусор на входе — мусор на выходе) здесь работает абсолютно буквально.

### **Изоляция Окружения: Venv против Conda**

Многие исследователи данных используют дистрибутив Anaconda, который по умолчанию устанавливает сотни библиотек (pandas, jupyter, scipy, matplotlib, qt). Если запустить PyInstaller в таком "базовом" окружении, высока вероятность "перекрестного опыления" (cross-contamination).  
Например, torch может иметь опциональную зависимость от numpy. В окружении Conda numpy может быть слинкован с библиотеками MKL (Math Kernel Library) от Intel, которые огромны. PyInstaller потянет torch \-\> numpy \-\> mkl, добавляя сотни мегабайт, даже если они не нужны.12  
**Рекомендация:** Единственно верный путь для Квеста 24.2 — создание стерильного виртуального окружения.

Bash

\# Создание чистого окружения  
python \-m venv.build_env

\# Активация (Windows)  
.build_env\\Scripts\\activate

В этом окружении должны быть установлены **только** те библиотеки, которые непосредственно импортируются в проекте, плюс pyinstaller. Это гарантирует, что никакие "призрачные" зависимости не попадут в сборку.1

### **Протокол Установки CPU-версий**

Если целевой "Самодостаточный Артефакт" не обязан использовать GPU (например, это клиентское приложение для инференса на офисных ноутбуках), самым мощным рычагом оптимизации является установка CPU-версии PyTorch.  
Стандартная команда pip install torch тянет версию с CUDA. Для установки легкой версии необходимо использовать флаг \--index-url, указывающий на репозиторий, где хранятся сборки без CUDA.  
**Команда для установки:**

Bash

pip install torch torchvision \--index-url https://download.pytorch.org/whl/cpu

Этот шаг заменяет пакет torch размером \~2.5 ГБ на пакет размером \~200 МБ.19  
**Анализ воздействия:**

- **До оптимизации:** Сборка включает cublas64_11.dll, cudnn_cnn_infer64_8.dll и т.д.
- **После оптимизации:** Этих файлов физически нет в директории site-packages. PyInstaller при всем желании не сможет их включить.
- **Результат:** Моментальное уменьшение размера итогового файла на 80-90%.

### **Управление Версиями и Индексами Пакетов**

Использование инструментов управления зависимостями, таких как poetry или uv, позволяет закрепить (pin) конкретные версии и источники.  
Например, в poetry можно явно указать источник:

Ini, TOML

\[\[tool.poetry.source\]\]  
name \= "pytorch_cpu"  
url \= "https://download.pytorch.org/whl/cpu"  
priority \= "explicit"

Это предотвращает случайное обновление до GPU-версии при очередном poetry update, обеспечивая воспроизводимость сборки "чистого" артефакта.22

## ---

**Глава IV: Инженерная Работа с Файлами Спецификаций (.spec)**

Командная строка PyInstaller (pyinstaller script.py \--onefile) удобна для быстрых тестов, но недостаточна для сложного проекта на PyTorch. Профессиональный подход требует использования файлов спецификаций (.spec). Это Python-скрипты, которые управляют процессом сборки.

### **Программируемая Сборка: Класс Analysis**

Файл .spec создается командой pyi-makespec script.py. Внутри него создается объект класса Analysis, который принимает ключевые аргументы:

- binaries: Список кортежей (путь*к*файлу, папка*в*архиве).
- datas: Неисполняемые файлы (конфиги, веса моделей).
- hiddenimports: Модули, невидимые для статического анализатора.
- excludes: Модули для принудительного исключения.

Пример структуры:

Python

a \= Analysis(  
 \['main.py'\],  
 pathex=,  
 binaries=,  
 datas=,  
 hiddenimports=,  
 hookspath=,  
 hooksconfig={},  
 runtime_hooks=,  
 excludes=\['tkinter', 'unittest'\],  
 noarchive=False,  
)

Редактирование этого файла позволяет вмешиваться в процесс сборки _после_ анализа, но _до_ упаковки.4

### **Алгоритмическая Фильтрация Бинарных Файлов**

Даже при использовании чистого окружения, некоторые лишние бинарники могут просочиться. Например, если необходим numpy, он может потянуть лишние библиотеки OpenBLAS.  
С мощью Python внутри .spec файла можно реализовать алгоритмическую фильтрацию свойства a.binaries.  
**Паттерн фильтрации:**

Python

\# Внутри.spec файла после инициализации Analysis  
exclusions \= \['mkl', 'libopenblas'\] \# Примеры подстрок  
a.binaries \= \[x for x in a.binaries if not any(e in x.lower() for e in exclusions)\]

Этот код проходит по списку всех найденных DLL и удаляет те, имена которых содержат запрещенные подстроки. Это хирургический инструмент, позволяющий, например, удалить конкретные библиотеки CUDA, если по какой-то причине полная CPU-установка невозможна, но нужно уменьшить размер.23

### **Управление Скрытыми Импортами и Хуками**

В сложных проектах PyInstaller часто пропускает импорты. В логах это выглядит как ModuleNotFoundError.  
Существует два способа решения:

1. **Аргумент hiddenimports:** Просто добавьте имя модуля в список в .spec файле.  
   Python  
   hiddenimports=\['sklearn.utils.\_cython_blas', 'timm.models.resnet'\]

2. **Хуки (Hooks):** PyInstaller имеет встроенную систему хуков. Файл hook-torch.py автоматически подгружает необходимые зависимости PyTorch. Однако, если вы используете специфические расширения (например, torchvision), может потребоваться создание пользовательского хука или использование флага \--additional-hooks-dir. Хук — это скрипт, который исполняется во время анализа и может добавлять файлы в datas и binaries.7

## ---

**Глава V: Техники Глубокой Оптимизации и Исключения**

После настройки окружения и базового .spec файла, наступает этап тонкой настройки для достижения минимального размера.

### **Хирургическое Удаление Подмодулей**

Пакет torch содержит множество подмодулей, которые могут быть не нужны для инференса конкретной модели. PyInstaller позволяет исключать целые пакеты через параметр excludes.  
Кандидаты на исключение:

- torch.testing: Модуль для тестирования, содержащий большие объемы данных.18
- torch.distributed: Если приложение работает на одной машине, распределенные вычисления не нужны.
- matplotlib, IPython, jupyter: Часто попадают в зависимости через другие библиотеки, но бесполезны в консольном приложении.24

Пример в .spec:

Python

excludes=\['torch.testing', 'torch.distributed', 'matplotlib', 'IPython', 'doctest'\]

### **Кейс: torch.distributions и Тестовые Наборы**

Исследования показывают, что модуль torch.distributions может занимать существенное место. Если ваша нейросеть детерминирована (не использует сэмплирование из распределений), этот модуль можно исключить.24  
Однако, здесь нужно быть осторожным: исключение torch.distributions может сломать загрузку сохраненных весов модели (torch.load), если при сохранении использовались объекты из этого модуля. Тестирование работоспособности после каждого исключения обязательно.

### **Проблема Исходных Кодов для TorchScript**

Как упоминалось ранее, JIT-компиляция требует наличия исходных кодов. Ошибка OSError: Can't get source блокирует работу приложения.  
Решение заключается в том, чтобы заставить PyInstaller собрать исходные .py файлы для критических модулей, а не только скомпилированные .pyc.  
Это можно сделать через функцию collect_all или вручную добавляя файлы в datas.

Python

from PyInstaller.utils.hooks import collect_all  
datas, binaries, hiddenimports \= collect_all('torch', include_py_files=True)

Использование include_py_files=True критично для JIT-совместимости.15 Это немного увеличивает размер, но необходимо для работоспособности моделей, использующих декораторы @torch.jit.script.

## ---

**Глава VI: Сжатие и Альтернативные Компиляторы**

Когда логическая оптимизация завершена, можно применить методы физического сжатия и рассмотреть альтернативные инструменты сборки.

### **UPX: Преимущества и Риски Стабильности**

UPX (Ultimate Packer for eXecutables) — это упаковщик исполняемых файлов. PyInstaller поддерживает интеграцию с UPX через флаг \--upx-dir.  
UPX сжимает бинарные секции исполняемого файла, распаковывая их в оперативную память при запуске.  
**Преимущества:**

- Уменьшение размера файла на 30-50%.1

**Риски и Недостатки:**

- **Медленный старт:** Распаковка требует времени CPU.
- **Коррупция DLL:** Сложные библиотеки, такие как torch и PyQt, часто имеют нетривиальную структуру, которую UPX может повредить, приводя к ошибкам "Segfault" или "DLL load failed".27
- **Ложные срабатывания антивирусов:** Упакованные UPX файлы часто помечаются эвристическими анализаторами как подозрительные.

**Рекомендация для Квеста:** Использовать UPX с осторожностью. Лучшая практика — исключить критические DLL из сжатия через параметр upx_exclude в .spec файле, если наблюдаются сбои. Для максимальной стабильности лучше отказаться от UPX в пользу логической оптимизации (удаления ненужных библиотек).

### **Nuitka: Трансляция в C++ против Упаковки**

Nuitka — это альтернатива PyInstaller, которая работает по другому принципу. Она транслирует Python-код в C++, а затем компилирует его в машинный код, линкуя с libpython.  
Сравнение размеров и производительности:  
Согласно бенчмаркам в исследовательских сниппетах, Nuitka может показывать противоречивые результаты по размеру:

- В некоторых случаях Nuitka создает файлы _меньшего_ размера (21.5 МБ против 3.58 МБ PyInstaller для простых скриптов — здесь PyInstaller выигрывает).
- Для сложных проектов с numpy/pandas/torch, Nuitka может создавать _большие_ файлы (900 МБ против 700 МБ у PyInstaller) из\-за статической линковки C-библиотек.29
- Однако, при использовании режима \--standalone и плагинов (например, nuitka \--standalone \--plugin-enable=torch), Nuitka может эффективнее отсекать неиспользуемый код, чем PyInstaller, сокращая размер дистрибутива с 250 МБ до 120 МБ в определенных сценариях.31

**Ключевое отличие:** Nuitka действительно _компилирует_ код, что дает прирост производительности (execution speed), тогда как PyInstaller просто _запускает_ его. Для задачи минимизации размера PyInstaller с правильно настроенным окружением (CPU-only) часто оказывается более предсказуемым и гибким инструментом, но Nuitka заслуживает внимания, если требуется защита кода (исходники превращаются в бинарник) и скорость исполнения.

## ---

**Глава VII: Диагностика и Разрешение Конфликтов**

Создание "Самодостаточного Артефакта" редко проходит без ошибок с первого раза. Умение диагностировать проблемы — часть квеста.

### **Отладка Загрузчика и Ошибки Импорта**

Если собранный файл падает при запуске:

1. **Консольный режим:** Всегда собирайте с флагом \--console (или console=True в spec), чтобы видеть traceback ошибки. Режим \--noconsole скрывает критическую информацию.28
2. **Verbose Imports:** Используйте флаг \--debug=imports при запуске собранного приложения (если поддерживается) или пересоберите его с отладочной информацией. Это покажет, какой модуль не удалось загрузить.
3. **Временные файлы:** Иногда проблема кроется в кеше PyInstaller. Команда pyinstaller \--clean обязательна перед финальной сборкой.18

### **Кросс-платформенные Нюансы и Драйверы**

Windows:  
Основная проблема — отсутствие DLL, которые не являются частью Python, но нужны библиотекам (например, vcruntime140.dll). PyInstaller обычно находит их, но иногда требуется установка "Visual C++ Redistributable" на целевой машине.  
Особая проблема: драйверы GPU. PyInstaller пакует CUDA Toolkit, но не драйвер видеокарты. Если на целевой машине старый драйвер NVIDIA, программа упадет с ошибкой Error 803: system has unsupported display driver.13 Это еще один аргумент в пользу CPU-версии для широкого распространения.  
Linux (Glibc):  
Сборка, сделанная на новой Ubuntu (с glibc 2.35), не запустится на старой CentOS (с glibc 2.17). PyInstaller не пакует glibc, так как это часть ядра системы.  
Решение: Собирать артефакт в Docker-контейнере с максимально старой версией Linux, которую вы планируете поддерживать (принцип "Holy Build Box").27

## ---

**Заключение и Итоговые Рекомендации**

Квест 24.2 по созданию "Самодостаточного Артефакта" на базе PyTorch требует перехода от мышления "разработчика скриптов" к мышлению "системного инженера". Проблема размера в 3 ГБ не является багом PyInstaller; это точное отражение сложности современных ML-фреймворков.  
**Итоговый алгоритм успеха:**

1. **Принцип Минимализма:** Всегда начинайте с создания чистого виртуального окружения (venv). Никогда не используйте глобальный Python.
2. **Выбор Архитектуры:** Если возможно, используйте **CPU-only** версию PyTorch (--index-url https://download.pytorch.org/whl/cpu). Это одно действие сокращает размер артефакта на \~90%.
3. **Контроль Сборки:** Используйте .spec файлы для явного исключения ненужных модулей (torch.testing, matplotlib) и фильтрации бинарных файлов.
4. **Совместимость с JIT:** Включайте исходные файлы (.py) для модулей, использующих TorchScript, чтобы избежать ошибок рантайма.
5. **Осторожность с Сжатием:** Используйте UPX только после тестирования стабильности и исключайте проблемные DLL.
6. **Тестирование:** Проверяйте артефакт на "чистой" виртуальной машине без установленного Python и драйверов CUDA, чтобы гарантировать истинную самодостаточность.

Следуя этому протоколу, можно превратить громоздкий проект в компактный, переносимый и профессиональный исполняемый файл, готовый к реальной эксплуатации.

#### **Источники**

1. Reduce pyinstaller executable size \- python \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/44681356/reduce-pyinstaller-executable-size](https://stackoverflow.com/questions/44681356/reduce-pyinstaller-executable-size)
2. How to reduce the size of the PyTorch package after using PyInstaller with \-D option, resulting in a 3GB size for the torch file?, дата последнего обращения: декабря 22, 2025, [https://discuss.pytorch.org/t/how-to-reduce-the-size-of-the-pytorch-package-after-using-pyinstaller-with-d-option-resulting-in-a-3gb-size-for-the-torch-file/194822](https://discuss.pytorch.org/t/how-to-reduce-the-size-of-the-pytorch-package-after-using-pyinstaller-with-d-option-resulting-in-a-3gb-size-for-the-torch-file/194822)
3. Reducing the Size of PyTorch-Based Executable · Issue \#8551 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/pyinstaller/pyinstaller/issues/8551](https://github.com/pyinstaller/pyinstaller/issues/8551)
4. Using PyInstaller, дата последнего обращения: декабря 22, 2025, [https://pyinstaller.org/en/v6.1.0/usage.html](https://pyinstaller.org/en/v6.1.0/usage.html)
5. Exe files created using pyinstaller are always larger than I expected, what am I doing wrong? \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/learnpython/comments/16kxnga/exe_files_created_using_pyinstaller_are_always/](https://www.reddit.com/r/learnpython/comments/16kxnga/exe_files_created_using_pyinstaller_are_always/)
6. How big is too big for a Pyinstaller exe file? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/63138311/how-big-is-too-big-for-a-pyinstaller-exe-file](https://stackoverflow.com/questions/63138311/how-big-is-too-big-for-a-pyinstaller-exe-file)
7. What PyInstaller Does and How It Does It, дата последнего обращения: декабря 22, 2025, [https://pyinstaller.org/en/stable/operating-mode.html](https://pyinstaller.org/en/stable/operating-mode.html)
8. Does PyInstaller include CUDA \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/59074416/does-pyinstaller-include-cuda](https://stackoverflow.com/questions/59074416/does-pyinstaller-include-cuda)
9. Understanding PyInstaller Hooks, дата последнего обращения: декабря 22, 2025, [https://pyinstaller.org/en/stable/hooks.html](https://pyinstaller.org/en/stable/hooks.html)
10. When Things Go Wrong — PyInstaller 6.17.0 documentation, дата последнего обращения: декабря 22, 2025, [https://pyinstaller.org/en/stable/when-things-go-wrong.html](https://pyinstaller.org/en/stable/when-things-go-wrong.html)
11. Pyinstaller automatically includes unneeded modules · Issue \#2652 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/pyinstaller/pyinstaller/issues/2652](https://github.com/pyinstaller/pyinstaller/issues/2652)
12. size of executable using pyinstaller and numpy \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/47769904/size-of-executable-using-pyinstaller-and-numpy](https://stackoverflow.com/questions/47769904/size-of-executable-using-pyinstaller-and-numpy)
13. How to deploy a pytorch detector built by pyinstaller in ubuntu to another host with cuda of different version, дата последнего обращения: декабря 22, 2025, [https://discuss.pytorch.org/t/how-to-deploy-a-pytorch-detector-built-by-pyinstaller-in-ubuntu-to-another-host-with-cuda-of-different-version/192842](https://discuss.pytorch.org/t/how-to-deploy-a-pytorch-detector-built-by-pyinstaller-in-ubuntu-to-another-host-with-cuda-of-different-version/192842)
14. PyInstaller, issue to build a package with CUDA, CuDNN and Tensorflow on Windows 10 \#7175 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/pyinstaller/pyinstaller/issues/7175](https://github.com/pyinstaller/pyinstaller/issues/7175)
15. Can't run the pytorch lightning program packaged with pyinstaller. \#7918 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/pyinstaller/pyinstaller/issues/7918](https://github.com/pyinstaller/pyinstaller/issues/7918)
16. Issue with torch.git source in pyinstaller \- PyTorch Forums, дата последнего обращения: декабря 22, 2025, [https://discuss.pytorch.org/t/issue-with-torch-git-source-in-pyinstaller/135446](https://discuss.pytorch.org/t/issue-with-torch-git-source-in-pyinstaller/135446)
17. Reducing size of pyinstaller exe \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/47692213/reducing-size-of-pyinstaller-exe](https://stackoverflow.com/questions/47692213/reducing-size-of-pyinstaller-exe)
18. python \- pyinstaller \- Excluding Modules \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/72341228/pyinstaller-excluding-modules](https://stackoverflow.com/questions/72341228/pyinstaller-excluding-modules)
19. Get Started \- PyTorch, дата последнего обращения: декабря 22, 2025, [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
20. Where do I get a CPU-only version of PyTorch? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/51730880/where-do-i-get-a-cpu-only-version-of-pytorch](https://stackoverflow.com/questions/51730880/where-do-i-get-a-cpu-only-version-of-pytorch)
21. Installing a CPU-Only Version of PyTorch \- GeeksforGeeks, дата последнего обращения: декабря 22, 2025, [https://www.geeksforgeeks.org/deep-learning/installing-a-cpu-only-version-of-pytorch/](https://www.geeksforgeeks.org/deep-learning/installing-a-cpu-only-version-of-pytorch/)
22. Installing a specific PyTorch build (f/e CPU-only) with Poetry \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/59158044/installing-a-specific-pytorch-build-f-e-cpu-only-with-poetry](https://stackoverflow.com/questions/59158044/installing-a-specific-pytorch-build-f-e-cpu-only-with-poetry)
23. How to remove/exclude modules and files from pyInstaller? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/17034434/how-to-remove-exclude-modules-and-files-from-pyinstaller](https://stackoverflow.com/questions/17034434/how-to-remove-exclude-modules-and-files-from-pyinstaller)
24. Python: Excluding Modules Pyinstaller \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/4890159/python-excluding-modules-pyinstaller](https://stackoverflow.com/questions/4890159/python-excluding-modules-pyinstaller)
25. Problem exporting gpytorch (jit) · Issue \#7647 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/pyinstaller/pyinstaller/issues/7647](https://github.com/pyinstaller/pyinstaller/issues/7647)
26. PyInstaller executable fails to get source code of TorchScript \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/61756222/pyinstaller-executable-fails-to-get-source-code-of-torchscript](https://stackoverflow.com/questions/61756222/pyinstaller-executable-fails-to-get-source-code-of-torchscript)
27. 169 INFO: UPX is available but is disabled on non-Windows due to known compatibility problems. · pyinstaller · Discussion \#8922 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/orgs/pyinstaller/discussions/8922](https://github.com/orgs/pyinstaller/discussions/8922)
28. How do I use UPX with pyinstaller? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/47730240/how-do-i-use-upx-with-pyinstaller](https://stackoverflow.com/questions/47730240/how-do-i-use-upx-with-pyinstaller)
29. The size of the .exe relative to the pyinstaller is much larger · Issue \#926 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/Nuitka/Nuitka/issues/926](https://github.com/Nuitka/Nuitka/issues/926)
30. How do I reduce Python executable sizes? : r/learnpython \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/learnpython/comments/1ayhxrs/how_do_i_reduce_python_executable_sizes/](https://www.reddit.com/r/learnpython/comments/1ayhxrs/how_do_i_reduce_python_executable_sizes/)
31. Compile time, created files size, and exec time all seem large (Win10/mingw64 in conda) · Issue \#599 · Nuitka/Nuitka \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/Nuitka/Nuitka/issues/599](https://github.com/Nuitka/Nuitka/issues/599)
