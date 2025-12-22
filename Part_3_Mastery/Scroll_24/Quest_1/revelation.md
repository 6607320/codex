# **Отчет об исследовании: Архитектура и реализация дистрибуции Python-пакетов (Квест "Магический Свиток")**

## **Введение: Метафора "Магического Свитка" в инженерном контексте**

В современной экосистеме разработки программного обеспечения на языке Python процесс упаковки кода представляет собой фундаментальный этап, трансформирующий локальные разработки в отчуждаемые, воспроизводимые артефакты. Пользовательский запрос, сформулированный как "Квест 24.1: Создание 'Магического Свитка'", аллегорически описывает процедуру сборки проекта в пакет формата Wheel (.whl) с использованием конфигурационного сценария setup.py. Этот "свиток" — бинарный дистрибутив — является стандартом де\-факто для распространения библиотек и приложений, обеспечивая их быструю инсталляцию и детерминированное поведение в целевых средах.  
Данный отчет представляет собой исчерпывающее руководство по выполнению этого квеста. Мы не просто рассмотрим последовательность команд, но и проведем глубокий анализ архитектурных решений, лежащих в основе setuptools, разберем анатомию формата Wheel, исследуем нюансы управления зависимостями (включая сложные случаи с PyTorch) и определим стратегии валидации собранных артефактов. Несмотря на то, что современный Python движется в сторону декларативной конфигурации через pyproject.toml (PEP 517), понимание работы setup.py остается критически важным навыком, так как он лежит в основе миллионов существующих проектов и предоставляет гибкость, недоступную статическим файлам.1

## **Глава 1\. Теоретические основы дистрибуции в Python**

Прежде чем приступать к написанию кода конфигурации, необходимо понять фундаментальное различие между типами дистрибутивов, существующими в экосистеме Python Packaging Authority (PyPA). Это различие определяет не только способ установки, но и философию распространения кода.

### **1.1. Дихотомия: Исходный код (sdist) против Бинарной сборки (Wheel)**

В основе упаковки Python лежат два основных формата: дистрибутив исходного кода (_Source Distribution_ или sdist) и собранный дистрибутив (_Built Distribution_ или wheel).  
Дистрибутив исходного кода (sdist)  
Традиционно представляемый в виде архива .tar.gz, sdist содержит "сырой" исходный код, сценарий setup.py и сопутствующие файлы манифеста. Это, по сути, рецепт для создания пакета.3 Когда пользователь устанавливает пакет из sdist (например, через pip install package.tar.gz), на его локальной машине запускается процесс сборки. Это означает, что интерпретатор Python выполняет setup.py на стороне клиента.

- **Преимущества:** Максимальная переносимость, так как код может быть скомпилирован под любую архитектуру непосредственно на целевой машине. Это также предпочтительный формат для дистрибьюторов Linux (например, Gentoo), так как структура архива повторяет репозиторий, облегчая наложение патчей.5
- **Недостатки:** Требует наличия инструментов сборки (компиляторов C/C++, заголовочных файлов) на машине пользователя. Инсталляция занимает больше времени, так как включает этап компиляции. Кроме того, выполнение setup.py на стороне клиента подразумевает выполнение произвольного кода, что несет потенциальные риски безопасности.6

Собранный дистрибутив (wheel) — "Магический Свиток"  
Формат Wheel (файлы с расширением .whl) представляет собой ZIP-архив с особо сформированным именем, содержащий файлы, готовые к распаковке в директорию site-packages. Это и есть "Магический Свиток" из квеста — артефакт, который не требует компиляции при установке.7

- **Преимущества:** Скорость установки возрастает на порядки, так как процесс сводится к копированию файлов. Отсутствует необходимость в компиляторах на стороне пользователя. Формат Wheel предкомпилирован: для чистых Python-пакетов это означает отсутствие выполнения setup.py при установке, а для пакетов с расширениями C/C++ — наличие скомпилированных бинарных модулей (.so или .dll).3
- **Стандартизация:** В отличие от устаревшего формата Egg, Wheel стандартизирован (PEP 427), имеет версионирование спецификации и четкую структуру метаданных, хранящихся в директории .dist-info.3

### **1.2. Эволюция инструментов: От distutils к setuptools и build**

Исторически сборка осуществлялась через стандартную библиотеку distutils. Однако ее возможностей оказалось недостаточно для современной экосистемы, что привело к появлению setuptools — расширенного набора инструментов, который де\-факто стал стандартом. Важно отметить, что многие старые руководства предлагают запускать команды вида python setup.py bdist_wheel. Согласно современным стандартам (PEP 517), прямое выполнение setup.py как скрипта командной строки является устаревшим (_deprecated_).1  
Современный подход предполагает разделение "фронтенда" сборки (инструмента, который инициирует процесс) и "бэкенда" (библиотеки, которая выполняет сборку). setuptools выступает в роли бэкенда. В качестве фронтенда рекомендуется использовать утилиту build (python \-m build), которая создает изолированное виртуальное окружение, устанавливает необходимые зависимости сборки и вызывает бэкенд для генерации артефактов.10 Тем не менее, файл setup.py остается центральным местом конфигурации проекта.

## **Глава 2\. Архитектура проекта и файловая структура**

Корректная настройка setup.py начинается задолго до написания первой строчки кода конфигурации. Она начинается с правильной организации файлов в проекте. Ошибки на этом этапе приводят к созданию "пустых" пакетов или пакетов, которые работают в тестах, но ломаются в продакшене.

### **2.1. Сравнение подходов: src-layout против Flat-layout**

Существует две доминирующие парадигмы организации структуры проекта: плоская структура (_Flat-layout_) и структура с директорией src (_Src-layout_).  
Flat-layout (Плоская структура)  
При таком подходе пакет находится в корневой директории репозитория:

project_root/  
├── setup.py  
├── mypackage/  
│ ├── \_\_init\_\_.py  
│ └── module.py

Хотя это кажется интуитивным, такая структура страдает от проблемы "паритета импорта". Когда вы запускаете тесты из корневой директории (project_root), Python добавляет текущую директорию в sys.path. В результате import mypackage импортирует локальную папку с исходным кодом, а не установленный пакет. Это маскирует ошибки упаковки: вы можете забыть включить файл данных в пакет, но тесты пройдут успешно, так как они видят локальный файл. В продакшене же код упадет с FileNotFoundError.12  
Src-layout (Рекомендуемая структура)  
Этот подход изолирует исходный код в поддиректорию src:

project_root/  
├── setup.py  
├── src/  
│ └── mypackage/  
│ ├── \_\_init\_\_.py  
│ └── module.py

В этой конфигурации корневая директория не содержит пакета mypackage напрямую. Чтобы запустить тесты или использовать код, вы _обязаны_ установить пакет (обычно в режиме редактирования: pip install \-e.). Это гарантирует, что тесты проверяют именно то, что будет установлено у пользователя. Использование src-layout требует явной настройки в setup.py, указывая setuptools, что пакеты следует искать внутри src.13

### **2.2. Роль \_\_init\_\_.py и механизмы обнаружения**

Файл \_\_init\_\_.py является маркером, превращающим обычную директорию в пакет Python. Несмотря на то, что Python 3.3+ поддерживает _namespace packages_ (пакеты пространства имен), которые могут работать без \_\_init\_\_.py, для корректной работы инструментов упаковки setuptools, в частности функции find_packages(), наличие этого файла строго рекомендуется.  
Без \_\_init\_\_.py функция find_packages() может проигнорировать директорию, что приведет к созданию колеса, которое не содержит кода. Хотя setuptools предоставляет альтернативу find_namespace_packages(), классические пакеты должны содержать инициализатор, даже если он пустой.15 В \_\_init\_\_.py также часто выносят версию пакета или экспортируют основные классы для упрощения API.

## **Глава 3\. Манифест Конфигурации: Глубокое погружение в setup.py**

Файл setup.py — это исполняемый скрипт Python, использующий функцию setup() из библиотеки setuptools. Рассмотрим детально ключевые аргументы этой функции, необходимые для выполнения квеста.

### **3.1. Базовые метаданные и идентификация**

Первый блок аргументов определяет идентичность пакета в глобальном индексе PyPI.

Python

from setuptools import setup, find_packages

setup(  
 name="magic-scroll-project", \# Уникальное имя (проверяйте на PyPI)  
 version="1.0.0", \# Версия согласно PEP 440  
 author="Code Wizard",  
 description="Библиотека для создания магических артефактов",  
 \#...  
)

Версионирование — критический аспект. Использование схемы семантического версионирования (Major.Minor.Patch) позволяет инструментам управления зависимостями корректно разрешать совместимость.

### **3.2. Стратегии обнаружения пакетов (packages и package_dir)**

Один из самых частых вопросов при настройке — как заставить setup.py видеть код. Здесь проявляется преимущество src-layout.  
Использование find_packages():  
Вместо ручного перечисления всех пакетов (что чревато ошибками), используется функция автоматического поиска.

Python

setup(  
 \#...  
 packages=find_packages(where="src", include=\["mypackage\*"\]),  
 package_dir={"": "src"},  
)

- **where="src"**: Указывает setuptools начать сканирование не в корне, а в директории src.
- **package_dir={"": "src"}**: Это критически важный словарь. Пустая строка "" обозначает корень пространства имен пакетов. Запись говорит: "Корневые пакеты находятся в директории src". Без этой строки, даже если find_packages найдет пакеты, Python не сможет правильно сформировать пути при установке.13
- **include и exclude**: Позволяют фильтровать найденное. Например, exclude=\["tests\*", "docs\*"\] предотвращает попадание тестового кода и документации в финальный дистрибутив, что уменьшает размер колеса и очищает пространство имен пользователя.17

Если использовать ручной список packages=\["mypackage"\] без package_dir, setuptools будет искать директорию mypackage в корне проекта, что не сработает для src-layout.14

### **3.3. Ограничения среды (python_requires)**

Современная практика требует явного указания поддерживаемых версий Python. Это предотвращает установку пакета на несовместимые интерпретаторы (например, попытку установить пакет с f-строками на Python 3.5).

Python

python_requires='\>=3.8, \<4',

Этот аргумент транслируется в метаданные дистрибутива, которые анализируются pip до загрузки пакета. Существует дискуссия относительно верхней границы (\<4). Некоторые эксперты (включая Пола Мура) утверждают, что ограничение \<4 излишне, так как переход на Python 4.0 не планируется как ломающий (в отличие от 2-\>3). Однако, для строгости и гарантии работы в проверенных условиях, многие проекты оставляют это ограничение.18

### **3.4. Включение не-кодовых файлов (package_data)**

Часто "Магический Свиток" должен содержать не только заклинания (код), но и артефакты (данные, шаблоны, конфиги). Для этого используется аргумент package_data или манифест MANIFEST.in с аргументом include_package_data=True.

Python

package_data={  
 "mypackage": \["\*.json", "data/\*.dat"\],  
},

Этот словарь мапит имя пакета на список глоб-паттернов файлов, которые нужно включить внутрь колеса. Важно: эти файлы должны находиться _внутри_ директории пакета.17

## **Глава 4\. Управление зависимостями: Лабиринт совместимости**

Правильное определение зависимостей — это, пожалуй, самая сложная часть квеста. Ошибки здесь приводят к "Dependency Hell".

### **4.1. install_requires против requirements.txt: Абстракция и Конкретика**

В материалах исследования четко прослеживается различие между этими двумя механизмами, которое часто игнорируется новичками.20

| Характеристика       | install_requires (setup.py)                                   | requirements.txt                                                    |
| :------------------- | :------------------------------------------------------------ | :------------------------------------------------------------------ |
| **Тип требований**   | **Абстрактные**                                               | **Конкретные**                                                      |
| **Цель**             | Определение _диапазона_ совместимости библиотеки.             | Определение _точной среды_ для воспроизводимости приложения.        |
| **Синтаксис версий** | Слабые ограничения (напр., \>=1.0, \!=1.2).                   | Жесткие пины (напр., \==1.0.4).                                     |
| **Обработка pip**    | Анализируется резолвером зависимостей для поиска пересечений. | Устанавливается линейно (хотя современные версии pip улучшили это). |
| **Использование**    | Для библиотек, публикуемых на PyPI.                           | Для деплоя конечных приложений (SaaS, скрипты).                     |

**Best Practice:** В setup.py вы должны указывать _минимально необходимые_ требования. Никогда не фиксируйте версии (==) в install_requires, если только это не критично, так как это блокирует пользователям возможность обновить зависимость при выходе патчей безопасности.22

Python

install_requires=\[  
 "requests\>=2.25.0",  
 "numpy\>=1.19.0; python_version \>= '3.8'", \# Условная зависимость  
\],

### **4.2. Кейс PyTorch: Проблема платформозависимых колес**

Особую сложность вызывает работа с библиотеками типа PyTorch (torch), которые имеют разные бинарные сборки для разных аппаратных платформ (CPU, CUDA 11.x, CUDA 12.x). Стандартный PyPI обычно хостит колеса с поддержкой CUDA, которые имеют огромный размер, либо pip по умолчанию тянет версию, не подходящую под железо пользователя.  
Исследование показывает, что попытка импортировать torch внутри setup.py для компиляции расширений — это антипаттерн, приводящий к ошибкам установки, так как torch может быть еще не установлен в системе.24  
**Решение проблемы:**

1. **Не указывать специфику в install_requires:** Не пишите torch==1.9.0+cpu. PyPI не поддерживает локальные версификаторы (с плюсом) корректно при разрешении зависимостей общего назначения. Указывайте просто torch\>=1.9.0.26
2. **Отделить установку:** Позвольте пользователю самому выбрать правильную версию torch (через \--index-url официального репозитория PyTorch) _до_ установки вашего пакета, либо позвольте pip установить дефолтную версию. Автоматический выбор между CPU и GPU версиями на уровне setup.py практически невозможен средствами стандартного pip без использования сложных хаков или сторонних инструментов типа light-the-torch.27
3. **Build-system requirements:** Если torch нужен для _сборки_ пакета (например, для C++ расширений), его необходимо указать в секции \[build-system\] файла pyproject.toml, чтобы утилита сборки могла подтянуть его в изолированное окружение.2

## **Глава 5\. Процесс сборки: Ковка Свитка**

Когда конфигурация готова, наступает этап генерации артефакта. Как упоминалось, метод прямого вызова setup.py устарел.

### **5.1. Устаревшая команда bdist_wheel**

Команда python setup.py bdist_wheel долгое время была стандартом. Она поручала setuptools (с подключенным расширением wheel) собрать пакет.

- **Частая ошибка:** "invalid command 'bdist_wheel'". Это происходит, если в окружении не установлен пакет wheel. distutils (стандартная библиотека) не знает о существовании формата wheel, это функциональность стороннего пакета.30
- **Решение:** Необходимо установить pip install wheel.

### **5.2. Современный стандарт: python \-m build**

Согласно PEP 517, процесс сборки должен быть изолирован. Рекомендуемый алгоритм выполнения квеста следующий:

1. Убедитесь, что установлен пакет build:  
   Bash  
   pip install build

2. Запустите сборку из корня проекта:  
   Bash  
   python \-m build

Эта команда выполнит следующие действия:

- Создаст изолированное виртуальное окружение.
- Установит туда зависимости сборки (указанные в pyproject.toml или подразумеваемые setuptools и wheel).
- Сгенерирует sdist (архив .tar.gz).
- Сгенерирует wheel (файл .whl).  
  Оба файла появятся в директории dist/.1

### **5.3. Универсальные и Платформенные колеса**

Если ваш пакет написан на чистом Python и совместим как с Python 2, так и с Python 3 (что сейчас редкость, но теоретически возможно), вы можете собрать "Универсальное колесо" (universal wheel).  
В setup.cfg:

Ini, TOML

\[bdist_wheel\]  
universal \= 1

Это создаст файл с тегом py2.py3-none-any. Однако, если пакет содержит C-расширения, колесо будет привязано к архитектуре (например, linux_x86_64) и версии Python (ABI tag, например, cp39), что делает его "Платформенным колесом".4

## **Глава 6\. Анатомия Магического Свитка (Wheel)**

Понимание внутренней структуры файла .whl необходимо для верификации сборки. Wheel — это ZIP-архив, но с жестко заданной структурой, описанной в PEP 427\.

### **6.1. Структура архива**

Если распаковать файл .whl (используя unzip или tar), мы увидим следующую структуру 7:

| Путь в архиве            | Назначение                                                                                                       |
| :----------------------- | :--------------------------------------------------------------------------------------------------------------- |
| mypackage/               | Директория с кодом пакета (результат работы find_packages).                                                      |
| mypackage-1.0.dist-info/ | Директория с метаданными. Именно наличие этой папки отличает wheel от простого zip-архива.                       |
| mypackage-1.0.data/      | (Опционально) Директория данных, содержимое которой распаковывается в системные пути (например, /usr/local/bin). |

### **6.2. Директория .dist-info**

Это "паспорт" пакета. В ней содержатся:

- **METADATA**: Файл в формате RFC-822, содержащий описание, автора, классификаторы и зависимости (Requires-Dist). Этот файл формируется на основе аргументов setup().33
- **WHEEL**: Техническая информация о самом архиве (версия спецификации Wheel, генератор, теги совместимости).
- **RECORD**: Криптографический реестр. Это CSV-файл, в котором перечислены _все_ файлы архива, их размеры и SHA-256 хеши.
  - _Механизм защиты:_ При установке pip сверяет хеши файлов с записями в RECORD. Это гарантирует целостность "свитка" — что ни один байт кода не был поврежден или подменен при передаче.8
- **entry_points.txt**: Если в setup.py были указаны entry_points (например, для создания консольных утилит console_scripts), они записываются сюда. Инсталлятор использует этот файл для генерации исполняемых файлов-оберток (shims) в bin директории пользователя.7

### **6.3. Логика инсталляции**

Важно понимать: установка Wheel — это не выполнение скрипта, а операция распаковки и копирования ("Spread").

1. **Unpack:** Архив распаковывается во временную директорию.
2. **Verify:** Проверяется совместимость тегов имени файла с системой и валидность хешей RECORD.
3. **Spread:** Файлы переносятся в site-packages (для библиотек) или в соответствующие системные директории (для скриптов и заголовков).
4. Compile: (Опционально) Создаются файлы байткода .pyc.  
   Этот процесс намного безопаснее и предсказуемее, чем запуск произвольного кода из setup.py install.6

## **Глава 7\. Валидация и Тестирование Сборки**

Создание колеса — это только половина квеста. Вторая половина — убедиться, что оно работает. Тестирование должно проводиться в "чистой комнате".

### **7.1. Инспекция содержимого**

Перед установкой рекомендуется заглянуть внутрь архива без распаковки.

Bash

\# Просмотр списка файлов  
tar \-tf dist/mypackage-1.0.0-py3-none-any.whl

Ищите следующие аномалии:

- Наличие папок tests/ или docs/ в корне (если вы не хотели их паковать).
- Наличие файлов \_\_pycache\_\_ или .pyc (они не должны быть в wheel).
- Отсутствие ожидаемых файлов данных.

Существует специализированный инструмент check-wheel-contents, который автоматизирует эти проверки. Он может предупредить о распространенных ошибках, таких как пустые директории или дублирование библиотек.35 Также полезен инструмент wheel-inspect для вывода метаданных в формате JSON.36

### **7.2. Установка в изолированном окружении**

**Критическая ошибка:** Тестирование пакета в той же директории, где лежит исходный код. Из-за того, как Python работает с путями, import mypackage может подхватить локальную папку src/mypackage, а не установленный wheel.  
**Правильный алгоритм тестирования:**

1. Создайте новое виртуальное окружение (venv или conda):  
   Bash  
   python \-m venv test_env  
   source test_env/bin/activate \# или test_env\\Scripts\\activate на Windows

.37  
2\. Перейдите в нейтральную директорию (например, cd /tmp или просто выйдите из корня проекта).  
3\. Установите собранный wheel файл напрямую. Синтаксис pip позволяет указывать путь к файлу:  
bash pip install /path/to/project/dist/mypackage-1.0.0-py3-none-any.whl  
.39  
4\. Запустите проверку импорта:  
bash python \-c "import mypackage; print(mypackage.\_\_version\_\_)"  
5\. Проверьте дерево зависимостей, чтобы убедиться, что install_requires отработал корректно:  
bash pip install pipdeptree pipdeptree \-p mypackage  
.41  
Если пакет установился, зависимости подтянулись, и импорт прошел успешно — квест можно считать выполненным.

## **Глава 8\. Диагностика проблем и Troubleshooting**

Даже опытные маги совершают ошибки. Разберем типичные проблемы при сборке.

### **8.1. Ошибка "invalid command 'bdist_wheel'"**

Симптом: При запуске старой команды python setup.py bdist_wheel процесс падает.  
Причина: Команда bdist_wheel не встроена в Python, она является частью пакета wheel. В некоторых минимальных средах (например, в CI/CD контейнерах Ubuntu) этот пакет может отсутствовать.30  
Решение:

- Локально: pip install wheel.
- В CI: Добавить явный шаг установки pip install \--upgrade pip setuptools wheel перед сборкой.
- Лучшее решение: Использовать python \-m build, который автоматически управляет зависимостями сборки.42

### **8.2. Ошибка "Source as Package" (Пакет src)**

Симптом: После установки пакета импорт mypackage не работает, но работает import src. Или в wheel-файле находится папка src.  
Причина: Неправильная конфигурация find_packages() при использовании src-layout. setuptools по умолчанию считает корневой директорией ту, где лежит setup.py. Если не указать перенаправление, он найдет папку src и подумает, что это и есть пакет.  
Решение: Убедитесь в наличии аргумента package_dir={'': 'src'} в функции setup().13

### **8.3. Проблемы с файлами данных (Manifest Hell)**

Симптом: Код падает с ошибкой FileNotFoundError при попытке открыть конфигурационный файл, который должен быть внутри пакета.  
Причина: Файл не попал в wheel. Это может произойти, если файл не является .py файлом и не указан в package_data или MANIFEST.in.  
Решение:

1. Проверьте include_package_data=True в setup.py.
2. Создайте файл MANIFEST.in и добавьте строку include src/mypackage/data/\*.json.
3. Помните, что pip версий до 20.x не всегда корректно обрабатывал MANIFEST.in для бинарных колес, полагаясь больше на package_data.17

## **Заключение**

Выполнение Квеста 24.1 по созданию "Магического Свитка" требует не только знания синтаксиса Python, но и понимания инженерных стандартов дистрибуции. Переход от простого скрипта к профессиональному пакету включает в себя принятие архитектурных решений (src-layout), грамотное управление метаданными и зависимостями, а также строгую дисциплину сборки и тестирования.  
Хотя экосистема Python активно эволюционирует, внедряя новые стандарты вроде pyproject.toml, механизмы, описанные в этом отчете — структура Wheel, хеширование RECORD, разделение абстрактных и конкретных зависимостей — остаются фундаментальной физикой, на которой держится вся инфраструктура PyPI. Следуя изложенным рекомендациям, разработчик гарантирует, что созданный им артефакт будет надежно работать в любой точке мира, куда бы ни был доставлен этот "свиток".

### **Чек-лист выполнения квеста**

1. **Структура:** Привести проект к виду src/mypackage.
2. **Конфигурация:** Создать setup.py с использованием find_packages(where='src') и package_dir={'': 'src'}.
3. **Зависимости:** Перенести библиотеки из requirements.txt в install_requires (убрав жесткие версии).
4. **Сборка:** Выполнить pip install build && python \-m build.
5. **Валидация:** Проверить содержимое через tar \-tf и установить .whl в чистое виртуальное окружение для теста импорта.

## ---

**Приложение: Справочные таблицы**

### **Таблица 1\. Сравнение методов сборки**

| Метод      | Команда                     | Статус          | Примечание                                                                           |
| :--------- | :-------------------------- | :-------------- | :----------------------------------------------------------------------------------- |
| **Legacy** | python setup.py bdist_wheel | **Deprecated**  | Требует установленных wheel и setuptools в текущем окружении. Не изолирован.         |
| **Modern** | python \-m build            | **Recommended** | Создает изолированное окружение, устанавливает build-deps, генерирует sdist и wheel. |
| **Direct** | pip wheel.                  | **Alternative** | Удобно для создания колес зависимостей, но build предпочтительнее для самого пакета. |

### **Таблица 2\. Основные аргументы setup()**

| Аргумент         | Тип  | Назначение                                  | Пример                                       |
| :--------------- | :--- | :------------------------------------------ | :------------------------------------------- |
| name             | str  | Имя пакета на PyPI (должно быть уникальным) | "my-magic-scroll"                            |
| version          | str  | Версия пакета (PEP 440\)                    | "1.0.0"                                      |
| packages         | list | Список пакетов для включения                | find_packages(where="src")                   |
| package_dir      | dict | Маппинг директорий                          | {"": "src"}                                  |
| install_requires | list | Зависимости времени выполнения              | \["requests\>=2.0"\]                         |
| python_requires  | str  | Поддерживаемые версии Python                | "\>=3.7"                                     |
| entry_points     | dict | Создание CLI-команд                         | {"console_scripts": \["cast=pkg.main:run"\]} |

#### **Источники**

1. Is setup.py deprecated? \- Python Packaging User Guide, дата последнего обращения: декабря 22, 2025, [https://packaging.python.org/en/latest/discussions/setup-py-deprecated/](https://packaging.python.org/en/latest/discussions/setup-py-deprecated/)
2. How to modernize a setup.py based project? \- Python Packaging User Guide, дата последнего обращения: декабря 22, 2025, [https://packaging.python.org/en/latest/guides/modernize-setup-py-project/](https://packaging.python.org/en/latest/guides/modernize-setup-py-project/)
3. Package Formats \- Python Packaging User Guide, дата последнего обращения: декабря 22, 2025, [https://packaging.python.org/en/latest/discussions/package-formats/](https://packaging.python.org/en/latest/discussions/package-formats/)
4. What is the difference between an 'sdist' .tar.gz distribution and an python egg?, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/6292652/what-is-the-difference-between-an-sdist-tar-gz-distribution-and-an-python-egg](https://stackoverflow.com/questions/6292652/what-is-the-difference-between-an-sdist-tar-gz-distribution-and-an-python-egg)
5. Sdists for pure-Python projects \- Packaging \- Discussions on Python.org, дата последнего обращения: декабря 22, 2025, [https://discuss.python.org/t/sdists-for-pure-python-projects/25191](https://discuss.python.org/t/sdists-for-pure-python-projects/25191)
6. The Story of Wheel — wheel 0.46.1 documentation \- Read the Docs, дата последнего обращения: декабря 22, 2025, [https://wheel.readthedocs.io/en/latest/story.html](https://wheel.readthedocs.io/en/latest/story.html)
7. Binary distribution format \- Python Packaging User Guide, дата последнего обращения: декабря 22, 2025, [https://packaging.python.org/specifications/binary-distribution-format/](https://packaging.python.org/specifications/binary-distribution-format/)
8. PEP 427 – The Wheel Binary Package Format 1.0 \- Python Enhancement Proposals, дата последнего обращения: декабря 22, 2025, [https://peps.python.org/pep-0427/](https://peps.python.org/pep-0427/)
9. Deprecate and remove the \`setup.py bdist_wheel\` code path. Aka always use PEP 517\. · Issue \#13314 · pypa/pip \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/pypa/pip/issues/13314](https://github.com/pypa/pip/issues/13314)
10. 'setup.py install is deprecated' warning shows up every time I open a terminal in VSCode, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/73257839/setup-py-install-is-deprecated-warning-shows-up-every-time-i-open-a-terminal-i](https://stackoverflow.com/questions/73257839/setup-py-install-is-deprecated-warning-shows-up-every-time-i-open-a-terminal-i)
11. Quickstart \- setuptools 80.9.0 documentation, дата последнего обращения: декабря 22, 2025, [https://setuptools.pypa.io/en/latest/userguide/quickstart.html](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)
12. A Practical Guide To Using Setup.py \- Xebia, дата последнего обращения: декабря 22, 2025, [https://xebia.com/blog/a-practical-guide-to-using-setup-py/](https://xebia.com/blog/a-practical-guide-to-using-setup-py/)
13. python \- What is "where" argument for in setuptools.find_packages? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/51286928/what-is-where-argument-for-in-setuptools-find-packages](https://stackoverflow.com/questions/51286928/what-is-where-argument-for-in-setuptools-find-packages)
14. Python setup.py: How to get find_packages() to identify packages in subdirectories, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/54430694/python-setup-py-how-to-get-find-packages-to-identify-packages-in-subdirectori](https://stackoverflow.com/questions/54430694/python-setup-py-how-to-get-find-packages-to-identify-packages-in-subdirectori)
15. Does setuptools' find_packages require \_\_init\_\_.py files or not to recognize packages, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/61503299/does-setuptools-find-packages-require-init-py-files-or-not-to-recognize-pac](https://stackoverflow.com/questions/61503299/does-setuptools-find-packages-require-init-py-files-or-not-to-recognize-pac)
16. Packaging namespace packages \- Python Packaging User Guide, дата последнего обращения: декабря 22, 2025, [https://packaging.python.org/guides/packaging-namespace-packages/](https://packaging.python.org/guides/packaging-namespace-packages/)
17. Packaging and distributing projects \- Python Packaging User Guide, дата последнего обращения: декабря 22, 2025, [https://packaging.python.org/guides/distributing-packages-using-setuptools/](https://packaging.python.org/guides/distributing-packages-using-setuptools/)
18. Use of "less-than next-major-version" (e.g., \`\<4\`) in \`python_requires\` (setup.py) \- Packaging, дата последнего обращения: декабря 22, 2025, [https://discuss.python.org/t/use-of-less-than-next-major-version-e-g-4-in-python-requires-setup-py/1066](https://discuss.python.org/t/use-of-less-than-next-major-version-e-g-4-in-python-requires-setup-py/1066)
19. Enforcing python version in setup.py \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py](https://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py)
20. python \- requirements.txt vs setup.py \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/43658870/requirements-txt-vs-setup-py](https://stackoverflow.com/questions/43658870/requirements-txt-vs-setup-py)
21. requirements.txt vs setup.py in Python \- Towards Data Science, дата последнего обращения: декабря 22, 2025, [https://towardsdatascience.com/requirements-vs-setuptools-python-ae3ee66e28af/](https://towardsdatascience.com/requirements-vs-setuptools-python-ae3ee66e28af/)
22. install_requires vs requirements files \- Python Packaging User Guide, дата последнего обращения: декабря 22, 2025, [https://packaging.python.org/discussions/install-requires-vs-requirements/](https://packaging.python.org/discussions/install-requires-vs-requirements/)
23. Python setup config install_requires "good practices" \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/65326080/python-setup-config-install-requires-good-practices](https://stackoverflow.com/questions/65326080/python-setup-config-install-requires-good-practices)
24. Python Packaging Best Practices \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@miqui.ferrer/python-packaging-best-practices-4d6da500da5f](https://medium.com/@miqui.ferrer/python-packaging-best-practices-4d6da500da5f)
25. Clean dependencies for setup of a Python package with C++ extensions using Torch, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/74112770/clean-dependencies-for-setup-of-a-python-package-with-c-extensions-using-torch](https://stackoverflow.com/questions/74112770/clean-dependencies-for-setup-of-a-python-package-with-c-extensions-using-torch)
26. Torch cpu as a dependency of package \- deployment \- PyTorch Forums, дата последнего обращения: декабря 22, 2025, [https://discuss.pytorch.org/t/torch-cpu-as-a-dependency-of-package/53978](https://discuss.pytorch.org/t/torch-cpu-as-a-dependency-of-package/53978)
27. Torch version selection (CUDA vs CPU) for software development : r/pytorch \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/pytorch/comments/1f12n2a/torch_version_selection_cuda_vs_cpu_for_software/](https://www.reddit.com/r/pytorch/comments/1f12n2a/torch_version_selection_cuda_vs_cpu_for_software/)
28. How does one install PyTorch and related tools from within the setup.py install_requires list?, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/70295885/how-does-one-install-pytorch-and-related-tools-from-within-the-setup-py-install](https://stackoverflow.com/questions/70295885/how-does-one-install-pytorch-and-related-tools-from-within-the-setup-py-install)
29. Issues from importing \`torch\` in \`setup.py\` · Issue \#265 · rusty1s/pytorch_scatter \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/rusty1s/pytorch_scatter/issues/265](https://github.com/rusty1s/pytorch_scatter/issues/265)
30. Why is python setup.py saying invalid command 'bdist_wheel' on Travis CI? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/34819221/why-is-python-setup-py-saying-invalid-command-bdist-wheel-on-travis-ci](https://stackoverflow.com/questions/34819221/why-is-python-setup-py-saying-invalid-command-bdist-wheel-on-travis-ci)
31. Why is python setup.py saying invalid command 'bdist_wheel' on Travis CI? \- Codemia, дата последнего обращения: декабря 22, 2025, [https://codemia.io/knowledge-hub/path/why_is_python_setuppy_saying_invalid_command_bdist_wheel_on_travis_ci](https://codemia.io/knowledge-hub/path/why_is_python_setuppy_saying_invalid_command_bdist_wheel_on_travis_ci)
32. What Are Python Wheels and Why Should You Care?, дата последнего обращения: декабря 22, 2025, [https://realpython.com/python-wheels/](https://realpython.com/python-wheels/)
33. Day 38 — What's inside a Python wheel? \- Vinayak Mehta, дата последнего обращения: декабря 22, 2025, [https://vinayak.io/2020/10/04/day-38-whats-inside-a-python-wheel/](https://vinayak.io/2020/10/04/day-38-whats-inside-a-python-wheel/)
34. Core metadata specifications \- Python Packaging User Guide, дата последнего обращения: декабря 22, 2025, [https://packaging.python.org/specifications/core-metadata/](https://packaging.python.org/specifications/core-metadata/)
35. jwodder/check-wheel-contents: Check your wheels have the right contents \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/jwodder/check-wheel-contents](https://github.com/jwodder/check-wheel-contents)
36. wheel-inspect \- PyPI, дата последнего обращения: декабря 22, 2025, [https://pypi.org/project/wheel-inspect/](https://pypi.org/project/wheel-inspect/)
37. Getting Started with conda Environments \- Anaconda, дата последнего обращения: декабря 22, 2025, [https://www.anaconda.com/blog/getting-started-with-conda-environments](https://www.anaconda.com/blog/getting-started-with-conda-environments)
38. 3.1. How to create virtual environments for python with conda \- Numdifftools \- Read the Docs, дата последнего обращения: декабря 22, 2025, [https://numdifftools.readthedocs.io/en/latest/how-to/create_virtual_env_with_conda.html](https://numdifftools.readthedocs.io/en/latest/how-to/create_virtual_env_with_conda.html)
39. Install Python packages from wheel (.whl) files \- Sentry, дата последнего обращения: декабря 22, 2025, [https://sentry.io/answers/install-python-packages-from-wheel-whl-files/](https://sentry.io/answers/install-python-packages-from-wheel-whl-files/)
40. python \- Wheel file installation \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/28002897/wheel-file-installation](https://stackoverflow.com/questions/28002897/wheel-file-installation)
41. Identifying the dependency relationship for python packages installed with pip, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/9232568/identifying-the-dependency-relationship-for-python-packages-installed-with-pip](https://stackoverflow.com/questions/9232568/identifying-the-dependency-relationship-for-python-packages-installed-with-pip)
42. unable to make a python3 wheel because bdist_wheel is an invalid command, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/70459113/unable-to-make-a-python3-wheel-because-bdist-wheel-is-an-invalid-command](https://stackoverflow.com/questions/70459113/unable-to-make-a-python3-wheel-because-bdist-wheel-is-an-invalid-command)
43. Documentation for using find_packages() and package_dir() might be wrong. · Issue \#1571 · pypa/setuptools \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/pypa/setuptools/issues/1571](https://github.com/pypa/setuptools/issues/1571)
