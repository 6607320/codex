# **Архитектура Воспроизводимости: Комплексный Инженерный Отчет о Внедрении DVC и Разделении Ответственности в MLOps**

## **1\. Введение: Кризис Управления Артефактами в Машинном Обучении**

В современной экосистеме операций машинного обучения (MLOps) синхронизация кода, данных и артефактов моделей представляет собой фундаментальный инженерный вызов. Традиционные системы контроля версий (VCS), такие как Git, блестяще решают задачу отслеживания изменений в текстовых файлах исходного кода, обеспечивая распределенную совместную работу. Однако эти системы сталкиваются с критическими ограничениями при работе с бинарными объектами большого объема — многогигабайтными датасетами, весами нейронных сетей (файлы .pth, .h5, .onnx) и сложными иерархиями директорий, содержащими миллионы изображений или аудиофайлов.1  
Проблема заключается не просто в неудобстве хранения, а в архитектурном несоответствии. Git спроектирован для вычисления построчных различий (diffs) и сжатия текстовой истории. При попытке сохранить в репозитории Git бинарный файл размером 10 ГБ, система не только экспоненциально увеличивает размер папки .git, делая операции клонирования и переключения веток невыносимо медленными, но и теряет семантическую связь между версией кода и версией данных, использованных для его исполнения.3 Это создает «разрыв воспроизводимости» — ситуацию, когда инженер может восстановить код модели месячной давности, но не может гарантированно восстановить точное состояние данных, на которых эта модель обучалась.

### **1.1 Философия Разделения Ответственности**

Для решения этой дихотомии современные MLOps-архитектуры применяют принцип «разделения ответственности», который лежит в основе легенды о «Хранителе Данных» и «Летописце». Этот подход делит ответственность за версионирование на два синхронизированных слоя:

1. **Летописец (Git):** Отвечает за легковесные текстовые файлы — исходный код, конфигурации (YAML/JSON) и, что критически важно, _метаданные_ или указатели на данные. Git остается единственным источником правды (Source of Truth) для истории проекта.
2. **Хранитель (DVC \- Data Version Control):** Отвечает за тяжелые артефакты. Он действует как слой абстракции, управляя физическим хранением, дедупликацией и извлечением файлов, предоставляя Git лишь компактные файлы-указатели (.dvc).1

Данный отчет представляет собой исчерпывающее техническое руководство по внедрению и эксплуатации DVC. Мы детально рассмотрим процесс инициализации («Ритуал Призыва»), внутреннюю механику контентно-адресуемого хранилища (CAS), процессы синхронизации данных («Ответ Техноманту») и стратегии обеспечения целостности данных в распределенных командах.

## ---

**2\. Архитектурные Основы: Инициализация и Структура Проекта**

Процесс внедрения DVC в проект, описанный в квесте как «Ритуал Призыва», является строго детерминированной последовательностью операций, создающих необходимую инфраструктуру для версионирования данных. Этот процесс начинается с установки инструментария и завершается фиксацией базовой конфигурации в системе контроля версий.

### **2.1 Установка Инструментария (Активация Гримуара)**

Первым шагом является интеграция DVC в окружение разработчика. Команда pip install dvc устанавливает ядро DVC как Python-пакет. Важно отметить, что DVC может быть установлен и как системная утилита (через brew, apt, choco или бинарный пакет), что часто предпочтительнее для CI/CD агентов, чтобы избежать конфликтов зависимостей с библиотеками машинного обучения внутри виртуальных окружений.6 Однако в контексте Python-проектов (например, при использовании conda) установка через pip позволяет жестко зафиксировать версию DVC в requirements.txt, обеспечивая единообразие инструментария у всей команды.

### **2.2 Ритуал Инициализации (dvc init)**

Команда dvc init — это идемпотентная операция, которая трансформирует стандартный Git-репозиторий в гибридное пространство DVC-Git. При выполнении этой команды в корне проекта происходит создание скрытой директории .dvc.6

#### **2.2.1 Анатомия директории .dvc**

Директория .dvc является «мозговым центром» системы, аналогично директории .git для Git. Она содержит конфигурацию, локальный кэш и базы данных состояния. Понимание её структуры критично для отладки и настройки 6:

- **config:** Основной конфигурационный файл (в формате INI). Здесь хранятся настройки удаленных хранилищ (remotes), типы кэширования и параметры, общие для всех участников проекта. Этот файл версионируется Git, обеспечивая синхронизацию настроек инфраструктуры.10
- **.gitignore:** DVC автоматически создает внутри .dvc файл .gitignore, чтобы предотвратить попадание в Git локальных кэшей и временных файлов.
- **cache/:** (Обычно создается при первом добавлении данных, но путь определяется здесь). Это контентно-адресуемое хранилище (CAS), где физически размещаются файлы данных. Имена файлов здесь представляют собой хэш-суммы их содержимого (подробнее в Разделе 4).12
- **tmp/:** Директория для временных файлов, блокировок (locks) и промежуточных состояний операций.
- **plots/:** Шаблоны для визуализации метрик экспериментов.

После выполнения dvc init, команда git status покажет наличие новой директории .dvc и файлов .dvcignore. Эти изменения необходимо зафиксировать («Запечатывание Пакта»), чтобы инициализация стала частью истории проекта:

Bash

git add.dvc.dvcignore  
git commit \-m "feat(mlops): ✨ Initialize DVC for artifact versioning"

Это действие официально закрепляет использование DVC в проекте, позволяя другим разработчикам автоматически получить настроенное окружение при клонировании репозитория.6

### **2.3 Роль и Синтаксис .dvcignore**

В процессе работы над ML-проектами рабочая директория часто заполняется временными артефактами: логами TensorBoard, чекпоинтами моделей, которые не нужно сохранять, или системными файлами (.DS_Store, \_\_pycache\_\_). Отслеживание этих файлов через DVC привело бы к засорению хранилища и замедлению операций вычисления хэшей.6  
Файл .dvcignore выполняет функцию, аналогичную .gitignore, но для DVC. Он указывает, какие файлы и директории «Хранитель» должен игнорировать при сканировании рабочего пространства.  
**Ключевые аспекты .dvcignore:**

1. **Синтаксис:** Полностью совместим с .gitignore. Поддерживаются шаблоны (glob patterns), исключения через \! и комментарии через \#.
2. **Производительность:** Корректная настройка .dvcignore критически важна для быстродействия. Команды dvc status и dvc add. сканируют файловую систему. Исключение глубоких вложенных структур (например, папок с миллионами временных изображений) может ускорить выполнение команд с минут до секунд.14
3. **Разделение областей видимости:** Файлы, игнорируемые в .gitignore, часто являются именно теми файлами, которые _должен_ отслеживать DVC (большие датасеты). И наоборот, файлы, игнорируемые DVC, могут отслеживаться Git (исходный код). Однако существуют файлы (например, локальные конфиги IDE), которые должны быть проигнорированы обеими системами.14

## ---

**3\. Механика «Двух Ключей»: Синхронизация и Обновление Данных**

Вопрос «Техноманта» о том, как происходит синхронизация при изменении данных, затрагивает фундаментальный паттерн работы DVC. Этот процесс представляет собой цикл взаимодействия между файловой системой пользователя, демоном DVC и реестром Git. Понимание этого цикла необходимо для избежания рассинхронизации проекта.

### **3.1 Шаг 1: Действие Хранителя (dvc add)**

Когда инженер добавляет новый датасет (например, data/raw.csv) или обновляет существующий, он выполняет команду dvc add data/raw.csv. В этот момент происходит сложная последовательность операций 16:

1. **Вычисление Хэша:** DVC считывает файл и вычисляет его контрольную сумму (обычно MD5). Этот хэш становится уникальным идентификатором данной версии файла.
2. **Миграция в Кэш:** Файл физически перемещается (или копируется, в зависимости от настроек) в директорию .dvc/cache. Имя файла в кэше соответствует его хэшу. Это гарантирует дедупликацию: если файл с таким содержимым уже встречался в проекте ранее, DVC не будет создавать его копию.12
3. **Создание Указателя (Link):** В рабочем пространстве оригинальный файл data/raw.csv заменяется на ссылку (reflink, hardlink или copy), указывающую на файл в кэше. Для пользователя это выглядит так, будто файл остался на месте, но технически он теперь управляется DVC.
4. **Генерация Метаданных:** DVC создает (или обновляет) файл data/raw.csv.dvc. Это текстовый файл малого размера (обычно несколько байт), содержащий хэш, размер файла и путь к нему.
5. **Изоляция от Git:** DVC автоматически добавляет путь data/raw.csv в .gitignore. Это критический шаг, предотвращающий случайную фиксацию тяжелого файла в Git.16

### **3.2 Шаг 2: Запись в Летопись (git add)**

После выполнения dvc add, команда git status покажет, что файл data/raw.csv.dvc изменен (или создан), а сам файл данных игнорируется. Теперь вступает в силу вторая часть ритуала:

Bash

git add data/raw.csv.dvc.gitignore  
git commit \-m "Update dataset to version X"

Git фиксирует _изменение указателя_. Если содержимое датасета объемом 10 ГБ изменилось полностью, хэш в .dvc файле изменится всего на одну строку. Git сохраняет это крошечное текстовое изменение. Таким образом, история изменений данных становится неотъемлемой частью истории кода.5

### **3.3 Цикл Обновления Данных**

Отвечая на вопрос о _синхронизации изменений_, рассмотрим сценарий, когда данные модифицируются:

1. **Модификация:** Пользователь редактирует data/raw.csv или заменяет его новым файлом.
2. **Детекция:** Команда dvc status сравнивает хэш текущего файла в рабочей директории с хэшем, записанным в data/raw.csv.dvc. Обнаружив несовпадение, она сообщает: modified: data/raw.csv.
3. **Пересчет:** Пользователь снова запускает dvc add data/raw.csv. DVC вычисляет новый хэш, сохраняет новую версию файла в кэш (старая версия также остается в кэше) и обновляет файл .dvc.
4. **Фиксация:** Пользователь делает git commit обновленного .dvc файла.

**Инсайт второго порядка:** Этот механизм обеспечивает **иммутабельность данных**. Каждая версия датасета сохраняется в кэше как независимый объект. Git лишь переключает указатели между этими объектами. Это позволяет мгновенно откатываться к любой прошлой версии данных, просто переключив Git-коммит и выполнив dvc checkout.13

## ---

**4\. Глубокое Погружение: Контентно-Адресуемое Хранилище (CAS)**

Для инженера важно понимать, как именно DVC хранит данные, чтобы эффективно управлять дисковым пространством и производительностью.

### **4.1 Структура Хранилища**

DVC использует модель CAS (Content-Addressable Storage). Файлы идентифицируются не по имени, а по содержимому. Структура кэша по умолчанию выглядит так:  
.dvc/cache/files/md5/\<первые*2*символа*хэша\>/\<оставшиеся_30*символов\>  
Например, файл с хэшем ec1d2935f811b77cc49b031b999cbf17 будет сохранен по пути:  
.dvc/cache/files/md5/ec/1d2935f811b77cc49b031b999cbf17  
Такое разделение по директориям необходимо для предотвращения проблем с производительностью файловых систем, которые плохо справляются с тысячами файлов в одной папке.12

### **4.2 Обработка Директорий**

Когда под управление DVC берется целая директория (dvc add data/images), DVC не создает отдельный .dvc файл для каждого изображения. Вместо этого:

1. Вычисляются хэши всех файлов внутри директории.
2. Создается специальный JSON-файл (манифест), перечисляющий соответствие путей и хэшей.
3. Этот JSON-файл сам хэшируется, и его хэш (с расширением .dir) сохраняется в кэше.
4. В рабочем пространстве создается один файл data/images.dvc, указывающий на хэш этого манифеста.  
   Это позволяет эффективно управлять датасетами, состоящими из миллионов мелких файлов, избегая перегрузки Git.21

### **4.3 Стратегии Связывания Файлов (File Linking)**

Критическим аспектом производительности DVC является то, как файлы из кэша попадают в рабочее пространство. Копирование 50 ГБ данных из кэша в рабочую папку (copy) удвоило бы потребление диска и заняло бы много времени. DVC поддерживает несколько стратегий линковки, настраиваемых через dvc config cache.type 23:

| Стратегия                   | Механизм                                                                     | Преимущества                                                                                   | Недостатки                                                                                                                                          |
| :-------------------------- | :--------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reflink (Copy-on-Write)** | Файловая система создает логическую копию, указывающую на те же блоки диска. | Мгновенное создание, экономия места, безопасность редактирования (копия создается при записи). | Требует поддержки ФС (APFS на macOS, Btrfs/XFS на Linux). Не работает на стандартных ext4/NTFS без настройки. **Рекомендовано по умолчанию.**       |
| **Hardlink**                | Жесткая ссылка на inode файла в кэше.                                        | Мгновенно, нулевое потребление доп. места.                                                     | **Опасно:** Изменение файла в рабочей папке повреждает кэш. DVC делает файлы read-only для защиты, что требует dvc unprotect перед редактированием. |
| **Symlink**                 | Символическая ссылка.                                                        | Быстро, легковесно.                                                                            | Некоторые инструменты ML не умеют переходить по симлинкам. Проблемы при переносе папок.                                                             |
| **Copy**                    | Физическое копирование байтов.                                               | Самая надежная изоляция, работает везде.                                                       | Медленно, дублирует занимаемое место (x2 дискового пространства). Используется как fallback.                                                        |

**Рекомендация:** Для максимальной производительности на Linux-серверах рекомендуется использовать файловую систему XFS или Btrfs и стратегию reflink. В средах Windows часто приходится использовать copy или настраивать права для симлинков.23

## ---

**5\. Распределенная Работа: Удаленные Хранилища (Remotes)**

Настройка DVC локально — это лишь первый шаг. Для командной работы необходимо настроить удаленное хранилище (DVC Remote), которое выступает аналогом origin в Git, но для данных.

### **5.1 Конфигурация Удаленного Хранилища**

DVC поддерживает множество бэкендов: Amazon S3, Google Cloud Storage (GCS), Azure Blob Storage, SSH, HDFS и даже Google Drive. Добавление удаленного хранилища производится командой:

Bash

dvc remote add \-d myremote s3://my-bucket/dvc-storage

Флаг \-d делает это хранилище дефолтным. Эта конфигурация сохраняется в файле .dvc/config и коммитится в Git, становясь доступной всей команде.10

### **5.2 Управление Доступом и Секретами**

Критически важно не коммитить секретные ключи доступа (AWS Access Key, GCS Credentials) в Git. DVC решает эту проблему через локальную конфигурацию:

Bash

dvc remote modify \--local myremote credentialpath /path/to/keys.json

Эта команда записывает настройки в файл .dvc/config.local, который находится в .gitignore. Таким образом, каждый разработчик (и CI/CD система) может использовать свои учетные данные для доступа к одному и тому же бакету.10

### **5.3 Жизненный Цикл Push/Pull**

Синхронизация данных требует явных команд, так как Git не управляет передачей тяжелых файлов:

- **dvc push**: Анализирует локальный кэш, определяет, каких объектов не хватает в удаленном хранилище, и загружает только их. Это обеспечивает инкрементальную загрузку — передаются только изменения.29
- **dvc pull**: Обратная операция. Анализирует .dvc файлы в текущем рабочем пространстве, скачивает недостающие объекты из удаленного хранилища в локальный кэш и создает ссылки (links) в рабочей директории.30

**Частая ошибка:** Разработчики часто делают git push, но забывают сделать dvc push. В результате коллеги скачивают обновления кода и .dvc файлов, но при попытке сделать dvc pull получают ошибку Cache not found. Для автоматизации этого процесса можно использовать Git-хуки (pre-push), которые блокируют отправку кода, если данные не были отправлены в DVC.32

## ---

**6\. Воспроизводимость и «Путешествия во Времени»**

Бизнес-ценность DVC заключается в гарантии воспроизводимости экспериментов. Возможность вернуться к любому состоянию проекта в прошлом — это стандарт, который DVC привносит в Data Science.

### **6.1 Механика Переключения Версий**

Предположим, модель, обученная месяц назад, показывала лучшие результаты. Чтобы вернуться к тому состоянию:

1. **git checkout \<commit_hash\>**: Эта команда возвращает состояние кода и .dvc файлов к моменту прошлого эксперимента. Однако файлы данных в рабочей директории _еще не изменились_ (или удалены, если Git очистил рабочее дерево). Они рассинхронизированы.
2. **dvc checkout**: Эта команда считывает "старые" .dvc файлы, находит соответствующие хэши в локальном кэше и восстанавливает именно те версии данных, которые использовались в том коммите.13

### **6.2 Автоматизация через Git Hooks**

Чтобы исключить человеческий фактор и необходимость вручную запускать dvc checkout после каждого переключения ветки, DVC предоставляет команду dvc install. Она устанавливает Git-хуки (в .git/hooks/), в частности post-checkout.  
Теперь, когда разработчик делает git checkout experiment-branch, хук автоматически запускает dvc checkout. Переключение между версиями датасетов становится таким же прозрачным, как переключение версий кода.30

## ---

**7\. Сравнительный Анализ: DVC против Альтернатив**

Для обоснования выбора DVC необходимо сравнить его с альтернативными подходами, такими как Git LFS и Data Lakes (LakeFS).

### **7.1 DVC vs. Git LFS (Large File Storage)**

Git LFS часто рассматривается как нативное решение, но оно имеет серьезные недостатки для ML-задач 2:

| Характеристика         | Git LFS                                                               | DVC                                                                         |
| :--------------------- | :-------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| **Бэкенд**             | Требует специализированного LFS-сервера.                              | Любое объектное хранилище (S3, GCS, Azure, SSH).                            |
| **Дедупликация**       | Зависит от реализации сервера.                                        | Глобальная клиентская дедупликация (CAS).                                   |
| **Производительность** | Фильтры smudge/clean замедляют операции Git при большом числе файлов. | Работает независимо от Git, оптимизировано для больших данных.              |
| **Работа с ML**        | Только хранение файлов.                                               | Поддержка пайплайнов, метрик, графиков (dvc metrics).                       |
| **Кэширование**        | Ограниченные возможности локального кэша.                             | Полноценный общий кэш, возможность работы без скачивания всего репозитория. |

**Вывод:** Git LFS подходит для дизайна и геймдева (хранение PSD, текстур), но DVC является стандартом для MLOps благодаря интеграции с пайплайнами и гибкости выбора хранилища.

### **7.2 DVC vs. LakeFS / Delta Lake**

Инструменты типа LakeFS предоставляют Git-подобный интерфейс поверх объектного хранилища (S3).

- **LakeFS** работает на уровне _бакета_, позволяя делать ветвление всего Data Lake. Это мощное решение для табличных данных и огромных озер данных, где нужен SQL-доступ и транзакционность.2
- **DVC** работает на уровне _файловой системы_ разработчика. Он идеален, когда модель обучается на файлах (изображения, аудио, CSV), которые нужно "материализовать" на диске для подачи в скрипт обучения.

## ---

**8\. Оптимизация и Обслуживание: Сборка Мусора**

По мере развития проекта кэш DVC накапливает множество версий файлов. Если датасет в 1 ТБ обновлялся 5 раз, кэш может занимать 5 ТБ. Для управления этим используется механизм сборки мусора (Garbage Collection).

### **8.1 Команда dvc gc**

Команда dvc gc позволяет удалить из кэша неиспользуемые объекты. Однако понятие "используемый" здесь контекстно-зависимо 38:

- dvc gc \-w (workspace): Оставляет только те файлы, на которые ссылаются .dvc файлы в _текущей_ рабочей директории. Это агрессивная очистка. Если вы переключитесь на соседнюю ветку, данные придется скачивать заново.
- dvc gc \--all-branches: Сохраняет файлы, необходимые для любой существующей ветки Git. Это безопасный стандарт для локальной очистки.
- dvc gc \--all-commits: Сохраняет историю всех коммитов.

Также существует флаг \--cloud, который удаляет старые объекты из удаленного хранилища (S3). Это критически важно для контроля расходов на облачное хранение, но требует осторожности, так как удаление необратимо.38

## ---

**9\. Интеграция в CI/CD и Автоматизация**

DVC играет ключевую роль в CI/CD пайплайнах (например, GitHub Actions или GitLab CI), обеспечивая доставку данных для тестов и обучения моделей.

### **9.1 Настройка GitHub Actions**

Для использования DVC в CI необходимо настроить аутентификацию и кэширование. Типичный workflow включает:

1. **Checkout кода:** actions/checkout.
2. **Установка DVC:** iterative/setup-dvc.
3. **Аутентификация:** Использование секретов репозитория (Repository Secrets) для передачи ключей доступа (например, GDRIVE_CREDENTIALS_DATA или AWS_ACCESS_KEY_ID).
4. **Pull данных:** Выполнение dvc pull для загрузки артефактов, необходимых для тестов.40

Пример конфигурации шага для Google Drive с использованием Service Account:

YAML

\- name: Pull data with DVC  
 env:  
 GDRIVE_CREDENTIALS_DATA: ${{ secrets.GDRIVE\_CREDENTIALS\_DATA }}  
 run: |  
 dvc remote modify \--local myremote gdrive_service_account_json_file_path credentials.json  
 dvc pull

Это позволяет CI-агенту получить доступ к защищенным данным без хардкода секретов.28

## ---

**10\. Устранение Неисправностей и Граничные Случаи**

### **10.1 Конфликты Слияния (Merge Conflicts)**

Когда две ветки модифицируют один и тот же датасет, при слиянии возникает конфликт в .dvc файле. Git покажет конфликт хэшей.

- **Решение:** DVC не может автоматически "слить" два бинарных файла или две версии папки с изображениями. Инженер должен выбрать одну из версий (ours/theirs) или вручную объединить данные в рабочей папке, после чего заново выполнить dvc add для генерации нового хэша и разрешения конфликта.42

### **10.2 Ошибка "Too many open files"**

При работе с S3 и большим количеством мелких файлов DVC пытается распараллелить загрузку, открывая множество сетевых соединений. На macOS или Linux это может упереться в лимиты операционной системы (ulimit).

- **Решение:** Увеличить лимит дескрипторов файлов (ulimit \-n 10240\) или снизить количество потоков DVC (dvc pull \-j 4).32

### **10.3 Права Доступа в Общем Кэше**

Если несколько пользователей работают на одном сервере с общим кэшем, могут возникать ошибки Permission denied, когда один пользователь пытается перезаписать или линковать файл, созданный другим.

- **Решение:** Использовать настройку dvc config cache.shared group. Это заставляет DVC создавать файлы с правами, доступными для записи всей группе пользователей Unix, а не только владельцу файла.11

## ---

**11\. Заключение**

Внедрение DVC трансформирует хаотичный процесс работы с данными в структурированную инженерную дисциплину. Выполнив «Ритуал Призыва» и настроив союз между Git и DVC, команда получает надежный фундамент для воспроизводимого ML.  
Разделение ответственности, где Git хранит _намерения_ (код и метаданные), а DVC хранит _результаты и сырье_ (артефакты), позволяет масштабировать проекты до петабайтных объемов данных, не жертвуя гибкостью разработки. Ответ на вопрос «Техноманта» о синхронизации раскрывает элегантность этой архитектуры: иммутабельность данных и версионирование через легковесные указатели делают историю проекта прозрачной и защищенной от случайных искажений.  
DVC — это не просто инструмент, это стандарт де\-факто для обеспечения целостности и воспроизводимости в жизненном цикле машинного обучения, превращающий магию экспериментов в надежную технологию.

---

**Использованные материалы:**.1

#### **Источники**

1. The Complete Guide to Data Version Control With DVC \- DataCamp, дата последнего обращения: декабря 22, 2025, [https://www.datacamp.com/tutorial/data-version-control-dvc](https://www.datacamp.com/tutorial/data-version-control-dvc)
2. DVC vs. Git-LFS vs. Dolt vs. lakeFS: Data Versioning Compared, дата последнего обращения: декабря 22, 2025, [https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/](https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/)
3. Intro to MLOps: Data and model versioning \- Wandb, дата последнего обращения: декабря 22, 2025, [https://wandb.ai/site/articles/intro-to-mlops-data-and-model-versioning/](https://wandb.ai/site/articles/intro-to-mlops-data-and-model-versioning/)
4. The Full MLOps Blueprint: Reproducibility and Versioning in ML Systems—Part A (With Implementation) \- Daily Dose of Data Science, дата последнего обращения: декабря 22, 2025, [https://www.dailydoseofds.com/mlops-crash-course-part-3/](https://www.dailydoseofds.com/mlops-crash-course-part-3/)
5. Data Versioning: Why You Need It and How to Get Started with DVC | by Asjad Ali | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@asjad_ali/data-versioning-why-you-need-it-and-how-to-get-started-with-dvc-f9429bd5fc53](https://medium.com/@asjad_ali/data-versioning-why-you-need-it-and-how-to-get-started-with-dvc-f9429bd5fc53)
6. DVC Setup and Initialization, дата последнего обращения: декабря 22, 2025, [https://campus.datacamp.com/courses/introduction-to-data-versioning-with-dvc/dvc-configuration-and-data-management?ex=1](https://campus.datacamp.com/courses/introduction-to-data-versioning-with-dvc/dvc-configuration-and-data-management?ex=1)
7. Getting Started with Data Version Control (DVC) \- Analytics Vidhya, дата последнего обращения: декабря 22, 2025, [https://www.analyticsvidhya.com/blog/2023/05/dvc-a-git-for-data-and-models/](https://www.analyticsvidhya.com/blog/2023/05/dvc-a-git-for-data-and-models/)
8. init | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/command-reference/init](https://doc.dvc.org/command-reference/init)
9. Project Structure | Data Version Control · DVC \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/user-guide/project-structure](https://doc.dvc.org/user-guide/project-structure)
10. Remote Storage | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/user-guide/data-management/remote-storage](https://doc.dvc.org/user-guide/data-management/remote-storage)
11. DVC Configuration \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/user-guide/project-structure/configuration](https://doc.dvc.org/user-guide/project-structure/configuration)
12. Internal Files | Data Version Control · DVC, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/user-guide/project-structure/internal-files](https://doc.dvc.org/user-guide/project-structure/internal-files)
13. Get Started with DVC | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/start](https://doc.dvc.org/start)
14. dvcignore Files | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/user-guide/project-structure/dvcignore-files](https://doc.dvc.org/user-guide/project-structure/dvcignore-files)
15. gitignore Documentation \- Git, дата последнего обращения: декабря 22, 2025, [https://git-scm.com/docs/gitignore](https://git-scm.com/docs/gitignore)
16. add | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/command-reference/add](https://doc.dvc.org/command-reference/add)
17. Data Version Control With Python and DVC, дата последнего обращения: декабря 22, 2025, [https://realpython.com/python-data-version-control/](https://realpython.com/python-data-version-control/)
18. Tutorial: Data and Model Versioning \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/use-cases/versioning-data-and-models/tutorial](https://doc.dvc.org/use-cases/versioning-data-and-models/tutorial)
19. MLOps | Versioning Datasets with Git & DVC \- Analytics Vidhya, дата последнего обращения: декабря 22, 2025, [https://www.analyticsvidhya.com/blog/2021/06/mlops-versioning-datasets-with-git-dvc/](https://www.analyticsvidhya.com/blog/2021/06/mlops-versioning-datasets-with-git-dvc/)
20. 9 tips for data version control in large projects \- Julia Wąsala, дата последнего обращения: декабря 22, 2025, [https://juliawasala.nl/blog/dvc-large-projects/](https://juliawasala.nl/blog/dvc-large-projects/)
21. How to add/update data with dvc workflow? \- git \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/75923500/how-to-add-update-data-with-dvc-workflow](https://stackoverflow.com/questions/75923500/how-to-add-update-data-with-dvc-workflow)
22. Better way to add large directories · Issue \#4782 · treeverse/dvc \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/iterative/dvc/issues/4782](https://github.com/iterative/dvc/issues/4782)
23. Large Dataset Optimization | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/user-guide/data-management/large-dataset-optimization](https://doc.dvc.org/user-guide/data-management/large-dataset-optimization)
24. Reflinks vs symlinks vs hard links, and how they can help machine learning projects, дата последнего обращения: декабря 22, 2025, [https://dev.to/robogeek/reflinks-vs-symlinks-vs-hard-links-and-how-they-can-help-machine-learning-projects-1cj4](https://dev.to/robogeek/reflinks-vs-symlinks-vs-hard-links-and-how-they-can-help-machine-learning-projects-1cj4)
25. MLOps : How DVC smartly manages your data sets for training your machine learning models on top of Git \- LittleBigCode, дата последнего обращения: декабря 22, 2025, [https://littlebigcode.fr/en/how-dvc-manages-data-sets-training-ml-models-git/](https://littlebigcode.fr/en/how-dvc-manages-data-sets-training-ml-models-git/)
26. Add hardlinks to cache · Issue \#10601 · treeverse/dvc \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/iterative/dvc/issues/10601](https://github.com/iterative/dvc/issues/10601)
27. dvc.org/content/docs/command-reference/remote/index.md at main · treeverse/dvc.org · GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/iterative/dvc.org/blob/main/content/docs/command-reference/remote/index.md](https://github.com/iterative/dvc.org/blob/main/content/docs/command-reference/remote/index.md)
28. Setting Up a Workflow with DVC, Google Drive and GitHub Actions | by Ajith Kumar V, дата последнего обращения: декабря 22, 2025, [https://medium.com/@ajithkumarv/setting-up-a-workflow-with-dvc-google-drive-and-github-actions-f3775de4bf63](https://medium.com/@ajithkumarv/setting-up-a-workflow-with-dvc-google-drive-and-github-actions-f3775de4bf63)
29. Trying to understand data storage \- Questions \- DVC, дата последнего обращения: декабря 22, 2025, [https://discuss.dvc.org/t/trying-to-understand-data-storage/1378](https://discuss.dvc.org/t/trying-to-understand-data-storage/1378)
30. install | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/command-reference/install](https://doc.dvc.org/command-reference/install)
31. Does dvc checkout pulls the data or just checkouts .dvc files? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/76436642/does-dvc-checkout-pulls-the-data-or-just-checkouts-dvc-files](https://stackoverflow.com/questions/76436642/does-dvc-checkout-pulls-the-data-or-just-checkouts-dvc-files)
32. Troubleshooting | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/user-guide/troubleshooting](https://doc.dvc.org/user-guide/troubleshooting)
33. Automate DVC Pipelines Reproduction with Makefile \- Theodo Data & AI, дата последнего обращения: декабря 22, 2025, [https://data-ai.theodo.com/en/technical-blog/automate-dvc-pipelines-reproduction-with-makefile](https://data-ai.theodo.com/en/technical-blog/automate-dvc-pipelines-reproduction-with-makefile)
34. checkout | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/command-reference/checkout](https://doc.dvc.org/command-reference/checkout)
35. Git LFS and DVC: The Ultimate Guide to Managing Large Artifacts in MLOps \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@pablojusue/git-lfs-and-dvc-the-ultimate-guide-to-managing-large-artifacts-in-mlops-c1c926e6c5f4](https://medium.com/@pablojusue/git-lfs-and-dvc-the-ultimate-guide-to-managing-large-artifacts-in-mlops-c1c926e6c5f4)
36. Why is DVC Better Than Git and Git-LFS in Machine Learning Reproducibility \- Harshil Patel, дата последнего обращения: декабря 22, 2025, [https://harshilp.medium.com/why-is-dvc-better-than-git-and-git-lfs-in-machine-learning-reproducibility-13102f47e00c](https://harshilp.medium.com/why-is-dvc-better-than-git-and-git-lfs-in-machine-learning-reproducibility-13102f47e00c)
37. Reproducibility in ML: The Role of Data Versioning | by Anay Nayak \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/inspiredbrilliance/reproducibility-in-ml-the-role-of-data-versioning-b5a504bea8b4](https://medium.com/inspiredbrilliance/reproducibility-in-ml-the-role-of-data-versioning-b5a504bea8b4)
38. gc | Data Version Control \- DVC Documentation, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/command-reference/gc](https://doc.dvc.org/command-reference/gc)
39. dvc gc TLDR page \- Cheat-Sheets.org, дата последнего обращения: декабря 22, 2025, [https://www.cheat-sheets.org/project/tldr/command/dvc-gc/](https://www.cheat-sheets.org/project/tldr/command/dvc-gc/)
40. Complete CI \-Pipeline — MLOps with Github Actions and DvC | by Basu Verma \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@basu.verma/complete-ci-pipeline-mlops-with-github-actions-and-dvc-564c67f3b43b](https://medium.com/@basu.verma/complete-ci-pipeline-mlops-with-github-actions-and-dvc-564c67f3b43b)
41. Automate DVC authentication when using github actions \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/74017026/automate-dvc-authentication-when-using-github-actions](https://stackoverflow.com/questions/74017026/automate-dvc-authentication-when-using-github-actions)
42. How to Resolve Merge Conflicts in DVC | Data Version Control, дата последнего обращения: декабря 22, 2025, [https://doc.dvc.org/user-guide/how-to/resolve-merge-conflicts](https://doc.dvc.org/user-guide/how-to/resolve-merge-conflicts)
43. What DVC does when git merge is executed? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/73567700/what-dvc-does-when-git-merge-is-executed](https://stackoverflow.com/questions/73567700/what-dvc-does-when-git-merge-is-executed)
44. DVC assigns user permissions to folders in the ssh repository, дата последнего обращения: декабря 22, 2025, [https://discuss.dvc.org/t/dvc-assigns-user-permissions-to-folders-in-the-ssh-repository/898](https://discuss.dvc.org/t/dvc-assigns-user-permissions-to-folders-in-the-ssh-repository/898)
