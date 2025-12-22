# **Архитектура и оптимизация высокопроизводительных конвейеров данных в PyTorch: Глубокий анализ реализации Dataset и DataLoader**

## **Введение**

В современной экосистеме глубокого обучения (Deep Learning) наблюдается парадоксальная ситуация: в то время как вычислительные мощности ускорителей (GPU/TPU) растут по экспоненте, общая эффективность обучения моделей часто ограничивается скоростью подачи данных. По мере того как архитектуры нейронных сетей становятся все более глубокими, а объемы обучающих выборок масштабируются от гигабайт до петабайт, подсистема ввода-вывода (I/O) и предварительной обработки данных превращается из вспомогательного компонента в критически важный инфраструктурный элемент.  
Задача создания эффективного загрузчика данных («даталоадера») в PyTorch — это не просто написание связующего кода на Python. Это сложная инженерная проблема, требующая глубокого понимания принципов работы операционной системы, управления памятью, механизмов многопроцессорности (multiprocessing), сериализации данных и аппаратных особенностей дисковых подсистем. «Голодание GPU» (GPU starvation) — состояние, при котором вычислительное устройство простаивает в ожидании очередного батча данных от центрального процессора (CPU), — является главным врагом производительности.  
В данном отчете представлен исчерпывающий анализ архитектурных паттернов, стратегий оптимизации и низкоуровневых механизмов загрузки данных в PyTorch. Мы детально рассмотрим дихотомию между подходами Map-style и Iterable-style, исследуем влияние Global Interpreter Lock (GIL) на параллелизацию, проведем сравнительный анализ форматов хранения данных (от плоских файлов до LMDB и WebDataset) и предоставим эталонные реализации для создания отказоустойчивых пайплайнов.1

## **1\. Теоретические основы и архитектура подсистемы данных PyTorch**

Цикл обучения модели машинного обучения концептуально можно представить как классическую задачу производителя и потребителя (Producer-Consumer problem). В этой схеме «потребителем» выступает GPU, выполняющий матричные операции, а «производителем» — CPU, отвечающий за извлечение данных с диска, их декодирование, аугментацию и формирование пакетов (батчей). Идеальная система поддерживает утилизацию потребителя на уровне, близком к 100%. Однако, как показывает практика, достижение этого показателя требует тщательной настройки баланса между I/O, CPU-нагрузкой и пропускной способностью шины PCIe.

### **1.1 Абстракция Dataset: Статическое определение данных**

В основе архитектуры PyTorch лежит четкое разделение ответственности (separation of concerns). Класс Dataset отвечает за _определение_ данных: где они находятся, каков их объем и как получить один конкретный пример. Это статическое описание.  
Существует два фундаментальных типа датасетов, выбор между которыми определяет всю дальнейшую архитектуру конвейера:

1. **Map-style Dataset**: Реализует протокол прямого доступа (random access).
2. **Iterable-style Dataset**: Реализует протокол последовательного потока (streaming).

### **1.2 Абстракция DataLoader: Динамический движок**

Класс DataLoader — это динамический механизм, который «оживляет» статический Dataset. Он инкапсулирует в себе сложную логику, которую исследователю было бы утомительно реализовывать каждый раз заново:

- **Батчинг (Batching):** Агрегация отдельных сэмплов в тензоры.
- **Перемешивание (Shuffling):** Рандомизация порядка выборки для обеспечения i.i.d. (independent and identically distributed) свойств данных, критичных для стохастического градиентного спуска.
- **Многопроцессорность (Multiprocessing):** Обход ограничений GIL путем порождения пула рабочих процессов (workers) для параллельной загрузки.
- **Закрепление памяти (Memory Pinning):** Использование cudaHostAlloc для ускорения трансфера данных из RAM в VRAM.2

## **2\. Map-style Datasets: Стандарт для произвольного доступа**

Для подавляющего большинства задач обучения с учителем (Supervised Learning), особенно в области компьютерного зрения (Computer Vision) и обработки естественного языка (NLP) на фиксированных корпусах, стандартом де\-факто является Map-style dataset. Такой класс должен наследовать torch.utils.data.Dataset и реализовывать два «магических» метода: \_\_getitem\_\_ и \_\_len\_\_.

### **2.1 Метод \_\_getitem\_\_: Критический путь**

Метод \_\_getitem\_\_(self, index) является «узким местом» всего пайплайна. Он получает на вход целочисленный индекс (от 0 до len(dataset) \- 1\) и должен вернуть один готовый к обучению сэмпл (пару «данные-метка»). Эффективность этого метода напрямую определяет максимальную пропускную способность системы.

#### **2.1.1 Стратегии загрузки: Eager vs. Lazy Loading**

Фундаментальное архитектурное решение при проектировании Dataset — выбор момента загрузки данных в оперативную память.  
Жадная загрузка (Eager Loading):  
При этом подходе все данные считываются в память (RAM) на этапе инициализации (\_\_init\_\_).

- _Механизм:_ В конструкторе класса происходит чтение всех файлов, их декодирование и сохранение в виде списка массивов (Numpy arrays) или тензоров.
- _Преимущества:_ Максимально быстрый доступ в процессе обучения, так как отсутствует задержка на I/O диска (disk seek/read latency). Доступ сводится к копированию памяти.
- _Недостатки:_ Жесткое ограничение объемом оперативной памяти. Неприменимо для датасетов объемом более 10-20% от доступной RAM. Долгое время старта обучения («холодный старт»).
- _Применимость:_ Небольшие табличные данные (например, Titanic, Iris), маленькие текстовые корпуса, MNIST.1

Ленивая загрузка (Lazy Loading):  
Это стандарт для промышленных систем. В \_\_init\_\_ загружаются только метаданные (например, пути к файлам или смещения в бинарном файле), а само чтение данных («тяжелого» контента) происходит внутри \_\_getitem\_\_ по требованию.

- _Механизм:_ \_\_getitem\_\_ открывает файл, читает байты, декодирует изображение/аудио, применяет трансформации и возвращает тензор.
- _Преимущества:_ Минимальное потребление RAM. Возможность работы с датасетами произвольного размера (терабайты/петабайты). Быстрый старт.
- _Недостатки:_ Внесение задержки (latency) на каждой итерации. Нагрузка на файловую систему (тысячи операций open/read/close в секунду).
- _Оптимизация:_ Эта задержка маскируется механизмом prefetching (предвыборки) в DataLoader, когда рабочие процессы готовят следующие батчи, пока GPU обрабатывает текущий.5

### **2.2 Управление памятью метаданных**

При реализации ленивой загрузки (Lazy Loading) критически важно эффективно хранить метаданные. Наивная реализация может привести к исчерпанию памяти еще до начала обучения.  
Рассмотрим сценарий обучения на наборе данных ImageNet-21k или внутреннем корпоративном датасете из 100 миллионов изображений. Хранение списка путей к файлам в виде стандартного списка строк Python (List\[str\]) крайне неэффективно.  
В Python каждый строковый объект (str) является сложной структурой, имеющей значительный оверхед (около 49-80 байт на заголовок объекта \+ длина самой строки в кодировке Unicode). Список из 100 миллионов строк может занимать десятки гигабайт памяти только под хранение путей.  
Анализ потребления памяти:  
В таблице ниже приведено сравнение структур данных для хранения 10 миллионов путей к файлам (средняя длина пути 50 символов).

| Структура данных        | Примерный объем RAM | Особенности                                           |
| :---------------------- | :------------------ | :---------------------------------------------------- |
| List\[str\] (Python)    | \~1.5 \- 2.5 GB     | Высокий оверхед на каждый объект, фрагментация кучи.  |
| List\[bytes\]           | \~1.0 GB            | Чуть лучше, но все еще много маленьких объектов.      |
| pandas.DataFrame        | \~600-800 MB        | Оптимизировано, но имеет накладные расходы на индекс. |
| numpy.array (dtype='S') | \~500 MB            | Компактное хранение в непрерывном блоке памяти.       |
| pyarrow.Array           | \~450 MB            | Zero-copy чтение, высокая эффективность.              |

**Рекомендации по оптимизации:**

1. **Дедупликация путей:** Вместо хранения полных абсолютных путей (/data/datasets/imagenet/train/n02084071/n02084071_1.jpg), храните один раз корневую директорию (self.root) и список относительных имен.
2. **Использование индексов:** Если имена файлов систематизированы (например, img_0000001.jpg), не храните строки вообще. Храните только целочисленный ID и генерируйте имя файла на лету с помощью f-строк: f"img\_{idx:07d}.jpg". Это сводит потребление памяти к нулю.
3. **Использование Arrow/Parquet:** Для огромных списков метаданных загрузка их из CSV может занимать минуты. Использование формата Parquet и библиотеки PyArrow позволяет загружать метаданные мгновенно (memory mapping) и хранить их компактно.7

### **2.3 Протокол \_\_len\_\_ и его подводные камни**

Метод \_\_len\_\_ должен возвращать точное количество доступных сэмплов. В контексте распределенного обучения (DistributedDataParallel \- DDP) этот метод используется DistributedSampler для разделения индексов между GPU. Если \_\_len\_\_ вернет некорректное значение:

- Некоторые данные могут быть пропущены.
- Может возникнуть IndexError, если сэмплер запросит индекс, выходящий за границы.
- В DDP процессы могут зависнуть (deadlock), ожидая данные, которых нет у одного из воркеров, если размеры шардов не выровнены.

## **3\. Iterable-style Datasets: Потоковая обработка**

Хотя Map-style датасеты доминируют, они опираются на допущение, что произвольный доступ (random seek) дешев. Это предположение разрушается в двух сценариях:

1. **Потоковые данные:** Данные поступают из сетевого сокета, очереди сообщений (Kafka/RabbitMQ) или курсора базы данных.
2. **Последовательные хранилища:** Данные хранятся на ленточных накопителях или в огромных архивах (например, .tar.gz на S3), где чтение произвольного файла требует распаковки всего архива до нужной точки.

Для этих случаев PyTorch предоставляет torch.utils.data.IterableDataset. Реализация требует переопределения метода \_\_iter\_\_, который возвращает итератор Python.10

### **3.1 Проблема дупликации данных в мультипроцессорной среде**

Критический нюанс реализации IterableDataset заключается в его поведении при num_workers \> 0\. Когда DataLoader запускает рабочие процессы, объект Dataset сериализуется (pickle) и отправляется каждому воркеру.  
Если метод \_\_iter\_\_ реализован наивно (например, просто возвращает iter(self.data_list)), **каждый рабочий процесс будет итерироваться по одной и той же копии данных**. Это приведет к тому, что за одну эпоху модель увидит каждый пример num_workers раз. Это катастрофически влияет на обучение: "эпоха" перестает соответствовать одному проходу по данным, а батч-норм слои получают смещенные статистики.13

### **3.2 Решение проблемы разделения нагрузки (Sharding)**

Чтобы избежать дублирования, IterableDataset должен быть «осведомлен» о том, что он работает внутри рабочего процесса. Внутри метода \_\_iter\_\_ необходимо использовать функцию torch.utils.data.get_worker_info().  
**Алгоритм реализации шардинга:**

1. Вызвать get_worker_info().
2. Если результат None — значит, загрузка идет в главном процессе (single-process). Можно возвращать полный итератор.
3. Если результат — объект WorkerInfo, значит, мы внутри воркера. Получаем worker_id (ID текущего воркера) и num_workers (общее число воркеров).
4. Настроить итератор так, чтобы он возвращал только свою часть данных (1/num_workers).

**Стратегии шардинга:**

- **Чередование (Interleaving):** Воркер i берет элементы с индексами i, i \+ N, i \+ 2N.... Этот метод требует, чтобы источник данных поддерживал пропуск элементов (skipping), что не всегда эффективно для потоков.
- **Блочный шардинг (File-based sharding):** Если датасет состоит из множества файлов (шардов), список файлов делится между воркерами. Воркер 0 читает файлы , воркер 1 — и т.д. Это наиболее эффективный подход для больших систем (используется в WebDataset).14

Python

import math  
import torch  
from torch.utils.data import IterableDataset, get_worker_info

class ShardedIterableDataset(IterableDataset):  
 def \_\_init\_\_(self, start, end):  
 super(ShardedIterableDataset).\_\_init\_\_()  
 self.start \= start  
 self.end \= end

    def \_\_iter\_\_(self):
        worker\_info \= get\_worker\_info()
        if worker\_info is None:
            \# Однопроцессный режим: отдаем всё
            iter\_start \= self.start
            iter\_end \= self.end
        else:
            \# Многопроцессный режим: делим диапазон
            per\_worker \= int(math.ceil((self.end \- self.start) / float(worker\_info.num\_workers)))
            worker\_id \= worker\_info.id
            iter\_start \= self.start \+ worker\_id \* per\_worker
            iter\_end \= min(iter\_start \+ per\_worker, self.end)

        return iter(range(iter\_start, iter\_end))

## **4\. Движок DataLoader: Внутреннее устройство и параметры**

DataLoader — это оркестратор, управляющий процессами, очередями и памятью. Глубокое понимание его параметров необходимо для устранения узких мест.

### **4.1 num_workers, GIL и накладные расходы на форк**

Параметр num_workers управляет параллелизмом.

- num_workers=0: Загрузка происходит синхронно в главном процессе. Просто для отладки, но блокирует обучение.
- num_workers\>0: Используется модуль multiprocessing.

В Python из\-за GIL (Global Interpreter Lock) потоки (threads) не могут выполнять байт-код Python параллельно на разных ядрах CPU. Поэтому для загрузки данных, требующей вычислений (аугментации, декодирование), используются _процессы_.  
Механизм:  
При запуске итерации главный процесс использует системный вызов fork (на Linux/MacOS) или spawn (на Windows/MacOS по умолчанию для PyTorch) для создания копий.

- **Fork:** Быстрый старт, использует Copy-on-Write (CoW). Память родителя доступна детям до первой записи.
- **Spawn:** Создает чистый интерпретатор Python, затем "пиклит" (pickle) объект Dataset и передает его воркеру. Это медленнее и требует, чтобы все атрибуты Dataset были сериализуемы.

**Инсайт:** Слишком большое значение num_workers вредит. Каждый процесс потребляет RAM (копия интерпретатора \+ данные). Если сумма памяти воркеров превысит физическую RAM, начнется свопинг (swapping) на диск, и производительность упадет до нуля. Оптимальное число — обычно равно количеству физических ядер CPU (не логических потоков Hyper-Threading), минус 1-2 ядра для системы и основного процесса.4

### **4.2 pin_memory: Ускорение трансфера Host-to-Device**

Параметр pin_memory=True активирует использование закрепленной (pinned, page-locked) памяти.  
Обычно память, выделяемая malloc в пользовательском пространстве, является страничной (pageable) — ОС может выгрузить её в swap-файл. GPU (через контроллер DMA) не может безопасно читать из такой памяти, так как адрес может измениться или данные могут отсутствовать.  
При передаче тензора на GPU (tensor.cuda()) из обычной памяти драйвер CUDA сначала неявно копирует данные во временный буфер закрепленной памяти («pinned staging buffer»), и только оттуда запускает DMA-трансфер на устройство.  
Включение pin_memory заставляет DataLoader сразу размещать выходные батчи в специальной области памяти, зарегистрированной в драйвере CUDA (через cudaHostAlloc). Это исключает лишнее копирование на CPU и позволяет асинхронную отправку данных, существенно ускоряя tensor.to(device).4

### **4.3 collate_fn: Конструктор батчей**

По умолчанию DataLoader использует default_collate, который пытается стекировать (stack) элементы списка в тензоры. Это работает только для данных фиксированной размерности.  
**Сценарии для кастомного collate_fn:**

1. **Последовательности переменной длины (NLP/Audio):** Нельзя сделать torch.stack для предложений разной длины. Кастомная функция должна применять паддинг (padding) — дополнять короткие последовательности нулями до длины самой длинной в батче.
2. **Обработка ошибок:** Если \_\_getitem\_\_ вернул None из\-за битого файла, default_collate упадет с ошибкой. Кастомный collate_fn может отфильтровать None из списка перед созданием батча.
3. **Сложные структуры:** Если датасет возвращает графы, деревья или собственные объекты Python, стандартный коллейтор не поймет, как их объединять.17

Python

def robust_collate(batch):  
 \# Фильтрация None (битых данных)  
 batch \= \[b for b in batch if b is not None\]  
 if len(batch) \== 0:  
 return None \# Или вернуть пустой тензор, но это нужно обрабатывать в цикле обучения  
 return torch.utils.data.dataloader.default_collate(batch)

## **5\. Форматы хранения и стратегии ввода-вывода (I/O)**

При масштабировании до терабайтов данных, узким местом становится файловая система. Наивный подход хранения миллионов отдельных файлов изображений (.jpg, .png) приводит к деградации производительности.

### **5.1 Проблема «миллиона мелких файлов» (Small Files Problem)**

Файловые системы (ext4, NTFS, XFS) оптимизированы для файлов среднего и большого размера. Чтение миллиона файлов размером по 50 КБ каждый вызывает следующие проблемы:

1. **Нагрузка на метаданные:** Для каждого файла ОС должна найти его индексный дескриптор (inode) в таблице размещения, проверить права доступа и найти физические блоки на диске. Это приводит к огромному числу случайных обращений к диску (random seeks), даже если файлы логически идут подряд.
2. **Фрагментация:** Файлы могут быть разбросаны по всему диску, увеличивая время позиционирования головки (для HDD) или нагрузку на контроллер (для SSD).
3. **Ограничение по инодам:** Можно исчерпать лимит инодов (inodes) файловой системы, даже если свободного места на диске еще много.

**Решение:** Агрегация. Упаковка мелких файлов в крупные контейнеры.21

### **5.2 Сравнительный анализ форматов хранения**

Для высокопроизводительного ML существуют специализированные форматы, решающие проблему I/O.

#### **5.2.1 Плоские файлы (Flat Files)**

- **Описание:** Классическая структура папок (ImageFolder).
- **Плюсы:** Простота отладки, возможность просмотра глазами.
- **Минусы:** Катастрофическая производительность на HDD, высокая нагрузка на CPU (syscalls), медленное копирование датасета.
- **Вердикт:** Только для прототипирования.

#### **5.2.2 LMDB (Lightning Memory-Mapped Database)**

LMDB — это встраиваемое key-value хранилище на основе B+ деревьев.

- **Механизм:** Использует отображение памяти (mmap). ОС отображает файл базы данных в виртуальное адресное пространство процесса.
- **Плюсы:**
  - Сверхбыстрое случайное чтение (random read).
  - Отсутствие копирования в пользовательском пространстве (Zero-copy), если данные не требуют декодирования.
  - Эффективное использование кэша страниц ОС (Page Cache). Горячие данные остаются в RAM автоматически.
- **Минусы:** Сложность создания (требует конвертации датасета), фиксированный размер базы данных при создании.
- **Use Case:** ImageNet, большие наборы изображений для Random Access обучения.22

#### **5.2.3 HDF5 (Hierarchical Data Format)**

Бинарный формат, популярный в науке. Позволяет хранить многомерные массивы и иерархии внутри одного файла.

- **Плюсы:** Удобен для матричных данных, поддержка сжатия, метаданных.
- **Минусы:** Плохо дружит с многопроцессорностью PyTorch. Библиотека h5py не является потокобезопасной при записи и требует осторожности при чтении в форкнутых процессах (нужно открывать файл заново внутри каждого воркера, передавать файловый дескриптор через pickle нельзя).
- **Вердикт:** Использовать с осторожностью.22

#### **5.2.4 WebDataset (Tar-based)**

Современный стандарт для петабайтных датасетов. Данные хранятся в виде последовательности POSIX tar-архивов (шардов).

- **Механизм:** Последовательное чтение. Датасет — это поток байтов.
- **Плюсы:**
  - Скорость чтения близка к линейной скорости диска/сети.
  - Идеально для потоковой передачи с S3/GCS без полной загрузки.
  - Совместимость со стандартными утилитами Unix (tar, pipe).
- **Минусы:** Сложно реализовать истинно случайный доступ (Random Access). Используется буфер перемешивания (shuffle buffer), который дает лишь локальную рандомизацию (approximate shuffling).
- **Use Case:** Обучение LLM, CLIP, Stable Diffusion на кластерах.21

| Формат     | Скорость (Seq)   | Скорость (Rand)   | Сложность | Параллелизм |
| :--------- | :--------------- | :---------------- | :-------- | :---------- |
| Файлы      | Низкая           | Очень низкая      | Низкая    | Хороший     |
| LMDB       | Высокая          | **Очень высокая** | Высокая   | Отличный    |
| HDF5       | Высокая          | Средняя           | Средняя   | Сложный     |
| WebDataset | **Максимальная** | Н/Д (потоковый)   | Средняя   | Отличный    |

#### **5.2.5 Табличные форматы: CSV vs Parquet**

Для табличных данных CSV является худшим выбором из\-за затрат на парсинг текста (строка \-\> число).

- **Parquet:** Колоночный бинарный формат. Поддерживает сжатие (Snappy/Gzip) и _projection pushdown_ — чтение только нужных колонок без считывания всей строки. Это радикально снижает I/O.24

### **5.3 Memory Mapping (numpy.memmap) для гигантских матриц**

Для задач, где данные представлены огромными плотными матрицами (например, гиперспектральные снимки, медицинские 3D-сканы), которые не помещаются в RAM, использование numpy.memmap является спасением.  
Это позволяет обращаться к файлу на диске как к массиву NumPy в памяти. ОС сама подгружает нужные страницы (pages) в физическую память при обращении к индексам и выгружает неиспользуемые.  
Реализация:  
В \_\_init\_\_ создается объект memmap (он легок, так как не читает данные). В \_\_getitem\_\_ происходит чтение слайса. Важно открывать memmap в режиме r (read-only), чтобы избежать блокировок и случайной порчи данных. При передаче в torch.from_numpy() следует помнить, что тензор может захватить "view" на файл. Для полной безопасности лучше делать .copy() внутри \_\_getitem\_\_, чтобы перенести данные в обычную память процесса и позволить ОС освободить файловый кэш, но это зависит от паттерна доступа.27

## **6\. Аугментация данных: CPU vs GPU**

После загрузки данных их необходимо трансформировать. Выбор места выполнения трансформаций критичен.

### **6.1 Библиотеки: Torchvision vs Albumentations**

Стандартная библиотека torchvision.transforms долгое время использовала PIL (Pillow) как бэкенд. PIL работает на CPU и не всегда оптимизирован (особенно для сложных геометрических преобразований).  
Albumentations:  
Библиотека, ставшая стандартом в соревнованиях Kaggle и индустрии.

- **Производительность:** Использует OpenCV и SIMD-инструкции процессора. Бенчмарки показывают ускорение от 2x до 4x по сравнению с Torchvision.
- **Функционал:** Огромный набор аугментаций, поддержка одновременной трансформации изображений, масок сегментации и ключевых точек (keypoints), что сложно сделать в vanilla Torchvision.
- **Интеграция:** Легко встраивается в \_\_getitem\_\_. Требует конвертации PIL \-\> Numpy.30

### **6.2 GPU-аугментация (Kornia / DALI)**

Если CPU становится узким местом (даже при num_workers=max), можно перенести аугментации на GPU.

- **Библиотека Kornia:** Реализует трансформации как дифференцируемые модули nn.Module.
- **Подход:** Dataset загружает «сырые» тензоры. DataLoader собирает батч. Аугментация происходит уже на GPU перед подачей в модель.
- **Плюсы:** Массивная параллелизация, разгрузка CPU.
- **Минусы:** Потребляет VRAM, которая может быть нужна модели.

## **7\. Практическая реализация: Квест 15.1**

Ниже представлено пошаговое руководство по созданию профессионального загрузчика, интегрирующего обсужденные выше концепции.

### **7.1 Задача**

Создать кастомный Dataset и DataLoader для загрузки изображений, обеспечив устойчивость к ошибкам, высокую скорость и корректную обработку.

### **7.2 Этап 1: Реализация надежного Dataset**

Мы используем os.scandir для быстрой инициализации и добавим обработку исключений (corrupted data handling).

Python

import os  
import torch  
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import numpy as np

class CustomImageDataset(Dataset):  
 def \_\_init\_\_(self, root_dir, transform=None):  
 """  
 Инициализация датасета.  
 Args:  
 root_dir (str): Путь к папке с изображениями.  
 transform (callable, optional): Трансформации (аугментации).  
 """  
 self.root_dir \= root_dir  
 self.transform \= transform

        \# Оптимизация 1: Использование os.scandir вместо os.listdir
        \# scandir возвращает итератор и избегает лишних системных вызовов stat()
        \# Это ускоряет сканирование папок с миллионами файлов в разы.
        self.image\_files \=
        try:
            with os.scandir(root\_dir) as entries:
                for entry in entries:
                    if entry.is\_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.image\_files.append(entry.name)
        except FileNotFoundError:
            print(f"Директория {root\_dir} не найдена.")
            self.image\_files \=

        \# Сортировка для детерминированности экспериментов
        self.image\_files.sort()

    def \_\_len\_\_(self):
        return len(self.image\_files)

    def \_\_getitem\_\_(self, idx):
        """
        Ленивая загрузка одного сэмпла.
        """
        img\_name \= os.path.join(self.root\_dir, self.image\_files\[idx\])

        try:
            \# Открытие изображения
            \#.convert('RGB') важен, так как PNG может иметь 4 канала (RGBA),
            \# а черно-белые JPEG \- 1 канал (L). Модели обычно ждут 3 канала.
            image \= Image.open(img\_name).convert('RGB')

            \# Применение трансформаций
            if self.transform:
                image \= self.transform(image)

            \# В реальной задаче здесь также загружалась бы метка (label)
            \# Для примера вернем фиктивную метку 0
            label \= 0

            return image, label

        except (IOError, OSError, SyntaxError) as e:
            \# Оптимизация 2: Обработка битых файлов.
            \# Вместо падения программы, мы логируем ошибку и возвращаем None.
            \# None будет отфильтрован в collate\_fn.
            print(f"CORRUPTED FILE DETECTED: {img\_name}. Error: {e}")
            return None

### **7.3 Этап 2: Реализация кастомного collate_fn**

Стандартный коллейтор упадет при встрече None. Напишем свой.

Python

def filter_none_collate(batch):  
 """  
 Кастомный collate_fn для фильтрации битых сэмплов (None).  
 """  
 \# batch \- это список кортежей \[(img1, lbl1), (img2, lbl2), None, (img4, lbl4)\]

    \# 1\. Фильтрация None
    batch \= \[item for item in batch if item is not None\]

    \# 2\. Обработка случая, когда весь батч оказался битым
    if len(batch) \== 0:
        \# Это редкий случай, но он может сломать обучение.
        \# Возвращаем пустые тензоры или None, но цикл обучения должен это учитывать.
        return torch.tensor(), torch.tensor()

    \# 3\. Использование стандартного collate для оставшихся элементов
    return torch.utils.data.dataloader.default\_collate(batch)

Важное замечание по размеру батча:  
Данный подход приводит к динамическому размеру батча. Если batch_size=32 и 2 файла битые, модель получит 30 примеров. Обычно это допустимо (так как усреднение лосса идет по батчу), но если жестко требуется фиксированный размер (например, в TPU), нужно использовать более сложную стратегию "Retry" внутри Dataset (рекурсивно брать другой индекс), однако это чревато бесконечной рекурсией и искажением распределения данных.19

### **7.4 Этап 3: Сборка DataLoader и запуск**

Настраиваем параметры для максимальной производительности.

Python

from torchvision import transforms

\# Определение аугментаций  
data_transforms \= transforms.Compose(, std=\[0.229, 0.224, 0.225\])  
\])

\# Создание экземпляра Dataset  
dataset \= CustomImageDataset(root_dir='./data/train', transform=data_transforms)

\# Конфигурация DataLoader  
\# num_workers: кол-во ядер CPU  
\# pin_memory: True (обязательно для GPU)  
\# drop_last: True (отбрасываем неполный последний батч для стабильности BatchNorm)  
dataloader \= DataLoader(  
 dataset,  
 batch_size=64,  
 shuffle=True,  
 num_workers=4,  
 pin_memory=True,  
 collate_fn=filter_none_collate,  
 drop_last=True  
)

\# Цикл обучения  
device \= torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Запуск цикла обучения...")  
for epoch in range(2):  
 for batch_idx, (images, labels) in enumerate(dataloader):  
 \# Проверка на пустой батч (если collate вернул пустые тензоры)  
 if len(images) \== 0:  
 continue

        \# Асинхронный перенос на GPU (благодаря pin\_memory это быстро)
        images \= images.to(device, non\_blocking=True)
        labels \= labels.to(device, non\_blocking=True)

        \#... forward pass, loss, backward...
        \# print(f"Batch {batch\_idx}: {images.shape}")

## **8\. Продвинутые техники оптимизации**

### **8.1 Инициализация воркеров (worker_init_fn)**

При использовании библиотек NumPy или стандартного random внутри воркеров DataLoader возникает проблема: при форке состояние генератора случайных чисел (RNG) копируется. В результате каждый воркер может генерировать _одинаковую_ последовательность случайных чисел для аугментаций.  
PyTorch автоматически заботится о сидировании (seeding) своего внутреннего RNG, но для NumPy нужно делать это вручную.

Python

def worker_init_fn(worker_id):  
 \# Получаем базовый сид из PyTorch (который уже уникален для воркера)  
 worker_seed \= torch.initial_seed() % 2\*\*32  
 \# Устанавливаем сид для NumPy  
 np.random.seed(worker_seed)  
 \# Устанавливаем сид для Python random  
 import random  
 random.seed(worker_seed)

Передавайте эту функцию в DataLoader(..., worker_init_fn=worker_init_fn).13

### **8.2 Профилирование узких мест**

Как понять, где тормозит загрузка? Использовать torch.profiler.

Python

with torch.profiler.profile(  
 activities=,  
 schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),  
 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler')  
) as p:  
 for iter, batch in enumerate(dataloader):  
 train_step(batch)  
 p.step()

В TensorBoard во вкладке "Trace" можно увидеть полосы "DataLoader". Если они длинные и GPU простаивает — нужно оптимизировать \_\_getitem\_\_ или увеличивать num_workers.

## **9\. Заключение**

Создание собственного загрузчика в PyTorch — это балансировка между гибкостью Python и жесткими требованиями производительности железа.  
Для успешного выполнения "Квеста 15.1" и построения профессиональных систем следует придерживаться иерархии решений:

1. **Базовый уровень:** Map-style dataset с ленивой загрузкой и os.scandir.
2. **Средний уровень:** Использование Albumentations, правильная настройка num_workers и pin_memory, обработка ошибок через collate_fn.
3. **Высокий уровень:** Переход на форматы WebDataset или LMDB, использование Memory Mapping, профилирование I/O.

Понимание того, как данные проходят путь от магнитного домена жесткого диска до тензорного ядра GPU, отличает специалиста по Data Science от инженера ML-систем.

#### **Источники**

1. PyTorch: How to use DataLoaders for custom Datasets \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets](https://stackoverflow.com/questions/41924453/pytorch-how-to-use-dataloaders-for-custom-datasets)
2. Datasets & DataLoaders — PyTorch Tutorials 2.9.0+cu128 documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
3. Writing Custom Datasets, DataLoaders and Transforms \- PyTorch documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html)
4. A clear explanation of what num_workers=0 means for a DataLoader \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/a-clear-explanation-of-what-num-workers-0-means-for-a-dataloader/177614](https://discuss.pytorch.org/t/a-clear-explanation-of-what-num-workers-0-means-for-a-dataloader/177614)
5. How to effectively load a large text dataset with PyTorch? : r/datascience \- Reddit, дата последнего обращения: декабря 20, 2025, [https://www.reddit.com/r/datascience/comments/q8x82t/how_to_effectively_load_a_large_text_dataset_with/](https://www.reddit.com/r/datascience/comments/q8x82t/how_to_effectively_load_a_large_text_dataset_with/)
6. PyTorch DataLoader Tutorial: Master torch.utils.data.DataLoader | Codecademy, дата последнего обращения: декабря 20, 2025, [https://www.codecademy.com/article/how-to-use-pytorch-dataloader-custom-datasets-transformations-and-efficient-techniques](https://www.codecademy.com/article/how-to-use-pytorch-dataloader-custom-datasets-transformations-and-efficient-techniques)
7. Set of 10-char strings in Python is 10 times bigger in RAM as expected \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/48092816/set-of-10-char-strings-in-python-is-10-times-bigger-in-ram-as-expected](https://stackoverflow.com/questions/48092816/set-of-10-char-strings-in-python-is-10-times-bigger-in-ram-as-expected)
8. Memory usage of a list of millions of strings in Python \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/71233311/memory-usage-of-a-list-of-millions-of-strings-in-python](https://stackoverflow.com/questions/71233311/memory-usage-of-a-list-of-millions-of-strings-in-python)
9. Understand How Much Memory Your Python Objects Use | Envato Tuts+, дата последнего обращения: декабря 20, 2025, [https://code.tutsplus.com/understand-how-much-memory-your-python-objects-use--cms-25609t](https://code.tutsplus.com/understand-how-much-memory-your-python-objects-use--cms-25609t)
10. PyTorch Datasets: When to use Map-Style vs. Iterable-Style | by hebiao064 | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@stefanhebuaa/pytorch-datasets-when-to-use-map-style-vs-iterable-style-894b3f8a7465](https://medium.com/@stefanhebuaa/pytorch-datasets-when-to-use-map-style-vs-iterable-style-894b3f8a7465)
11. Differences between Dataset and IterableDataset \- Hugging Face, дата последнего обращения: декабря 20, 2025, [https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable](https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable)
12. Dataset map-style vs iterable-style \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/dataset-map-style-vs-iterable-style/92329](https://discuss.pytorch.org/t/dataset-map-style-vs-iterable-style/92329)
13. Iterable pytorch dataset with multiple workers \- distributed, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/iterable-pytorch-dataset-with-multiple-workers/135475](https://discuss.pytorch.org/t/iterable-pytorch-dataset-with-multiple-workers/135475)
14. Incorrect batch-size when using IterableDataset \+ num_workers \> 0 \#44108 \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/pytorch/pytorch/issues/44108](https://github.com/pytorch/pytorch/issues/44108)
15. Use with PyTorch \- Hugging Face, дата последнего обращения: декабря 20, 2025, [https://huggingface.co/docs/datasets/use_with_pytorch](https://huggingface.co/docs/datasets/use_with_pytorch)
16. torch.utils.data.DataLoader \- PyTorch documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/docs/stable/data.html](https://docs.pytorch.org/docs/stable/data.html)
17. How to use 'collate_fn' with dataloaders? \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders](https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders)
18. How to use collate_fn() \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/how-to-use-collate-fn/27181](https://discuss.pytorch.org/t/how-to-use-collate-fn/27181)
19. Handling corrupted data in Pytorch Dataloader \- Vivek Maskara, дата последнего обращения: декабря 20, 2025, [https://www.maskaravivek.com/post/pytorch-dataloader-with-corrupted-data/](https://www.maskaravivek.com/post/pytorch-dataloader-with-corrupted-data/)
20. DataLoader for various length of data \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418](https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418)
21. Efficient PyTorch I/O library for Large Datasets, Many Files, Many GPUs, дата последнего обращения: декабря 20, 2025, [https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/](https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/)
22. Most efficient way to use a large data set for PyTorch? \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/53576113/most-efficient-way-to-use-a-large-data-set-for-pytorch](https://stackoverflow.com/questions/53576113/most-efficient-way-to-use-a-large-data-set-for-pytorch)
23. \[P\] H5Records : Store large datasets in one single files with index access \- Reddit, дата последнего обращения: декабря 20, 2025, [https://www.reddit.com/r/MachineLearning/comments/nsq3ai/p_h5records_store_large_datasets_in_one_single/](https://www.reddit.com/r/MachineLearning/comments/nsq3ai/p_h5records_store_large_datasets_in_one_single/)
24. In Spark, is there a performance difference between querying DataFrames on CSV and JSON \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/33509619/in-spark-is-there-a-performance-difference-between-querying-dataframes-on-csv-a](https://stackoverflow.com/questions/33509619/in-spark-is-there-a-performance-difference-between-querying-dataframes-on-csv-a)
25. Parquet Data Format: Exploring Its Pros and Cons for 2025 \- EdgeDelta, дата последнего обращения: декабря 20, 2025, [https://edgedelta.com/company/blog/parquet-data-format](https://edgedelta.com/company/blog/parquet-data-format)
26. CSV vs Parquet vs JSON for Data Science | by Stephen \- Medium, дата последнего обращения: декабря 20, 2025, [https://weber-stephen.medium.com/csv-vs-parquet-vs-json-for-data-science-cf3733175176](https://weber-stephen.medium.com/csv-vs-parquet-vs-json-for-data-science-cf3733175176)
27. How to boost PyTorch Dataset using memory-mapped files | Towards Data Science, дата последнего обращения: декабря 20, 2025, [https://towardsdatascience.com/how-to-boost-pytorch-dataset-using-memory-mapped-files-6893bff27b99/](https://towardsdatascience.com/how-to-boost-pytorch-dataset-using-memory-mapped-files-6893bff27b99/)
28. Dataloader and memmaps \- data \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/dataloader-and-memmaps/180614](https://discuss.pytorch.org/t/dataloader-and-memmaps/180614)
29. What's the fastest way to save/load a large collection (list/set) of strings in Python 3.6?, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/51181556/whats-the-fastest-way-to-save-load-a-large-collection-list-set-of-strings-in](https://stackoverflow.com/questions/51181556/whats-the-fastest-way-to-save-load-a-large-collection-list-set-of-strings-in)
30. Doubling PyTorch Image Augmentation Speed \[With Code\] \- Ruman \- Medium, дата последнего обращения: декабря 20, 2025, [https://rumn.medium.com/doubling-pytorch-image-augmentation-speed-with-code-c8e95546f6ad](https://rumn.medium.com/doubling-pytorch-image-augmentation-speed-with-code-c8e95546f6ad)
31. We switched from Pillow to Albumentations and got 2x speedup \- Lightly AI, дата последнего обращения: декабря 20, 2025, [https://www.lightly.ai/blog/we-switched-from-pillow-to-albumentations-and-got-2x-speedup](https://www.lightly.ai/blog/we-switched-from-pillow-to-albumentations-and-got-2x-speedup)
32. Replacing Torchivision with Albumentations Transforms is Lowering Performance, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/replacing-torchivision-with-albumentations-transforms-is-lowering-performance/166987](https://discuss.pytorch.org/t/replacing-torchivision-with-albumentations-transforms-is-lowering-performance/166987)
33. pytorch collate_fn reject sample and yield another \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another](https://stackoverflow.com/questions/57815001/pytorch-collate-fn-reject-sample-and-yield-another)
34. PyTorch Dataset. Table of Contents | by ifeelfree \- Medium, дата последнего обращения: декабря 20, 2025, [https://majianglin2003.medium.com/pytorch-dataset-1b7273152f82](https://majianglin2003.medium.com/pytorch-dataset-1b7273152f82)
