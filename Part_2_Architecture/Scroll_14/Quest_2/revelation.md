# **Архитектура Универсальности: Исчерпывающее руководство по стандарту ONNX и среде выполнения ONNX Runtime для промышленного развертывания нейронных сетей**

## **1\. Введение: От «Вавилонской башни» фреймворков к единому стандарту**

В современной экосистеме глубокого обучения наблюдается фундаментальная дихотомия между средами разработки и средами исполнения. Исследователи и инженеры данных создают модели — аллегорических «Големов» — в гибких, динамических фреймворках, таких как PyTorch, которые приоритезируют удобство отладки и эвристическую свободу. Однако промышленное внедрение (deployment) требует жесткости, предсказуемости и максимальной производительности на целевом оборудовании, будь то облачные серверы с GPU, мобильные устройства или встраиваемые системы Интернета вещей (IoT).  
Исторически этот разрыв приводил к фрагментации: модель, обученная в PyTorch, требовала полной переписывания кода для запуска на мобильном устройстве или конвертации через проприетарные инструменты производителей чипов. Появление **Open Neural Network Exchange (ONNX)** стало ответом на этот вызов интероперабельности. ONNX — это не просто формат файла, а открытый стандарт, определяющий универсальное промежуточное представление (Intermediate Representation, IR) для моделей машинного обучения. Он позволяет сериализовать вычислительный граф, созданный в любом поддерживаемом фреймворке, в статический артефакт — «Свиток ONNX», который может быть интерпретирован и исполнен высокопроизводительным движком **ONNX Runtime (ORT)**.1  
Данный отчет представляет собой глубокое техническое исследование экосистемы ONNX. Мы детально разберем механизмы экспорта моделей из PyTorch, архитектурные особенности графа ONNX, стратегии управления динамическими размерностями, а также передовые методы оптимизации, включая квантование и слияние операторов, необходимые для эффективного запуска моделей через ONNX Runtime на различных аппаратных ускорителях.3

## ---

**2\. Архитектура и Структура Стандарта ONNX**

### **2.1. Фундаментальные принципы построения графа**

В основе ONNX лежит концепция вычислительного графа (computation graph), который, в отличие от динамических графов PyTorch (eager execution), является статическим и топологически упорядоченным. Модель ONNX описывается с использованием **Protocol Buffers** (protobuf) — расширяемого механизма сериализации структурированных данных, разработанного Google. Это обеспечивает бинарную компактность и кросс-языковую совместимость «свитка».5  
Структура файла .onnx иерархична:

- **ModelProto:** Верхнеуровневый контейнер, содержащий метаданные (версия IR, производитель, домен) и сам граф.
- **GraphProto:** Определяет топологию сети. Он содержит списки узлов (NodeProto), инициализаторов (TensorProto — веса и смещения) и описания входов/выходов (ValueInfoProto).
- **NodeProto:** Представляет отдельную операцию (например, Conv, Relu, MatMul). Каждый узел ссылается на имена тензоров входов и выходов, создавая связи данных (data dependencies) между операторами.

Важнейшим аспектом является то, что ONNX определяет **набор стандартных операторов** (Operator Set). Если PyTorch использует слой nn.Conv2d, при экспорте он транслируется в оператор ONNX Conv с атрибутами (kernel_shape, pads, strides), чье математическое поведение строго специфицировано стандартом.6 Это гарантирует, что свертка, выполненная в Python на Linux, даст тот же математический результат, что и на C++ в Windows.

### **2.2. Версионирование наборов операторов (Opset Versioning)**

Экосистема глубокого обучения развивается стремительно, и стандарт ONNX адаптируется через механизм версионирования наборов операторов (Opset). Каждая версия Opset добавляет новые операторы или изменяет спецификации существующих.

- **Эволюция:** Ранние версии (Opset 7-9) поддерживали базовые операции CNN. Opset 10+ ввели поддержку квантования (INT8). Opset 13-17 добавили сложные операторы для обработки аудио и трансформеров.8
- **Совместимость:** При экспорте модели из PyTorch необходимо явно указывать opset_version. Попытка экспортировать модель, использующую современные функции (например, специфические виды интерполяции или torch.fft), в старый Opset (например, 9\) приведет к ошибке, так как в «словаре» старого стандарта просто нет соответствующих слов для описания этих операций.10 Рекомендуется использовать стабильные версии (на текущий момент 14-17) для максимальной совместимости с различными версиями ONNX Runtime.10

## ---

**3\. Экспорт моделей из PyTorch: Процесс Трансмутации**

Процесс перевода модели из «Кузницы PyTorch» в формат ONNX осуществляется функцией torch.onnx.export. Это не просто конвертация, а компиляция, в ходе которой Python-функции транслируются в узлы графа ONNX. Существует два основных механизма экспорта: **Трассировка (Tracing)** и **Скриптинг (Scripting)**.

### **3.1. Механизм Трассировки (Tracing)**

По умолчанию torch.onnx.export использует трассировку. Этот метод работает путем запуска модели с **фиктивным входом** (dummy_input). Экспортер «наблюдает» за выполнением тензорных операций и записывает их последовательность.5

#### **Роль Dummy Input**

Поскольку граф PyTorch строится динамически во время исполнения (forward), экспортер не может знать структуру сети без прогона данных. dummy_input — это тензор правильной размерности и типа (обычно случайные числа), который подается в модель. Значения в нем не важны для структуры графа, важны лишь их свойства (shape, dtype).5

Python

import torch  
import torchvision

\# 1\. Загрузка предобученного Голема (модели)  
model \= torchvision.models.resnet18(pretrained=True)

\# 2\. Перевод в режим инференса (Критически важно\!)  
model.eval()

\# 3\. Создание фиктивного входа (формат NCHW)  
dummy_input \= torch.randn(1, 3, 224, 224, device='cpu')

\# 4\. Экспорт (Трансмутация)  
torch.onnx.export(  
 model,  
 dummy_input,  
 "resnet18.onnx",  
 verbose=True,  
 input_names=\['input_image'\],  
 output_names=\['classification_scores'\],  
 opset_version=14  
)

**Важность model.eval():** Перед экспортом необходимо перевести модель в режим оценки. Слои, такие как Dropout и BatchNorm, ведут себя по-разному при обучении и инференсе. Экспорт в режиме train приведет к тому, что в граф ONNX попадут ненужные операторы (или Dropout не будет исключен), что исказит результаты предсказаний.15

### **3.2. Ограничения Трассировки и Решение через Скриптинг**

Трассировка имеет фундаментальный недостаток: она «видит» только тот путь исполнения кода, который был пройден с конкретным dummy_input.

- **Проблема Control Flow:** Если в модели есть логика if x.sum() \> 0:, трассировщик запишет только одну ветку (True или False), жестко зашив её в граф. Вторая ветка будет отброшена навсегда. Это делает трассировку непригодной для моделей с логикой, зависящей от данных.17
- **Решение — Scripting:** Для сохранения логических конструкций (циклов, условий) используется torch.jit.script. Этот инструмент компилирует Python-код модели в промежуточное представление TorchScript, которое затем может быть конвертировано в ONNX. Скриптинг требует, чтобы код модели соответствовал строгому подмножеству Python (статическая типизация), что часто требует рефакторинга кода.5

В последних версиях PyTorch (2.0+) внедряется новый движок torch.onnx.dynamo_export, который использует технологию Dynamo для захвата графа. Он обещает объединить преимущества обоих подходов, анализируя байт-код Python для построения полного графа без недостатков слепой трассировки, однако этот инструмент все еще находится в стадии активной стабилизации.19

## ---

**4\. Управление Динамическими Осями (Dynamic Axes)**

Одной из самых частых ошибок при экспорте является фиксация размерностей входа. При трассировке с dummy_input размером (1, 3, 224, 224\) ONNX граф по умолчанию жестко фиксирует размер пакета (batch size) равным 1\. Если попытаться подать на вход такого «свитка» пакет из 4 изображений, ONNX Runtime выдаст ошибку несоответствия форм.14  
Для создания универсальной модели необходимо явно указать, какие оси тензоров являются динамическими.

### **Таблица 1: Конфигурация динамических осей**

| Параметр         | Описание                                                                                         | Пример использования                                                         |
| :--------------- | :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **Ключ словаря** | Имя входа или выхода, определенное в input_names/output_names.                                   | 'input_image', 'output_scores'                                               |
| **Значение**     | Словарь, отображающий индекс оси в понятное имя.                                                 | {0: 'batch_size', 2: 'height', 3: 'width'}                                   |
| **Результат**    | В графе ONNX вместо числа (например, 1\) будет стоять символ (строка), разрешающий любой размер. | Позволяет обрабатывать батчи любого размера или картинки разного разрешения. |

**Пример кода:**

Python

dynamic_axes_config \= {  
 'input_image': {0: 'batch_size'}, \# Ось 0 (batch) может меняться  
 'classification_scores': {0: 'batch_size'}  
}

torch.onnx.export(  
 model,  
 dummy_input,  
 "resnet_dynamic.onnx",  
 input_names=\['input_image'\],  
 output_names=\['classification_scores'\],  
 dynamic_axes=dynamic_axes_config \# \<-- Магия гибкости  
)

Правильное использование dynamic_axes 12 позволяет одной и той же модели обрабатывать как единичные запросы с камеры смартфона, так и пакетную обработку на сервере, что является ключевым требованием квеста на «универсальность».

## ---

**5\. ONNX Runtime (ORT): Движок Исполнения**

После того как модель конвертирована в ONNX, она становится независимой от PyTorch. Для её запуска используется **ONNX Runtime (ORT)** — кросс-платформенный высокопроизводительный движок. ORT берет на себя роль «виртуальной машины», которая читает граф операций и исполняет его, оптимально используя доступное оборудование.3

### **5.1. Сессия Инференса (InferenceSession)**

Центральным объектом в ORT является InferenceSession. При его инициализации происходит загрузка модели, проверка целостности графа, выделение памяти и применение оптимизаций.  
**Жизненный цикл сессии:**

1. **Загрузка:** Парсинг protobuf файла.
2. **Оптимизация графа:** ORT применяет ряд трансформаций (см. раздел 6).
3. **Партиционирование:** Граф разбивается на подграфы в зависимости от доступных провайдеров исполнения (например, часть на GPU, часть на CPU).
4. **Исполнение:** Метод run() принимает данные (в виде numpy массивов или OrtValue) и возвращает результаты.

Python

import onnxruntime as ort  
import numpy as np

\# Инициализация сессии (Загрузка Свитка)  
\# Указываем приоритет провайдеров: сначала GPU (CUDA), если нет \- CPU  
providers \=  
session \= ort.InferenceSession("resnet_dynamic.onnx", providers=providers)

\# Подготовка данных  
input_name \= session.get_inputs().name  
input_data \= np.random.randn(1, 3, 224, 224).astype(np.float32)

\# Запуск инференса  
outputs \= session.run(None, {input_name: input_data})  
print("Результат получен:", outputs.shape)

Важно отметить, что инициализация сессии — тяжелая операция. В продакшн-системах объект session создается один раз при старте сервиса и переиспользуется для обработки множества запросов.21

### **5.2. Провайдеры Исполнения (Execution Providers)**

Главная сила ONNX Runtime — в модульной архитектуре **Execution Providers (EP)**. ORT не пытается реализовать все ядра операций для всех устройств самостоятельно. Вместо этого он делегирует выполнение специализированным библиотекам.23

#### **Таблица 2: Основные Провайдеры Исполнения и их Применение**

| Провайдер (EP)                | Целевое оборудование         | Библиотека-бэкенд | Сценарий использования                                                                                |
| :---------------------------- | :--------------------------- | :---------------- | :---------------------------------------------------------------------------------------------------- |
| **CPUExecutionProvider**      | Любой CPU (Intel, AMD, ARM)  | MLAS / Eigen      | Базовый, универсальный запуск. Высокая переносимость.                                                 |
| **CUDAExecutionProvider**     | NVIDIA GPU                   | cuDNN / cuBLAS    | Стандарт для серверного инференса. Высокая пропускная способность.                                    |
| **TensorRTExecutionProvider** | NVIDIA GPU                   | TensorRT          | Максимальная производительность на NVIDIA GPU. Требует компиляции движка, но дает выигрыш в задержке. |
| **OpenVINOExecutionProvider** | Intel CPU/iGPU/VPU           | OpenVINO          | Оптимизация под архитектуру Intel (AVX-512, iGPU).                                                    |
| **NNAPIExecutionProvider**    | Android устройства           | Android NNAPI     | Аппаратное ускорение на мобильных чипах (Snapdragon, MediaTek) через NPU/DSP.                         |
| **CoreMLExecutionProvider**   | Apple устройства (iOS/macOS) | CoreML            | Использование Apple Neural Engine (ANE) для энергоэффективного инференса.                             |
| **DirectMLExecutionProvider** | Windows PC                   | DirectX 12        | Ускорение на любых GPU (NVIDIA, AMD, Intel) в среде Windows.                                          |

При создании сессии разработчик передает список провайдеров. ORT пытается использовать первый в списке; если оператор не поддерживается этим провайдером (например, специфическая операция не реализована в NPU), происходит автоматический откат (fallback) к следующему провайдеру (обычно CPU).23

## ---

**6\. Оптимизация Производительности: От Грамматики к Поэзии**

Простая конвертация в ONNX часто дает прирост скорости за счет более легкого runtime (отсутствие Python-оверхеда, C++ реализация). Однако для раскрытия полного потенциала используются продвинутые техники оптимизации.

### **6.1. Слияние Операторов (Graph Fusion)**

ONNX Runtime автоматически применяет несколько уровней оптимизации графа при загрузке 25:

1. **Basic:** Удаление избыточных узлов (Identity, Dropout), сворачивание констант (Constant Folding — предварительное вычисление веток графа, зависящих только от констант).
2. **Extended (Fusion):** Объединение нескольких мелких операций в одно ядро (Kernel Fusion).

Классический пример — слияние свертки (Convolution), нормализации батча (BatchNorm) и активации (ReLU).  
В наивном исполнении это три отдельных прохода:

1. Чтение данных \-\> Свертка \-\> Запись в память.
2. Чтение \-\> Нормализация \-\> Запись.
3. Чтение \-\> ReLU \-\> Запись.

В режиме Fusion это превращается в **Conv+BN+ReLU**: параметры BatchNorm «впекаются» в веса свертки (так как на инференсе они константы), а ReLU применяется прямо в регистрах процессора/GPU перед записью результата. Это радикально снижает нагрузку на пропускную способность памяти (memory bandwidth), что критично для скорости инференса.27

### **6.2. Квантование (Quantization)**

Квантование — это процесс снижения точности весов и вычислений с плавающей точки (FP32) до целых чисел (INT8). Это уменьшает размер модели в 4 раза и ускоряет вычисления в 2-4 раза на поддерживаемом оборудовании (например, используя инструкции VNNI на Intel или Tensor Cores на NVIDIA).29  
Существует два основных подхода:

- **Динамическое квантование (Dynamic Quantization):** Веса квантуются заранее, а активации (промежуточные данные) квантуются «на лету» во время инференса.
  - _Применение:_ Идеально для **NLP моделей (BERT, LSTM, Transformer)**, где вычислительная нагрузка сосредоточена в слоях MatMul (умножение матриц), а диапазоны активаций сильно зависят от входного текста.31
- **Статическое квантование (Static Quantization):** И веса, и активации имеют заранее вычисленные параметры масштабирования (scale и zero-point). Для этого требуется этап **калибровки**: модель прогоняется на небольшом наборе данных, чтобы измерить диапазоны значений активаций.
  - _Применение:_ Критически важно для **Компьютерного зрения (CNN, ResNet, YOLO)**. Поскольку активации в сверточных сетях занимают много памяти, вычислять их масштаб динамически слишком дорого. Статика дает лучший прирост скорости.33

Формула квантования:

$$Val\_{fp32} \= Scale \\times (Val\_{quantized} \- Zero\\\_Point)$$

Где $Val\_{quantized}$ — значение int8 или uint8.

## ---

**7\. Сценарии Развертывания: Практическая Магия**

Универсальность ONNX позволяет развертывать модели в средах, радикально отличающихся от Python-окружения разработчика.

### **7.1. Мобильная разработка: Android (Kotlin) и NNAPI**

Запуск тяжелых нейросетей на телефоне требует использования NPU. Через ONNX Runtime это достигается подключением библиотеки onnxruntime-android.  
Алгоритм интеграции 34:

1. Добавить зависимость в build.gradle.
2. Поместить файл .onnx в папку assets.
3. Инициализировать среду с флагом NNAPI:

Kotlin

// Пример на Kotlin  
val sessionOptions \= OrtSession.SessionOptions()  
sessionOptions.addNnapi() // Включаем аппаратное ускорение Android NNAPI  
val env \= OrtEnvironment.getEnvironment()  
// Чтение модели и создание сессии  
val session \= env.createSession(readModelFromAssets(), sessionOptions)

Использование NNAPI позволяет делегировать вычисления DSP или NPU процессора (Snapdragon, MediaTek), разгружая CPU и экономя батарею.35

### **7.2. iOS и CoreML**

Для экосистемы Apple используется провайдер CoreMLExecutionProvider. Он позволяет задействовать Apple Neural Engine (ANE). Следует помнить, что CoreML имеет более ограниченный набор операторов, чем полный стандарт ONNX. Если модель содержит несовместимые операторы, ORT автоматически переключится на CPU, что может быть медленнее. Экспорт и тестирование на совместимость операторов критичны для iOS.37

### **7.3. Веб-приложения: ONNX Runtime Web**

onnxruntime-web позволяет запускать модели прямо в браузере клиента, используя **WebAssembly (WASM)** для CPU-вычислений и **WebGL/WebGPU** для ускорения графическим процессором. Это открывает возможности для создания privacy-first приложений, где данные (например, видео с веб\-камеры) никогда не покидают устройство пользователя, а обработка происходит локально с производительностью, близкой к нативной.39

## ---

**8\. Диагностика и Устранение Проблем**

Даже при наличии стандарта, перенос моделей — сложный процесс. Рассмотрим типичные проблемы.

### **8.1. Конфликты версий Opset и Ошибки Экспорта**

Частая ошибка: RuntimeError: Exporting the operator... to ONNX opset version X is not supported.

- _Причина:_ Вы используете функцию PyTorch, которая была добавлена в стандарт ONNX только в новой версии (например, torch.fft в Opset 17\) или вообще не имеет прямого аналога.
- _Решение:_ Повысить opset_version в аргументах export до 14 или 17\. Если оператор нестандартный (например, из кастомной CUDA-вставки), потребуется написать **символическую функцию (symbolic function)** регистрации оператора.10

### **8.2. Потеря информации о формах (Shape Inference)**

Иногда экспортированная модель имеет тензоры с неизвестными размерностями (обозначаются как ? или None в визуализаторах типа Netron). Это мешает ORT оптимизировать память.

- _Решение:_ Использовать утилиту onnx.shape_inference для явного прогона вывода форм после экспорта.14

### **8.3. Проблемы производительности**

«Почему ONNX медленнее PyTorch?» — распространенный вопрос.

- _Причина:_ Чаще всего это происходит на очень маленьких моделях, где накладные расходы на вызов C++ API из Python превышают время самого вычисления.
- _Вторая причина:_ Неэффективная структура графа после трассировки циклов (разворачивание цикла for на 100 итераций в 100 отдельных узлов графа). В таких случаях нужно переходить на Scripting.13

## ---

**9\. Заключение**

Выполнение Квеста 14.2 по экспорту в «Свиток ONNX» открывает перед разработчиком двери в мир настоящей кросс-платформенной разработки. ONNX перестает быть просто форматом файла и становится «lingua franca» — универсальным языком, позволяющим идеям, рожденным в исследовательских лабораториях на PyTorch, мгновенно материализоваться в приложениях на миллиардах устройств по всему миру.  
Освоение torch.onnx, понимание нюансов динамических осей и грамотный выбор провайдеров исполнения в ONNX Runtime превращает хрупкие исследовательские прототипы в надежные промышленные решения. В эпоху повсеместного ИИ, способность эффективно «переводить» и оптимизировать модели является столь же важным навыком, как и умение их обучать.

### ---

**Приложение: Сравнительная таблица методов экспорта**

| Характеристика              | Tracing (torch.onnx.export)                  | Scripting (torch.jit.script)    | Dynamo (torch.onnx.dynamo_export) |
| :-------------------------- | :------------------------------------------- | :------------------------------ | :-------------------------------- |
| **Поддержка Control Flow**  | Нет (только статика)                         | Да (полная)                     | Да (Python байт-код)              |
| **Сложность использования** | Низкая (работает "из коробки" для CNN)       | Средняя (требует типизации)     | Высокая (пока в бета-версии)      |
| **Надежность**              | Средняя (риск "тихих" ошибок)                | Высокая                         | Высокая                           |
| **Рекомендация**            | Стандарт для Vision/NLP (без сложной логики) | Для сложной логики и продакшена | Будущий стандарт PyTorch 2.x      |

5

#### **Источники**

1. дата последнего обращения: декабря 20, 2025, [https://www.unitxlabs.com/onnx-open-neural-network-exchange-machine-vision-system-ai/\#:\~:text=Open%20Neural%20Network%20Exchange%20Overview,and%20use%20them%20in%20another.](https://www.unitxlabs.com/onnx-open-neural-network-exchange-machine-vision-system-ai/#:~:text=Open%20Neural%20Network%20Exchange%20Overview,and%20use%20them%20in%20another.)
2. Open Neural Network Exchange (ONNX) Explained \- Splunk, дата последнего обращения: декабря 20, 2025, [https://www.splunk.com/en_us/blog/learn/open-neural-network-exchange-onnx.html](https://www.splunk.com/en_us/blog/learn/open-neural-network-exchange-onnx.html)
3. ONNX | Home, дата последнего обращения: декабря 20, 2025, [https://onnx.ai/](https://onnx.ai/)
4. Unlocking the Power of ONNX: Model Interoperability and Boosting Performance \- Comet, дата последнего обращения: декабря 20, 2025, [https://www.comet.com/site/blog/unlocking-the-power-of-onnx-model-interoperability-and-boosting-performance/](https://www.comet.com/site/blog/unlocking-the-power-of-onnx-model-interoperability-and-boosting-performance/)
5. torch.onnx — PyTorch master documentation, дата последнего обращения: декабря 20, 2025, [https://glaringlee.github.io/onnx.html](https://glaringlee.github.io/onnx.html)
6. onnx/onnx: Open standard for machine learning interoperability \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/onnx/onnx](https://github.com/onnx/onnx)
7. Conv2d — PyTorch 2.9 documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
8. ONNX Runtime compatibility, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/reference/compatibility.html](https://onnxruntime.ai/docs/reference/compatibility.html)
9. \[ONNX\] Missing support opset 17 operators · Issue \#80834 \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/pytorch/pytorch/issues/80834](https://github.com/pytorch/pytorch/issues/80834)
10. Which opset version is supported by torch.oonnx \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/77180659/which-opset-version-is-supported-by-torch-oonnx](https://stackoverflow.com/questions/77180659/which-opset-version-is-supported-by-torch-oonnx)
11. ValueError: Unsupported ONNX opset version: 21 \- deployment \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/valueerror-unsupported-onnx-opset-version-21/221976](https://discuss.pytorch.org/t/valueerror-unsupported-onnx-opset-version-21/221976)
12. torch.export-based ONNX Exporter — PyTorch 2.9 documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/docs/stable/onnx_export.html](https://docs.pytorch.org/docs/stable/onnx_export.html)
13. TorchScript: Tracing vs. Scripting \- Yuxin's Blog, дата последнего обращения: декабря 20, 2025, [https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/](https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/)
14. Dynamic dummy input when exporting a PyTorch model? · Issue \#654 \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/onnx/onnx/issues/654](https://github.com/onnx/onnx/issues/654)
15. Convert your PyTorch training model to ONNX \- Microsoft Learn, дата последнего обращения: декабря 20, 2025, [https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model](https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model)
16. (optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime, дата последнего обращения: декабря 20, 2025, [https://h-huang.github.io/tutorials/advanced/super_resolution_with_onnxruntime.html](https://h-huang.github.io/tutorials/advanced/super_resolution_with_onnxruntime.html)
17. When should I use Tracing rather than Scripting? \- jit \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/when-should-i-use-tracing-rather-than-scripting/53883](https://discuss.pytorch.org/t/when-should-i-use-tracing-rather-than-scripting/53883)
18. With ONNX export, does it mean it will avoid the problem when not using jit.script in torch.trace? \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/with-onnx-export-does-it-mean-it-will-avoid-the-problem-when-not-using-jit-script-in-torch-trace/100405](https://discuss.pytorch.org/t/with-onnx-export-does-it-mean-it-will-avoid-the-problem-when-not-using-jit-script-in-torch-trace/100405)
19. torch.onnx — PyTorch 2.9 documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/docs/stable/onnx.html](https://docs.pytorch.org/docs/stable/onnx.html)
20. onnx export error · Issue \#107922 \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/pytorch/pytorch/issues/107922](https://github.com/pytorch/pytorch/issues/107922)
21. Python \- ONNX Runtime, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/get-started/with-python.html](https://onnxruntime.ai/docs/get-started/with-python.html)
22. ONNX Runtime Performance Tuning, дата последнего обращения: декабря 20, 2025, [https://iot-robotics.github.io/ONNXRuntime/docs/performance/tune-performance.html](https://iot-robotics.github.io/ONNXRuntime/docs/performance/tune-performance.html)
23. ONNX Runtime Execution Providers, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/execution-providers/](https://onnxruntime.ai/docs/execution-providers/)
24. Select execution providers using the ONNX Runtime included in Windows ML, дата последнего обращения: декабря 20, 2025, [https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers)
25. Graph Optimizations in ONNX Runtime, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
26. A Deep Dive into ONNX & ONNX Runtime (Part 2\) | by Mohsen Mahmoodzadeh, дата последнего обращения: декабря 20, 2025, [https://becominghuman.ai/a-deep-dive-into-onnx-onnx-runtime-part-2-785b523e0cca](https://becominghuman.ai/a-deep-dive-into-onnx-onnx-runtime-part-2-785b523e0cca)
27. How to optimize inference speed with ONNX Runtime? \- Tencent Cloud, дата последнего обращения: декабря 20, 2025, [https://www.tencentcloud.com/techpedia/126078](https://www.tencentcloud.com/techpedia/126078)
28. Faster Models with Graph Fusion: How Deep Learning Frameworks Optimize Your Computation | Practical ML, дата последнего обращения: декабря 20, 2025, [https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/)
29. Quantize ONNX Models \- ONNXRuntime \- GitHub Pages, дата последнего обращения: декабря 20, 2025, [https://iot-robotics.github.io/ONNXRuntime/docs/performance/quantization.html](https://iot-robotics.github.io/ONNXRuntime/docs/performance/quantization.html)
30. Quantize ONNX Models \- onnxruntime \- GitHub Pages, дата последнего обращения: декабря 20, 2025, [https://lenisha.github.io/onnxruntime/docs/how-to/quantization.html](https://lenisha.github.io/onnxruntime/docs/how-to/quantization.html)
31. Onnx Model Quantization | by Nashrakhan \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@nashrakhan1008/model-quantization-8f10c537e0eb](https://medium.com/@nashrakhan1008/model-quantization-8f10c537e0eb)
32. PyTorch to Quantized ONNX Model \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27](https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27)
33. Quantize ONNX models | onnxruntime, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
34. ONNX Runtime on Android: The Ultimate Guide to Lightning-Fast AI Inference \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/softaai-blogs/onnx-runtime-on-android-the-ultimate-guide-to-lightning-fast-ai-inference-097123814ee3](https://medium.com/softaai-blogs/onnx-runtime-on-android-the-ultimate-guide-to-lightning-fast-ai-inference-097123814ee3)
35. Running onnx with nnapi \- Android \- Khadas Community, дата последнего обращения: декабря 20, 2025, [https://forum.khadas.com/t/running-onnx-with-nnapi/19488](https://forum.khadas.com/t/running-onnx-with-nnapi/19488)
36. Build ONNX Runtime for Android, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/build/android.html](https://onnxruntime.ai/docs/build/android.html)
37. CoreML \- onnxruntime \- GitHub Pages, дата последнего обращения: декабря 20, 2025, [https://oliviajain.github.io/onnxruntime/docs/execution-providers/CoreML-ExecutionProvider.html](https://oliviajain.github.io/onnxruntime/docs/execution-providers/CoreML-ExecutionProvider.html)
38. Apple \- CoreML | onnxruntime, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
39. Web | onnxruntime, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/get-started/with-javascript/web.html](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
40. How to add machine learning to your web application with ONNX Runtime, дата последнего обращения: декабря 20, 2025, [https://onnxruntime.ai/docs/tutorials/web/](https://onnxruntime.ai/docs/tutorials/web/)
41. Loading ONNX Model in Java \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/47464416/loading-onnx-model-in-java](https://stackoverflow.com/questions/47464416/loading-onnx-model-in-java)
42. ONNX vs PyTorch Speed: In-Depth Performance Comparison \- Dev-kit, дата последнего обращения: декабря 20, 2025, [https://dev-kit.io/blog/machine-learning/onnx-vs-pytorch-speed-comparison](https://dev-kit.io/blog/machine-learning/onnx-vs-pytorch-speed-comparison)
