# **Архитектурная Эффективность в Edge AI: Глубокий Анализ Depthwise Separable Convolutions и Семейства MobileNet**

## **1\. Введение: Эра Ограниченных Ресурсов и Поиск "Экономичной Магии"**

В современном ландшафте искусственного интеллекта мы наблюдаем фундаментальный сдвиг парадигмы. Если последнее десятилетие характеризовалось гонкой за точностью, подпитываемой бесконечными ресурсами облачных дата-центров и массивными кластерами GPU, то сегодня вектор развития смещается в сторону периферийных вычислений (Edge AI). Смартфоны, автономные дроны, системы умного города и IoT-устройства требуют внедрения нейронных сетей непосредственно «на борту». Однако эти устройства жестко ограничены в ресурсах: емкости батареи, теплопакете, объеме памяти и вычислительной мощности. В терминологии "Школы Оптимизации", мы сталкиваемся с дефицитом "маны" — энергии и вычислительных циклов, необходимых для сотворения "заклинаний" компьютерного зрения.  
Традиционные архитектуры, такие как VGG или ранние ResNet, являются "жадными" потребителями ресурсов. Стандартная свертка (Standard Convolution), являющаяся их основным строительным блоком, обладает колоссальной избыточностью. Она пытается одновременно решать две задачи: выявлять пространственные паттерны (узоры) и комбинировать информацию из разных каналов (спектральные признаки). Этот подход, хоть и эффективен с точки зрения выразительной силы, крайне расточителен.  
Ответом на этот вызов стало появление **Depthwise Separable Convolution (Разделимой свертки по глубине)**. Этот архитектурный примитив, который мы будем именовать "Экономным блоком", совершил революцию в мобильном глубоком обучении. Разделив процесс свертки на два независимых этапа — пространственную фильтрацию ("работу гномов") и канальное смешивание ("алхимию"), — архитекторы смогли сократить количество параметров и операций умножения-сложения (MAdd) на порядок, практически не теряя в точности.  
Данный отчет представляет собой исчерпывающее исследование этого феномена. Мы не только реализуем этот блок с нуля, следуя ритуалу "Квеста 13.1", но и проведем глубокий теоретический анализ его работы, проследим эволюцию архитектур MobileNet (V1, V2, V3) и разберем сложные взаимоотношения между математической эффективностью (FLOPs) и реальной скоростью выполнения (Latency) на различных типах "железа" — от CPU до NPU.

## **2\. Теоретический Фундамент: Анатомия Свертки**

Для того чтобы оценить изящество решения Depthwise Separable Convolution, необходимо сначала препарировать стандартную свертку и понять природу ее "дороговизны".

### **2.1. Стандартная Свертка: Цена Универсальности**

Рассмотрим входной тензор (карту признаков) размером $D\_f \\times D\_f \\times M$, где $D\_f$ — пространственная размерность (высота и ширина), а $M$ — количество входных каналов. Мы хотим преобразовать его в выходной тензор размером $D\_f \\times D\_f \\times N$, где $N$ — количество выходных каналов.  
В стандартной свертке используется ядро размером $D\_k \\times D\_k \\times M \\times N$. Обратите внимание на множитель $M$: каждый из $N$ фильтров имеет глубину, равную глубине входного тензора. Это означает, что каждый фильтр взаимодействует со _всеми_ входными каналами одновременно.  
Вычислительная стоимость (в количестве операций умножения) для одного слоя стандартной свертки рассчитывается по формуле:

$$C\_{std} \= D\_k \\cdot D\_k \\cdot M \\cdot N \\cdot D\_f \\cdot D\_f$$  
Здесь кроется мультипликативный взрыв: стоимость растет пропорционально произведению количества входных и выходных каналов ($M \\cdot N$). Если мы имеем 512 входных и 512 выходных каналов (типичная ситуация для глубоких слоев), то количество операций становится огромным.  
Количество обучаемых параметров (весов) составляет:

$$P\_{std} \= D\_k \\cdot D\_k \\cdot M \\cdot N$$  
Именно эта зависимость $M \\times N$ делает стандартные свертки "тяжелыми" и "жадными". Они пытаются выучить корреляции одновременно в трех измерениях (ширина, высота, глубина), что является избыточным для многих задач.1

### **2.2. Философия Разделения (Decomposition)**

Идея Depthwise Separable Convolution заключается в гипотезе о том, что пространственные корреляции и канальные корреляции можно рассматривать независимо друг от друга. Этот подход, известный как факторизация, позволяет разбить одну "тяжелую" операцию на две "легкие".

#### **Этап 1: Depthwise Convolution (Глубинная свертка) — "Гномы-специалисты"**

На этом этапе мы занимаемся только пространственной фильтрацией. Для каждого из $M$ входных каналов мы применяем свой собственный фильтр размером $D\_k \\times D\_k \\times 1$. Важно, что этот фильтр не смотрит на соседние каналы — он работает изолированно со своим "слоем руды".  
Количество фильтров равно количеству входных каналов $M$. Выходной тензор имеет ту же глубину $M$.  
Вычислительная стоимость:

$$C\_{dw} \= D\_k \\cdot D\_k \\cdot M \\cdot D\_f \\cdot D\_f$$

Параметры: $D\_k \\cdot D\_k \\cdot M$.  
Здесь нет множителя $N$. Сложность растет линейно с количеством каналов, а не квадратично.

#### **Этап 2: Pointwise Convolution (Точечная свертка) — "Магистр-Алхимик"**

После того как "гномы" извлекли пространственные паттерны из каждого канала, необходимо объединить эту информацию. Эту задачу выполняет обычная свертка, но с ядром $1 \\times 1$. Она берет вектор значений из всех $M$ каналов в одной пространственной точке и смешивает их, создавая новые признаки.  
Количество фильтров равно желаемому количеству выходных каналов $N$. Размер ядра $1 \\times 1 \\times M$.  
Вычислительная стоимость:

$$C\_{pw} \= 1 \\cdot 1 \\cdot M \\cdot N \\cdot D\_f \\cdot D\_f$$

Параметры: $1 \\cdot 1 \\cdot M \\cdot N$.

#### **Суммарная эффективность**

Общая стоимость разделимой свертки ($C\_{sep}$) равна сумме стоимостей двух этапов:

$$C\_{sep} \= M \\cdot D\_f^2 \\cdot (D\_k^2 \+ N)$$  
Отношение стоимости разделимой свертки к стандартной дает нам коэффициент эффективности:

$$\\frac{C\_{sep}}{C\_{std}} \= \\frac{M \\cdot D\_f^2 \\cdot (D\_k^2 \+ N)}{D\_k^2 \\cdot M \\cdot N \\cdot D\_f^2} \= \\frac{D\_k^2 \+ N}{D\_k^2 \\cdot N} \= \\frac{1}{N} \+ \\frac{1}{D\_k^2}$$  
Для типичного случая, где размер ядра $D\_k \= 3$ (то есть $D\_k^2 \= 9$), а количество каналов $N$ велико (например, 64, 128 или больше), слагаемое $\\frac{1}{N}$ становится пренебрежимо малым.  
Таким образом, эффективность стремится к $\\frac{1}{9}$.  
**Вывод:** Depthwise Separable Convolution требует в **8–9 раз меньше** вычислительных операций и параметров, чем стандартная свертка, при сохранении тех же размерностей входа и выхода.1 Это и есть то "численное доказательство", которое мы ищем в нашем Квесте.

## **3\. Ритуал Реализации: Квест 13.1 (Solution)**

Перейдем от теории к практике. Задача Квеста 13.1 требует создать класс DepthwiseSeparableConv на языке Python с использованием библиотеки PyTorch. Ключевым моментом здесь является понимание параметра groups в модуле nn.Conv2d.

### **3.1. Ключевая Руна: groups=in_channels**

В PyTorch стандартная свертка (groups=1) соединяет все входные каналы со всеми выходными. Параметр groups позволяет разбить каналы на независимые группы. Если установить groups=in_channels, то каждая группа будет состоять ровно из одного канала. Это заставляет сверточное ядро работать изолированно с каждым каналом, что и является определением Depthwise Convolution.4

### **3.2. Код Артефакта**

Ниже представлен полный код решения, реализующий требуемую архитектуру и проводящий сравнение параметров.

Python

import torch  
import torch.nn as nn

\# \--- Акт 2: Чертеж "Экономного Рудокопа" (DepthwiseSeparableConv) \---  
class DepthwiseSeparableConv(nn.Module):  
 def \_\_init\_\_(self, in_channels, out_channels, kernel_size, padding=0):  
 super().\_\_init\_\_()

        \# 1\. Depthwise Convolution ("Гномы-специалисты")
        \# Ключевой момент: groups=in\_channels.
        \# Это означает, что каждый входной канал свертывается со своим собственным фильтром.
        \# Количество выходных каналов на этом этапе должно быть равно входным.
        self.depthwise \= nn.Conv2d(
            in\_channels=in\_channels,
            out\_channels=in\_channels,
            kernel\_size=kernel\_size,
            padding=padding,
            groups=in\_channels, \# \<--- МАГИЧЕСКАЯ РУНА
            bias=False \# Обычно bias не нужен перед BatchNorm (в реальных MobileNet)
        )

        \# 2\. Pointwise Convolution ("Магистр-Алхимик")
        \# Стандартная свертка 1x1, которая смешивает каналы.
        \# groups=1 (по умолчанию), так как нам нужно полное смешивание.
        self.pointwise \= nn.Conv2d(
            in\_channels=in\_channels,
            out\_channels=out\_channels,
            kernel\_size=1,
            bias=False
        )

    def forward(self, x):
        x \= self.depthwise(x)
        x \= self.pointwise(x)
        return x

\# \--- Акт 3: Ритуал Сравнения \---  
def run_comparison_ritual():  
 print("--- ЗАПУСК РИТУАЛА СРАВНЕНИЯ \---")

    \# Параметры эксперимента
    in\_channels \= 32
    out\_channels \= 64
    kernel\_size \= 3
    padding \= 1
    image\_size \= 64

    \# Создаем тензор "руды" (случайные данные)
    input\_tensor \= torch.randn(1, in\_channels, image\_size, image\_size)

    \# 1\. Жадный Рудокоп (Standard Conv2d)
    standard\_conv \= nn.Conv2d(in\_channels, out\_channels, kernel\_size, padding=padding, bias=False)
    std\_params \= sum(p.numel() for p in standard\_conv.parameters())
    std\_output \= standard\_conv(input\_tensor)

    \# 2\. Экономный Рудокоп (Separable Conv)
    separable\_conv \= DepthwiseSeparableConv(in\_channels, out\_channels, kernel\_size, padding=padding)
    sep\_params \= sum(p.numel() for p in separable\_conv.parameters())
    sep\_output \= separable\_conv(input\_tensor)

    \# \--- Анализ Результатов \---
    print(f"Входной тензор: {input\_tensor.shape}")
    print(f"\\n1. Стандартная свертка (Standard Conv2d):")
    print(f"   \- Параметры: {std\_params}")
    print(f"   \- Форма выхода: {std\_output.shape}")

    print(f"\\n2. Разделимая свертка (DepthwiseSeparableConv):")
    print(f"   \- Параметры: {sep\_params}")
    print(f"   \- Форма выхода: {sep\_output.shape}")

    \# Проверка совпадения форм
    assert std\_output.shape \== sep\_output.shape, "ОШИБКА: Формы тензоров не совпадают\!"
    print("\\n\[УСПЕХ\] Формы выходных тензоров идентичны.")

    \# Расчет эффективности
    ratio \= std\_params / sep\_params
    print(f"\\n\>\>\> ВЕРДИКТ: Экономный блок в {ratio:.2f} раз эффективнее по параметрам\!")

    \# Теоретический расчет для проверки
    \# Ratio ≈ 1/N \+ 1/K^2 \= 1/64 \+ 1/9 ≈ 0.0156 \+ 0.111 \= 0.126
    \# Inverted Ratio ≈ 1 / 0.126 ≈ 7.9
    theoretical\_ratio \= 1 / (1/out\_channels \+ 1/(kernel\_size\*\*2))
    print(f"\>\>\> Теоретическое ускорение: {theoretical\_ratio:.2f}x")

if \_\_name\_\_ \== "\_\_main\_\_":  
 run_comparison_ritual()

### **3.3. Анализ Результатов Выполнения**

При запуске этого кода мы получаем следующие данные:

- **Standard Conv:** $3 \\times 3 \\times 32 \\times 64 \= 18,432$ параметра.
- **Separable Conv:**
  - Depthwise: $3 \\times 3 \\times 32 \= 288$ параметров.
  - Pointwise: $1 \\times 1 \\times 32 \\times 64 \= 2,048$ параметров.
  - Итого: $2,336$ параметров.
- **Эффективность:** $18,432 / 2,336 \\approx 7.89$.

Результат практически совпадает с теоретическим предсказанием (около 8-9 раз). Мы создали архитектурный блок, который выполняет ту же структурную задачу, но "стоит" в 8 раз дешевле. Формы тензоров на выходе идентичны $(1, 64, 64, 64)$, что подтверждает возможность использования нашего блока как drop-in замены (прямой подстановки) в существующих сетях.

## **4\. Эволюция Архитектур MobileNet: От Простоты к Совершенству**

Изобретение разделимой свертки стало лишь началом. Чтобы создать действительно эффективную нейросеть, недостаточно просто заменить слои. Необходимо переосмыслить всю архитектуру. Семейство MobileNet демонстрирует, как инженеры Google шаг за шагом улучшали этот концепт.

### **4.1. MobileNetV1: Закладка Фундамента**

Первая версия MobileNet (Howard et al., 2017\) была построена как простой стек (стопка) из слоев Depthwise Separable Convolution.  
**Ключевые особенности:**

- **Прямая структура:** Сеть состоит из последовательности блоков Depthwise \-\> BatchNorm \-\> ReLU \-\> Pointwise \-\> BatchNorm \-\> ReLU.
- **Глобальные гиперпараметры:** Чтобы адаптировать сеть под разные устройства, были введены два множителя:
  - _Width Multiplier ($\\alpha$)_: Уменьшает количество каналов в каждом слое на коэффициент $\\alpha$ (например, 0.75 или 0.5). Это снижает количество параметров квадратично ($\\alpha^2$).
  - _Resolution Multiplier ($\\rho$)_: Уменьшает разрешение входного изображения. Снижает вычислительную нагрузку квадратично ($\\rho^2$), но не влияет на количество параметров.6

Проблема "Мертвых ядер" (Dead Kernels):  
В процессе эксплуатации выяснилось, что Depthwise ядра часто "умирают". Поскольку у ядра глубинной свертки размер всего $3 \\times 3 \\times 1$ (мало параметров), градиенты могут легко увести веса в отрицательную область. Если после этого идет ReLU, нейрон перестает активироваться навсегда. Это приводило к тому, что значительная часть емкости модели V1 оставалась неиспользованной.7

### **4.2. MobileNetV2: Инвертированные Остаточные Блоки**

MobileNetV2 (Sandler et al., 2018\) принес два фундаментальных улучшения, решивших проблемы V1 и значительно повысивших точность.

#### **4.2.1. Inverted Residuals (Инвертированные остаточные связи)**

В классическом ResNet блок имеет структуру "Wide \-\> Narrow \-\> Wide" (широкий-узкий-широкий). Сначала размерность уменьшается (bottleneck), происходит свертка, затем размерность восстанавливается. Это делалось для экономии вычислений в "тяжелых" слоях 3x3.  
В MobileNetV2 авторы перевернули эту концепцию: **"Narrow \-\> Wide \-\> Narrow"**.

1. **Expansion (Расширение):** Входной тензор с малым числом каналов сначала "раздувается" с помощью $1 \\times 1$ свертки (обычно в 6 раз, коэффициент $t=6$).
2. **Depthwise Conv:** Тяжелая пространственная фильтрация происходит в этом _высокомерном_ пространстве. Поскольку свертка Depthwise дешевая, мы можем позволить себе работать с большим количеством каналов.
3. **Projection (Проекция):** Затем тензор "сжимается" обратно в малое число каналов с помощью $1 \\times 1$ свертки.
4. **Shortcut:** Остаточная связь (skip connection) соединяет _узкие_ части блоков (bottlenecks). Это критически важно для экономии памяти, так как в памяти нужно хранить только небольшие тензоры остаточных связей.8

#### **4.2.2. Linear Bottlenecks (Линейные узкие места)**

Второе озарение авторов касалось функции активации ReLU. Они показали, что ReLU разрушает информацию в пространствах низкой размерности. Если тензор сжат (мало каналов), обнуление отрицательных значений приводит к безвозвратной потере данных ("коллапс многообразия").  
Решение: Убрать нелинейность (ReLU) после последнего проекционного слоя ($1 \\times 1$) в блоке. Выход блока остается линейным перед сложением с остаточной связью. Это позволяет сохранить "выразительную силу" модели даже при сильном сжатии.9

### **4.3. MobileNetV3: Эра AutoML и NAS**

MobileNetV3 (Howard et al., 2019\) знаменует отказ от чисто ручного проектирования. Архитектура была найдена с помощью алгоритмов **NAS (Neural Architecture Search)** — автоматического поиска архитектуры.  
**Инновации V3:**

- **NetAdapt:** Алгоритм автоматически подбирал количество каналов в каждом слое, оптимизируя баланс между точностью и реальной задержкой (latency) на конкретном мобильном CPU. В результате архитектура стала менее регулярной, чем V1/V2.11
- **Squeeze-and-Excitation (SE):** В блоки были внедрены легкие модули внимания (Attention). Они глобально усредняют пространственную информацию и перевзвешивают каналы. Это добавляет небольшие вычисления, но значительно повышает точность, позволяя сети фокусироваться на важных признаках.12
- **h-swish (Hard Swish):** Стандартная функция активации Swish ($x \\cdot \\sigma(x)$) эффективна, но требует вычисления сигмоиды, что дорого на мобильных процессорах. Авторы заменили её на аппроксимацию: $h\\text{-swish}(x) \= x \\frac{\\text{ReLU6}(x+3)}{6}$. Это можно вычислить используя только простые операции и сдвиги битов, что идеально для квантованных вычислений.14

## **5\. Аппаратное Обеспечение: Где теория расходится с практикой**

Вопрос Техноманта о том, "всегда ли этот блок лучше", вскрывает глубокую инженерную проблему. Теоретическое уменьшение FLOPs в 9 раз не всегда приводит к ускорению в 9 раз. Иногда разделимая свертка может работать даже медленнее обычной. Почему?

### **5.1. Арифметическая Интенсивность (Arithmetic Intensity)**

Скорость работы процессора (GPU/CPU) зависит от двух факторов: скорости вычислений (Compute) и скорости доступа к памяти (Memory Bandwidth).  
Показатель Arithmetic Intensity (AI) — это отношение количества вычислений к количеству загруженных байт памяти (FLOPs/Byte).

- **Стандартная свертка:** Имеет высокую AI. Мы загружаем веса один раз и используем их для огромного количества вычислений (каждый вес умножается на каждый пиксель всех каналов). Это режим **Compute Bound** (ограничено вычислениями). GPU здесь работают эффективно.
- **Depthwise свертка:** Имеет очень низкую AI. Мы загружаем вес, делаем всего одно умножение и переходим дальше. Процессор простаивает, ожидая загрузки данных из памяти. Это режим **Memory Bound** (ограничено памятью).

На мощных серверных GPU (например, NVIDIA A100/V100) пропускная способность памяти огромна, но вычислительная мощь еще больше. Depthwise свертки не могут насытить тысячи ядер GPU работой, и "магия" экономии рассыпается — время тратится на пересылку данных, а не на счет.15

### **5.2. Roofline Model Analysis**

Модель "Roofline" (Крыша) визуализирует эти ограничения.

- Горизонтальная линия ("Крыша") — пиковая производительность процессора (GFLOPS).
- Наклонная линия — ограничение пропускной способности памяти.  
  Depthwise свертки часто попадают под наклонную линию. Это означает, что даже если мы уменьшим количество FLOPs в 10 раз, мы не ускоримся, если не уменьшим объем передаваемой памяти.

### **5.3. Роль NPU и Специализированных Ускорителей**

Именно поэтому появились NPU (Neural Processing Units) в чипах Apple (A-series), Qualcomm (Snapdragon) и Google (Tensor). Эти процессоры спроектированы специально для эффективного выполнения операций с низкой арифметической интенсивностью, таких как Depthwise Conv. Они используют жестко "зашитые" конвейеры данных, чтобы минимизировать накладные расходы на чтение памяти.  
На мобильном процессоре Snapdragon 888 выполнение MobileNetV2 на NPU (Hexagon) в разы энергоэффективнее и быстрее, чем на встроенном GPU (Adreno), и на порядок быстрее, чем на CPU.17

## **6\. Ответ Техноманту: Компромиссы и Выбор Инструмента**

_Вопрос Техноманта:_ "Означает ли это, что он всегда и во всем лучше стандартного Conv2d? Есть ли у него слабости?"  
_Развернутый ответ Мастера:_  
Нет, юный Архитектор. Экономный блок не является абсолютным идеалом. Как и в любой сложной магии, здесь действует закон равноценного обмена.

1. Потеря Информационной Емкости:  
   Разделив свертку, мы разорвали связи между каналами и пространством. В стандартной свертке нейрон может выучить сложный паттерн, например, "красный вертикальный край". В разделимой свертке "гном" видит только "вертикальный край", а "алхимик" знает только, что "было много красного". Связь "красный \+ вертикальный" восстанавливается лишь косвенно. Это приводит к тому, что при одинаковом числе параметров MobileNet может проигрывать в точности (Accuracy) полноценным ResNet на сложных задачах.19
2. Проблемы с Обучением:  
   Из-за малой емкости и отсутствия перекрестных связей внутри ядра $K \\times K$, Depthwise слои сложнее обучать. Они более чувствительны к инициализации весов и подбору скорости обучения (learning rate). Также они хуже поддаются прунингу (прореживанию), так как и так уже предельно "тонки".20
3. Неэффективность на "Большом Железе":  
   Если твоя битва происходит на мощных серверах с видеокартами NVIDIA A100 (например, обучение базовых моделей), использование Depthwise Convolutions может замедлить процесс обучения по сравнению с ResNet-50 из\-за низкой утилизации GPU. "Мана" экономится, но "время каста" может увеличиться.15

**Твоя Стратегия:**

- Используй **Depthwise Separable Conv (MobileNet)**, если твоя цель — **Inference (Вывод)** на смартфонах, камерах, дронах или Raspberry Pi. Здесь выигрыш в скорости и энергопотреблении критичен.
- Используй **Standard Conv (ResNet, EfficientNet-B4+)**, если твоя цель — максимальная точность на сервере, победа в соревновании Kaggle или обучение фундаментальной модели, где ресурсы не являются узким местом.

## **7\. Бизнес-Ценность и Заключение**

Владение техникой Depthwise Separable Convolution открывает для бизнеса двери, ранее закрытые "тяжелыми" нейросетями:

1. **Real-time Video Analytics on Edge:** Камеры наблюдения могут распознавать лица и опасные ситуации локально, не отправляя видеопоток на сервер. Это экономит гигабайты трафика и снимает вопросы приватности данных.
2. **Offline AI:** Приложения-переводчики и навигаторы могут работать в режиме дополненной реальности без доступа к Интернету (например, в роуминге).
3. **Cost Savings:** Даже в облаке замена ResNet-50 на MobileNetV3 может сократить счет за инстансы AWS/GCP в 2-3 раза при сохранении приемлемого уровня качества для многих задач классификации.

## В этом Квесте вы не просто написали код. Вы освоили философию **эффективного дизайна**. Вы научились видеть за абстракциями nn.Conv2d реальные операции умножения и потоки данных. Это знание — главное оружие Архитектора в эпоху повсеместного и невидимого ИИ.

**Таблица 1: Сравнительная характеристика архитектур**

| Архитектура      | Основной Блок         | Инновации                         | Use Case                           |
| :--------------- | :-------------------- | :-------------------------------- | :--------------------------------- |
| **Standard CNN** | Conv 3x3              | Базовая свертка                   | Серверное обучение, макс. точность |
| **MobileNetV1**  | Depthwise Sep Conv    | Разделение каналов и пространства | Ранние мобильные приложения        |
| **MobileNetV2**  | Inverted Residual     | Линейные узкие места, инверсия    | Стандарт де\-факто для Android/iOS |
| **MobileNetV3**  | V2 Block \+ SE \+ NAS | h-swish, авто-поиск архитектуры   | Современные SoC с NPU              |

_Конец отчета._

#### **Источники**

1. Depth-Wise Separable Convolution Neural Network with Residual Connection for Hyperspectral Image Classification \- MDPI, дата последнего обращения: декабря 20, 2025, [https://www.mdpi.com/2072-4292/12/20/3408](https://www.mdpi.com/2072-4292/12/20/3408)
2. Depth-wise Convolution and Depth-wise Separable Convolution | by Atul Pandey \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec](https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec)
3. Depth wise Separable Convolutional Neural Networks \- GeeksforGeeks, дата последнего обращения: декабря 20, 2025, [https://www.geeksforgeeks.org/machine-learning/depth-wise-separable-convolutional-neural-networks/](https://www.geeksforgeeks.org/machine-learning/depth-wise-separable-convolutional-neural-networks/)
4. Understanding groups parameter in Conv2d | by Ashish Jha \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@mechatronics420/understanding-groups-parameter-in-conv2d-b330d87208d2](https://medium.com/@mechatronics420/understanding-groups-parameter-in-conv2d-b330d87208d2)
5. Conv2d — PyTorch 2.9 documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
6. MobileNetV1: depthwise separable Convolution : | by Aymen Boukhari | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@aymne011/mobilenetv1-depthwise-separable-filters-4ab6eeb01c79](https://medium.com/@aymne011/mobilenetv1-depthwise-separable-filters-4ab6eeb01c79)
7. MobileNetV1 Paper Walkthrough: The Tiny Giant \- Towards Data Science, дата последнего обращения: декабря 20, 2025, [https://towardsdatascience.com/the-tiny-giant-mobilenetv1/](https://towardsdatascience.com/the-tiny-giant-mobilenetv1/)
8. A Summary of the “MobileNetV2: Inverted Residuals and Linear Bottlenecks” Paper | by Zaynab Awofeso | CodeX | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/codex/a-summary-of-the-mobilenetv2-inverted-residuals-and-linear-bottlenecks-paper-e19b187cb78a](https://medium.com/codex/a-summary-of-the-mobilenetv2-inverted-residuals-and-linear-bottlenecks-paper-e19b187cb78a)
9. MobileNetV2: Inverted Residuals and Linear Bottlenecks \- CVF Open Access, дата последнего обращения: декабря 20, 2025, [https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf)
10. MobileNetV2 Paper Walkthrough: The Smarter Tiny Giant | Towards Data Science, дата последнего обращения: декабря 20, 2025, [https://towardsdatascience.com/mobilenetv2-paper-walkthrough-the-smarter-tiny-giant/](https://towardsdatascience.com/mobilenetv2-paper-walkthrough-the-smarter-tiny-giant/)
11. Understanding and Implementing MobileNetV3 | by Rishabh Singh \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a](https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a)
12. DSS-MobileNetV3: An Efficient Dynamic-State-Space- Enhanced Network for Concrete Crack Segmentation \- MDPI, дата последнего обращения: декабря 20, 2025, [https://www.mdpi.com/2075-5309/15/11/1905](https://www.mdpi.com/2075-5309/15/11/1905)
13. A visual deep-dive into the building blocks of MobileNetV3 \- \- Francesco Pochetti, дата последнего обращения: декабря 20, 2025, [https://francescopochetti.com/a-visual-deep-dive-into-the-building-blocks-of-mobilenetv3/](https://francescopochetti.com/a-visual-deep-dive-into-the-building-blocks-of-mobilenetv3/)
14. Introducing the Next Generation of On-Device Vision Models: MobileNetV3 and MobileNetEdgeTPU \- Google Research, дата последнего обращения: декабря 20, 2025, [https://research.google/blog/introducing-the-next-generation-of-on-device-vision-models-mobilenetv3-and-mobilenetedgetpu/](https://research.google/blog/introducing-the-next-generation-of-on-device-vision-models-mobilenetv3-and-mobilenetedgetpu/)
15. Optimizing Depthwise Separable Convolution Operations on GPUs \- White Rose Research Online, дата последнего обращения: декабря 20, 2025, [https://eprints.whiterose.ac.uk/id/eprint/174797/1/main.pdf](https://eprints.whiterose.ac.uk/id/eprint/174797/1/main.pdf)
16. Why would \`DepthwiseConv2D\` be slower than \`Conv2D\` \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/63332819/why-would-depthwiseconv2d-be-slower-than-conv2d](https://stackoverflow.com/questions/63332819/why-would-depthwiseconv2d-be-slower-than-conv2d)
17. CPU vs GPU vs NPU: What's the difference? \- Corsair, дата последнего обращения: декабря 20, 2025, [https://www.corsair.com/us/en/explorer/diy-builder/power-supply-units/cpu-vs-gpu-vs-npu-whats-the-difference/](https://www.corsair.com/us/en/explorer/diy-builder/power-supply-units/cpu-vs-gpu-vs-npu-whats-the-difference/)
18. MLPerf Mobile Inference Benchmark \- MLSys Proceedings, дата последнего обращения: декабря 20, 2025, [https://proceedings.mlsys.org/paper_files/paper/2022/file/a2b2702ea7e682c5ea2c20e8f71efb0c-Paper.pdf](https://proceedings.mlsys.org/paper_files/paper/2022/file/a2b2702ea7e682c5ea2c20e8f71efb0c-Paper.pdf)
19. Understanding Depthwise Separable Convolutions and the efficiency of MobileNets | by Arjun Sarkar | TDS Archive | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/data-science/understanding-depthwise-separable-convolutions-and-the-efficiency-of-mobilenets-6de3d6b62503](https://medium.com/data-science/understanding-depthwise-separable-convolutions-and-the-efficiency-of-mobilenets-6de3d6b62503)
20. DEPrune: Depth-wise Separable Convolution Pruning for Maximizing GPU Parallelism, дата последнего обращения: декабря 20, 2025, [https://proceedings.neurips.cc/paper_files/paper/2024/file/c16a99558b0b4f6b10966ca9bdb98ade-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/c16a99558b0b4f6b10966ca9bdb98ade-Paper-Conference.pdf)
21. Diagonalwise Refactorization: An Efficient Training Method for Depthwise Convolutions \- arXiv, дата последнего обращения: декабря 20, 2025, [https://arxiv.org/pdf/1803.09926](https://arxiv.org/pdf/1803.09926)
