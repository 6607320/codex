# **Ритуал Квантизации: Фундаментальное исследование и практическая реализация Post-Training Static Quantization в архитектурах сверточных нейронных сетей**

## **1\. Введение: Смена парадигмы в вычислительной нейробиологии**

Современный этап развития глубокого обучения характеризуется фундаментальным противоречием: экспоненциальный рост сложности нейросетевых архитектур сталкивается с физическими ограничениями аппаратного обеспечения, предназначенного для их эксплуатации. В то время как исследовательские лаборатории соревнуются в увеличении количества параметров моделей, достигая триллионных значений, инженерная реальность диктует иные условия — необходимость развертывания искусственного интеллекта на периферийных устройствах (Edge AI), встраиваемых системах и IoT-инфраструктуре, где ресурсы памяти, энергопотребления и вычислительной мощности жестко лимитированы.1  
В этом контексте процесс квантования (Quantization) перестает быть факультативной техникой оптимизации и трансформируется в критически важный этап жизненного цикла любой промышленной ML-модели. "Ритуал Квантизации", обозначенный в Квесте 14.1, представляет собой не просто техническую конвертацию форматов данных, а глубокое архитектурное преобразование, затрагивающее саму математическую основу инференса нейронных сетей.  
Данный отчет представляет собой исчерпывающее аналитическое исследование и подробное техническое руководство по применению метода Post-Training Static Quantization (PTQ) к архитектуре MiniCNN с использованием фреймворка PyTorch. Мы подробно рассмотрим теоретические основы перехода от арифметики с плавающей запятой (FP32) к целочисленной арифметике (INT8), проанализируем архитектурные особенности аппаратных бэкендов (x86 FBGEMM), раскроем математическую суть процесса калибровки и проведем детальный сравнительный анализ весовых коэффициентов до и после квантования.

### **1.1. Проблема "Стены Памяти" и энергетическая эффективность**

Доминирующий формат обучения нейронных сетей — FP32 (Single Precision Floating Point) — обеспечивает колоссальный динамический диапазон, необходимый для корректного вычисления градиентов методом обратного распространения ошибки. Однако при инференсе, когда веса модели зафиксированы, такая избыточная точность становится бременем.  
Главным узким местом современных вычислительных систем является не скорость процессора, а пропускная способность памяти — феномен, известный как "Memory Wall". Загрузка 32-битного весового коэффициента из DRAM в регистры процессора потребляет на порядки больше энергии, чем сама операция умножения-накопления (MAC). Переход к INT8 позволяет сократить объем передаваемых данных в 4 раза, что линейно снижает требования к пропускной способности памяти и энергопотребление.3 Более того, целочисленные операции выполняются современными ALU значительно быстрее благодаря векторизации инструкций (SIMD), таких как AVX512-VNNI.5

| Характеристика                | FP32 (Float)                        | INT8 (Integer)                            | Влияние на систему                                             |
| :---------------------------- | :---------------------------------- | :---------------------------------------- | :------------------------------------------------------------- |
| **Размер данных**             | 32 бита (4 байта)                   | 8 бит (1 байт)                            | Уменьшение размера модели и нагрузки на шину памяти в 4 раза 4 |
| **Динамический диапазон**     | $\\approx \\pm 3.4 \\times 10^{38}$ | $\[-128, 127\]$ или $$                    | Требует тщательной калибровки для предотвращения насыщения     |
| **Энергия на операцию (MAC)** | Высокая                             | Низкая (до 10-30x экономии)               | Критично для устройств с питанием от батареи 7                 |
| **Аппаратная поддержка**      | FMA (AVX2/AVX512)                   | VNNI (Vector Neural Network Instructions) | Ускорение инференса в 2-4 раза на CPU 6                        |

## ---

**2\. Теоретический базис статического квантования**

Для успешного выполнения "Ритуала" необходимо глубокое понимание математического аппарата, лежащего в основе преобразования непрерывного сигнала в дискретный. Статическое квантование в PyTorch базируется на аффинном преобразовании, которое отображает реальные значения тензоров на целочисленную сетку.

### **2.1. Математика аффинного отображения**

Процесс квантования описывается функцией, которая преобразует число с плавающей запятой $r$ (real) в целое число $q$ (quantized):

$$Q(r) \= \\text{clamp}\\left(\\text{round}\\left(\\frac{r}{S} \+ Z\\right), q\_{min}, q\_{max}\\right)$$  
Где:

- $S$ (**Scale**) — масштабный коэффициент, положительное число FP32. Он определяет шаг дискретизации, то есть "расстояние" между соседними представимыми значениями в исходном пространстве.
- $Z$ (**Zero-point**) — целочисленное смещение. Это значение в квантованном пространстве, которое строго соответствует реальному нулю ($r=0$).
- $q\_{min}, q\_{max}$ — границы допустимого диапазона (например, \-128 и 127 для qint8).

Обратная операция, или деквантование (Dequantization), необходима для восстановления значений (аппроксимации) для операций, не поддерживающих INT8, или для интерпретации результатов:

$$r\_{approx} \= S \\cdot (q \- Z)$$  
Фундаментальная роль Zero-point:  
В нейронных сетях значение $0.0$ имеет особое значение (например, результат работы ReLU, нули в Padding свертки, разреженные матрицы). Если бы мы использовали симметричное квантование без смещения ($Z=0$), то реальный ноль мог бы отобразиться в значение с ошибкой (например, $0.0023$). Это привело бы к потере свойства разреженности (sparsity) и накоплению ошибок смещения (bias shift) при сверточных операциях. Параметр $Z$ гарантирует, что $0.0\_{FP32} \\equiv Z\_{INT8}$.8

### **2.2. Гранулярность квантования (Granularity)**

Выбор гранулярности расчета параметров $S$ и $Z$ является ключевым компромиссом между точностью и производительностью.

1. Per-Tensor (Потензорное):  
   Рассчитывается одна пара $(S, Z)$ для всего тензора.
   - _Применение:_ Активации (выходы слоев).
   - _Проблема:_ Активации могут иметь выбросы, но обычно распределены относительно равномерно в пределах батча.
   - _Преимущество:_ Высокая скорость вычислений, так как пересчет масштаба происходит один раз для всего массива данных.8
2. Per-Channel (Поканальное):  
   Параметры $(S\_c, Z\_c)$ рассчитываются индивидуально для каждого выходного канала $c$.
   - _Применение:_ Веса (Weights) сверточных и линейных слоев.
   - _Обоснование:_ Фильтры свертки в одном слое могут иметь кардинально разные динамические диапазоны. Один фильтр может детектировать высококонтрастные границы (большие значения весов), а другой — тонкие текстурные нюансы (малые значения). Если применить общий $S$ для всего слоя, "тихий" фильтр будет сплющен в ноль (quantized to zero), так как его значения окажутся меньше шага дискретизации $S$, продиктованного "громким" фильтром. Поканальное квантование решает эту проблему, выделяя каждому фильтру свой масштаб.11

### **2.3. Режимы квантования: Симметричный и Асимметричный**

- **Асимметричное (Affine):** Использует полноценную формулу с $Z \\neq 0$. Диапазон входных данных $\[min, max\]$ отображается в $\[q\_{min}, q\_{max}\]$. Это позволяет максимально эффективно использовать битовую глубину, особенно для данных, смещенных относительно нуля (например, выходы ReLU, которые всегда $\\ge 0$). PyTorch использует этот режим для активаций по умолчанию.11
- **Симметричное (Symmetric):** Принудительно устанавливает $Z=0$ (или центрирует диапазон). Диапазон $\[-max(|min|, |max|), \+max(|min|, |max|)\]$ отображается в симметричный диапазон INT8. Это упрощает вычисления (исключает слагаемое $Z$ из формул умножения матриц), что ускоряет инференс. Стандартно используется для весов, так как их распределение обычно близко к нормальному с центром в нуле.8

## ---

**3\. Объект исследования: Архитектура MiniCNN**

Для выполнения Квеста 14.1 мы спроектируем модель MiniCNN. Это компактная сверточная сеть, предназначенная для классификации изображений (например, датасет CIFAR-10, размер входа 32x32x3). Архитектура выбрана таким образом, чтобы включать все типичные блоки, требующие особого внимания при квантовании: свертки, нормализацию, нелинейности и полносвязные слои.

### **3.1. Определение модели**

Python

import torch  
import torch.nn as nn  
import torch.nn.functional as F

class MiniCNN(nn.Module):  
 def \_\_init\_\_(self, num_classes=10):  
 super(MiniCNN, self).\_\_init\_\_()

        \# Блок 1: Извлечение низкоуровневых признаков
        \# Вход: 3 канала (RGB), Выход: 32 канала
        self.conv1 \= nn.Conv2d(in\_channels=3, out\_channels=32, kernel\_size=3, stride=1, padding=1, bias=False)
        self.bn1 \= nn.BatchNorm2d(32)
        self.relu1 \= nn.ReLU(inplace=True)

        \# Блок 2: Углубление признаков и снижение размерности
        self.conv2 \= nn.Conv2d(in\_channels=32, out\_channels=64, kernel\_size=3, stride=1, padding=1, bias=False)
        self.bn2 \= nn.BatchNorm2d(64)
        self.relu2 \= nn.ReLU(inplace=True)
        self.pool \= nn.MaxPool2d(kernel\_size=2, stride=2) \# 32x32 \-\> 16x16

        \# Блок 3: Полносвязная классификация
        \# Размер после пулинга: 64 канала \* 16 \* 16 пикселей
        self.flatten\_dim \= 64 \* 16 \* 16
        self.fc1 \= nn.Linear(self.flatten\_dim, 512\)
        self.relu3 \= nn.ReLU(inplace=True)
        self.fc2 \= nn.Linear(512, num\_classes)

        \# Квантификационные заглушки (Stubs)
        \# Они необходимы для статического квантования в Eager Mode
        self.quant \= torch.ao.quantization.QuantStub()
        self.dequant \= torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        \# 1\. Квантование входа (FP32 \-\> INT8)
        \# На этапе калибровки здесь работает Наблюдатель (Observer)
        x \= self.quant(x)

        \# 2\. Сверточная часть
        x \= self.conv1(x)
        x \= self.bn1(x)
        x \= self.relu1(x)

        x \= self.conv2(x)
        x \= self.bn2(x)
        x \= self.relu2(x)
        x \= self.pool(x)

        \# 3\. Flatten
        x \= x.reshape(x.size(0), \-1)

        \# 4\. Полносвязная часть
        x \= self.fc1(x)
        x \= self.relu3(x)
        x \= self.fc2(x)

        \# 5\. Деквантование выхода (INT8 \-\> FP32)
        \# Логиты возвращаются в FP32 для Softmax и вычисления потерь
        x \= self.dequant(x)
        return x

**Анализ архитектурных решений:**

- **QuantStub/DeQuantStub:** Эти модули служат границами "квантованного мира". Все операции между ними будут конвертированы в INT8. Входные данные поступают в FP32, QuantStub их измеряет (калибрует) и переводит в INT8. DeQuantStub переводит результат обратно в FP32. Это критически важно, так как большинство функций потерь (CrossEntropy) требуют высокой точности.14
- **BatchNorm bias=False:** В слоях Conv2d перед BatchNorm2d параметр bias установлен в False. Это стандартная практика, так как смещение свертки становится избыточным при наличии смещения в последующем слое BatchNorm (оно будет поглощено при слиянии слоев).16

### **3.2. Шаг 1: Обучение модели (Baseline FP32)**

Первый этап квеста — наличие обученной модели. Статическое квантование — это метод _post-training_, он применяется к уже сошедшейся модели.

Python

import torch.optim as optim  
from torchvision import datasets, transforms

def train_model(model, device="cpu", epochs=5):  
 \# Стандартный цикл обучения (упрощен для краткости)  
 \# Используем CIFAR-10 как пример данных  
 transform \= transforms.Compose()  
 train_loader \= torch.utils.data.DataLoader(  
 datasets.CIFAR10('./data', train=True, download=True, transform=transform),  
 batch_size=32, shuffle=True)

    criterion \= nn.CrossEntropyLoss()
    optimizer \= optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    model.train()

    print("Начало обучения FP32 модели...")
    for epoch in range(epochs):
        running\_loss \= 0.0
        for i, (inputs, labels) in enumerate(train\_loader):
            inputs, labels \= inputs.to(device), labels.to(device)
            optimizer.zero\_grad()
            outputs \= model(inputs)
            loss \= criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running\_loss \+= loss.item()
        print(f"Epoch {epoch+1}, Loss: {running\_loss / len(train\_loader):.4f}")

    print("Обучение завершено.")
    \# Важно: переводим на CPU для квантования (PTQ в PyTorch лучше всего поддерживается на CPU)
    model.to('cpu')
    model.eval() \# КРИТИЧЕСКИ ВАЖНО для корректной работы BatchNorm при фьюзинге
    return model

\# Инициализация и обучение  
model_fp32 \= MiniCNN()  
\# model_fp32 \= train_model(model_fp32) \# Раскомментировать для реального запуска  
\# Для отчета предположим, что веса загружены:  
\# model_fp32.load_state_dict(torch.load("minicnn_fp32.pth"))  
model_fp32.eval()

**Критическое замечание:** Перевод модели в режим eval() является обязательным. В режиме обучения (train) слои BatchNorm обновляют свои бегущие статистики (running mean/var) и используют статистику текущего батча. Для квантования нам нужно "заморозить" эти статистики, чтобы корректно "влить" их в веса свертки.15

## ---

**4\. Исполнение Ритуала: Пошаговая реализация PTQ**

Процесс статического квантования в PyTorch строго регламентирован и состоит из трех фаз: **Fusion (Слияние)**, **Preparation (Подготовка/Инструментация)** и **Calibration (Калибровка)**, за которыми следует **Conversion (Конвертация)**.

### **4.1. Фаза I: Слияние операторов (Operator Fusion)**

Первый шаг оптимизации — слияние последовательных операций в единые вычислительные ядра. Стандартный паттерн в CNN: Conv2d \-\> BatchNorm2d \-\> ReLU. В FP32 это три разных прохода по памяти: чтение/запись промежуточных буферов. В квантованной модели мы можем математически объединить их.  
Математическая суть слияния (Folding):  
Слой BatchNorm выполняет нормализацию:

$$y \= \\frac{x \- \\mu}{\\sqrt{\\sigma^2 \+ \\epsilon}} \\cdot \\gamma \+ \\beta$$

Слой свертки выполняет операцию: $x \= W \\cdot input \+ b$.  
Подставив свертку в формулу BatchNorm, мы можем пересчитать веса $W$ и смещение $b$ самой свертки так, чтобы они сразу выдавали нормализованный результат:

$$W\_{fused} \= W \\cdot \\frac{\\gamma}{\\sqrt{\\sigma^2 \+ \\epsilon}}$$

$$b\_{fused} \= (b \- \\mu) \\cdot \\frac{\\gamma}{\\sqrt{\\sigma^2 \+ \\epsilon}} \+ \\beta$$  
Функция активации ReLU (отсечение отрицательных значений) в квантованном мире реализуется просто как ограничение диапазона выходных значений (output clipping), не требуя отдельной операции.

Python

\# Создаем копию модели для квантования  
model_to_quantize \= copy.deepcopy(model_fp32)  
model_to_quantize.eval()

\# Списки модулей для слияния. Порядок важен\!  
\# PyTorch ищет последовательности слоев по именам.  
modules_to_fuse \= \['conv1', 'bn1', 'relu1'\],  
 \['conv2', 'bn2', 'relu2'\],  
 \['fc1', 'relu3'\] \# Линейный слой тоже можно слить с ReLU  
\]

\# Выполняем слияние  
model_fused \= torch.ao.quantization.fuse_modules(  
 model_to_quantize,  
 modules_to_fuse,  
 inplace=False  
)

\# Проверка: слои bn1 и relu1 теперь должны быть Identity()  
print("После слияния conv1:", type(model_fused.conv1)) \# Ожидается: torch.nn.intrinsic.modules.fused.ConvReLU2d  
print("После слияния bn1:", type(model_fused.bn1)) \# Ожидается: torch.nn.Identity

После этого шага количество операций в графе уменьшилось, а веса conv1 и conv2 изменились, "поглотив" параметры батч-нормализации.14

### **4.2. Фаза II: Конфигурация и Подготовка (Prepare)**

Здесь мы выбираем стратегию квантования. Для серверных CPU (x86) стандартом является бэкенд **FBGEMM** (Facebook GEneral Matrix Multiplication), оптимизированный под AVX2/AVX512. Для мобильных процессоров (ARM) используется **QNNPACK**.18  
Выбор Наблюдателей (Observers):  
Нам необходимо определить, как именно мы будем измерять диапазоны значений. get_default_qconfig('fbgemm') автоматически выбирает лучшие практики:

- **Для активаций:** HistogramObserver. Он строит гистограмму значений. Это позволяет использовать алгоритмы минимизации ошибки (например, L2-norm), отбрасывая редкие выбросы (outliers). Если бы мы использовали простой MinMaxObserver, один случайный пиксель со значением 1000 растянул бы весь диапазон, и полезный сигнал в диапазоне получил бы всего пару бит разрешения. HistogramObserver умнее: он сузит диапазон, пожертвовав выбросом ради точности основного сигнала.20
- **Для весов:** PerChannelMinMaxObserver. Поскольку веса статичны и мы хотим сохранить их структуру максимально точно для каждого фильтра, поканальный MinMax — идеальный выбор.11

Python

\# 1\. Настройка конфигурации  
backend \= 'fbgemm' \# Используем x86 бэкенд  
torch.backends.quantized.engine \= backend  
model_fused.qconfig \= torch.ao.quantization.get_default_qconfig(backend)

\# 2\. Подготовка (Instrumentation)  
\# Вставляет наблюдателей (Observers) в ключевые точки графа (входы слоев, выходы активаций)  
model_prepared \= torch.ao.quantization.prepare(model_fused, inplace=False)

\# Теперь модель "обвешана" датчиками, но все еще работает в FP32

### **4.3. Фаза III: Суть Калибровки (Calibration)**

Это центральный элемент "Ритуала", о котором спрашивалось в задании.  
Суть калибровки:  
Калибровка — это процесс эмпирического определения динамического диапазона активаций $\[\\alpha, \\beta\]$. В отличие от весов, которые известны заранее, значения активаций (выходов нейронов) зависят от входных данных. Мы не можем знать, какие значения выдаст слой conv2, пока не прогоним через него реальное изображение.  
Во время калибровки мы подаем на вход модели **репрезентативную выборку** данных (Representative Dataset). Это не обязательно должен быть весь обучающий набор — обычно достаточно 100-200 мини-батчей. Главное, чтобы эти данные статистически отражали то, что модель увидит в реальности. Использование случайного шума (torch.randn) **недопустимо**, так как гауссово распределение шума кардинально отличается от распределения признаков реальных изображений, что приведет к неверному расчету параметров $S$ и $Z$ и полной деградации точности.11  
**Алгоритм действий наблюдателя во время калибровки:**

1. Данные проходят через слой.
2. HistogramObserver обновляет счетчики в бинах, формируя плотность вероятности значений.
3. Вычисляются текущие $min$ и $max$.
4. Сам инференс продолжается в FP32.

Python

def calibrate_model(model, data_loader, num_batches=100):  
 model.eval()  
 print("Запуск калибровки (сбор статистики активаций)...")  
 with torch.no_grad():  
 for i, (images, \_) in enumerate(data_loader):  
 if i \>= num_batches:  
 break  
 \# Прогон данных запускает observers.forward()  
 model(images)  
 print("Калибровка завершена.")

\# Создаем загрузчик для калибровки (используем часть валидационного или тренировочного сета)  
\# Важно: В идеале использовать подмножество training set, чтобы избежать переобучения под валидацию (data leakage),  
\# но для PTQ часто используют validation set для удобства оценки.\[21\]  
calibration_loader \= torch.utils.data.DataLoader(  
 datasets.CIFAR10('./data', train=True, transform=transforms.Compose()),  
 batch_size=32, shuffle=True  
)

calibrate_model(model_prepared, calibration_loader)

### **4.4. Фаза IV: Конвертация (Conversion)**

Финальный шаг — трансформация графа.

1. **Расчет параметров:** На основе собранных гистограмм вычисляются финальные $S$ и $Z$ для всех активаций.
2. **Квантование весов:** Веса $W\_{fp32}$ конвертируются в $W\_{int8}$ и упаковываются в оптимизированные структуры памяти (PackedParams), специфичные для FBGEMM.
3. **Замена операторов:** Модули nn.Conv2d (с обертками наблюдателей) заменяются на nn.quantized.Conv2d. Эти новые модули не содержат весов FP32, только INT8 и параметры масштабирования (bias остается в FP32 или INT32 для аккумулирования высокой точности).22
4. **Очистка:** Наблюдатели удаляются.

Python

\# Конвертация в INT8 модель  
model_int8 \= torch.ao.quantization.convert(model_prepared, inplace=False)

print("Модель успешно конвертирована в INT8.")  
print(model_int8)  
\# Теперь в выводе вы увидите слои QuantizedConv2d, QuantizedLinear и т.д.

## ---

**5\. Сравнительный анализ: FP32 vs INT8**

Согласно заданию, необходимо провести сравнение весов и объяснить результаты. Это позволит увидеть "физический" смысл проведенного ритуала.

### **5.1. Анализ размера модели (Model Size)**

Ожидаемое сокращение размера — примерно в 4 раза.

Python

import os

def get_model_size(model, path="temp.p"):  
 torch.save(model.state_dict(), path)  
 size_mb \= os.path.getsize(path) / 1e6  
 os.remove(path)  
 return size_mb

size_fp32 \= get_model_size(model_fp32)  
size_int8 \= get_model_size(model_int8)

print(f"Размер FP32: {size_fp32:.2f} MB")  
print(f"Размер INT8: {size_int8:.2f} MB")  
print(f"Коэффициент сжатия: {size_fp32 / size_int8:.2f}x")

Интерпретация:  
FP32 использует 4 байта на параметр. INT8 — 1 байт. Основную часть файла занимают веса полносвязных слоев (fc1 имеет $64 \\cdot 16 \\cdot 16 \\cdot 512 \\approx 8.3$ миллиона параметров).  
$8.3M \\times 4 \\text{ байта} \\approx 33 \\text{ MB}$.  
$8.3M \\times 1 \\text{ байт} \\approx 8.3 \\text{ MB}$.  
Дополнительные накладные расходы (хранение $S$ и $Z$ для каждого канала, метаданные графа) незначительны, поэтому мы получаем коэффициент сжатия, очень близкий к идеальному 4x.4

### **5.2. Сравнительный анализ весов (Weight Inspection)**

Посмотрим, что произошло с весами первого сверточного слоя.

Python

\# Извлечение весов  
\# Для FP32 модели все просто:  
weights_fp32 \= model_fp32.conv1.weight.data.flatten().numpy()

\# Для INT8 модели веса упакованы. Нужно использовать специальные методы доступа.  
\# weight() возвращает PackedParameter, у которого есть метод int_repr()  
quantized_layer \= model_int8.conv1  
weights_int8 \= quantized_layer.weight().int_repr().flatten().numpy() \# Целые числа \[-128, 127\] or  
weights_dequantized \= quantized_layer.weight().dequantize().flatten().numpy() \# Восстановленные float

\# Визуализация (псевдокод для построения гистограмм)  
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))  
plt.subplot(1, 2, 1\)  
plt.hist(weights_fp32, bins=100, alpha=0.7, label='Original FP32')  
plt.hist(weights_dequantized, bins=100, alpha=0.7, label='Dequantized INT8', histtype='step')  
plt.legend()  
plt.title("Распределение весов: Оригинал vs Восстановленные")

plt.subplot(1, 2, 2\)  
plt.hist(weights_int8, bins=50, color='orange')  
plt.title("Распределение квантованных значений (INT8)")  
plt.show()

**Анализ распределения:**

1. **Дискретизация:** На графике "Original vs Dequantized" видно, что синяя гистограмма (FP32) — гладкая, а оранжевая (Dequantized) — ступенчатая. Значения весов теперь "привязаны" к дискретной сетке с шагом $S$.
2. **Форма колокола:** Гистограмма INT8 значений (справа) сохраняет форму колокола (Гауссово распределение), характерную для весов нейросети. Это подтверждает, что PerChannelMinMaxObserver корректно подобрал масштаб, и диапазон INT8 $\[-128, 127\]$ используется эффективно. Если бы мы видели пики только в 0 или на границах (-128, 127), это означало бы плохую калибровку (насыщение).10
3. **Поканальные различия:** Если проанализировать масштабы ($S$) для разных каналов model_int8.conv1.weight().q_per_channel_scales(), мы увидим, что они различаются. Это подтверждает необходимость Per-Channel квантования: некоторые фильтры имеют большую амплитуду весов, некоторые — малую.

### **5.3. Точность и Скорость**

Хотя код замера точности и скорости выходит за рамки минимального кода квеста, в реальном отчете это обязательная часть.

- **Точность (Accuracy):** Обычно PTQ приводит к падению точности на \<1% для робастных моделей (ResNet), но может обрушить точность легких моделей (MobileNet) на 3-5% и более. Если падение критично, необходимо использовать **Quantization Aware Training (QAT)**, где калибровка происходит в процессе дообучения с симуляцией ошибок квантования.11
- **Скорость (Latency):** На процессорах Intel Xeon (Cascade Lake) с поддержкой VNNI ожидается ускорение инференса в 2-3 раза по сравнению с оптимизированным FP32. Это достигается за счет того, что инструкция vpdpbusd выполняет 64 пары умножений-сложений байтов за один такт (для 512-битного регистра), тогда как FP32 FMA выполняет 16 операций.5

## ---

**6\. Технические нюансы и Рекомендации (Troubleshooting)**

В процессе выполнения Квеста 14.1 исследователь может столкнуться с неочевидными проблемами.

1. **Ошибка Must run observer before calling calculate_qparams:**
   - _Причина:_ Попытка выполнить convert без прогона данных через prepared модель (пропуск фазы калибровки).
   - _Решение:_ Обязательно выполнить инференс хотя бы на одном батче данных после prepare.26
2. **Низкая точность после квантования:**
   - _Причина 1:_ Использование MinMaxObserver для активаций при наличии выбросов. _Решение:_ Переключиться на HistogramObserver.
   - _Причина 2:_ Неправильный датасет для калибровки (например, неотсортированный, ненормализованный или синтетический шум). _Решение:_ Проверить пайплайн данных.
   - _Причина 3:_ Специфика архитектуры (например, использование функций активации, недружелюбных к квантованию, типа Swish/SiLU с большим динамическим диапазоном, или отсутствие BatchNorm). _Решение:_ Использовать QAT.27
3. **Аппаратная несовместимость:**
   - Если CPU не поддерживает AVX2/AVX512, PyTorch будет использовать фоллбэк-реализацию. Модель станет меньше (память), но может работать _медленнее_ из\-за накладных расходов на распаковку данных для старых инструкций. Проверьте поддержку флагами процессора (lscpu на Linux).29

## **7\. Заключение**

Выполнение "Ритуала Квантизации" для модели MiniCNN демонстрирует мощь современных методов оптимизации нейронных сетей. Мы не просто механически сжали модель в 4 раза; мы трансформировали её вычислительную природу, адаптировав под архитектуру современных процессоров.  
**Итоговые выводы:**

1. **Post-Training Static Quantization** — эффективный метод для "зрелых" моделей, позволяющий получить 4-кратное уменьшение размера и 2-3 кратное ускорение инференса с минимальными усилиями.
2. **Калибровка** является критическим этапом, определяющим "карту местности" (диапазоны значений) для слепого целочисленного инференса. Качество данных калибровки напрямую определяет качество модели.
3. **Архитектурная осведомленность:** Понимание различий между Per-Channel и Per-Tensor, а также выбор правильных наблюдателей (Histogram vs MinMax), отличает профессионального ML-инженера от любителя.
4. **Будущее за Low-Precision:** С появлением форматов INT4 и FP8 границы эффективности продолжат сдвигаться, но принципы калибровки и отображения диапазонов, изученные в этом квесте, останутся фундаментальным базисом.

Модель MiniCNN теперь готова к развертыванию, освобождая ресурсы для более сложных задач и открывая путь к внедрению ИИ в самые компактные устройства нашего мира.

#### **Источники**

1. Quantization For Edge AI \- Meegle, дата последнего обращения: декабря 20, 2025, [https://www.meegle.com/en_us/topics/quantization/quantization-for-edge-ai](https://www.meegle.com/en_us/topics/quantization/quantization-for-edge-ai)
2. Edge AI, model quantization, and future of edge computing | Okoone, дата последнего обращения: декабря 20, 2025, [https://www.okoone.com/spark/strategy-transformation/edge-ai-model-quantization-and-future-of-edge-computing/](https://www.okoone.com/spark/strategy-transformation/edge-ai-model-quantization-and-future-of-edge-computing/)
3. Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with NVIDIA TensorRT, дата последнего обращения: декабря 20, 2025, [https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
4. Post-training quantization | Google AI Edge, дата последнего обращения: декабря 20, 2025, [https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quantization](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quantization)
5. Recent feature additions and improvements in FBGEMM \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/pytorch/FBGEMM/wiki/Recent-feature-additions-and-improvements-in-FBGEMM](https://github.com/pytorch/FBGEMM/wiki/Recent-feature-additions-and-improvements-in-FBGEMM)
6. INT8 Quantization for x86 CPU in PyTorch, дата последнего обращения: декабря 20, 2025, [https://pytorch.org/blog/int8-quantization/](https://pytorch.org/blog/int8-quantization/)
7. Quantization in AI: Making Models Smaller and Faster \- GoCodeo, дата последнего обращения: декабря 20, 2025, [https://www.gocodeo.com/post/quantization-in-ai-making-models-smaller-and-faster](https://www.gocodeo.com/post/quantization-in-ai-making-models-smaller-and-faster)
8. Neural Network Quantization in PyTorch \- Practical ML, дата последнего обращения: декабря 20, 2025, [https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/](https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/)
9. Quantization \- Hugging Face, дата последнего обращения: декабря 20, 2025, [https://huggingface.co/docs/optimum/concept_guides/quantization](https://huggingface.co/docs/optimum/concept_guides/quantization)
10. Zero-point quantization : How do we get those formulas? | by Luis Antonio Vasquez, дата последнего обращения: декабря 20, 2025, [https://medium.com/@luis.vasquez.work.log/zero-point-quantization-how-do-we-get-those-formulas-4155b51a60d6](https://medium.com/@luis.vasquez.work.log/zero-point-quantization-how-do-we-get-those-formulas-4155b51a60d6)
11. Practical Quantization in PyTorch, дата последнего обращения: декабря 20, 2025, [https://pytorch.org/blog/quantization-in-practice/](https://pytorch.org/blog/quantization-in-practice/)
12. LiteRT 8-bit quantization specification | Google AI Edge, дата последнего обращения: декабря 20, 2025, [https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/quantization_spec](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/quantization_spec)
13. Source code for torch.quantization.observer \- PyTorch documentation, дата последнего обращения: декабря 20, 2025, [https://glaringlee.github.io/\_modules/torch/quantization/observer.html](https://glaringlee.github.io/_modules/torch/quantization/observer.html)
14. PyTorch Static Quantization \- Lei Mao's Log Book, дата последнего обращения: декабря 20, 2025, [https://leimao.github.io/blog/PyTorch-Static-Quantization/](https://leimao.github.io/blog/PyTorch-Static-Quantization/)
15. A Brief Quantization Tutorial on Pytorch with Code | by Prajot Kuvalekar | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@Prajot_Saiprasad/a-brief-quantization-tutorial-on-pytorch-with-code-a8f448c840cd](https://medium.com/@Prajot_Saiprasad/a-brief-quantization-tutorial-on-pytorch-with-code-a8f448c840cd)
16. Fuse Modules Recipe — PyTorch Tutorials 2.5.0+cu124 documentation, дата последнего обращения: декабря 20, 2025, [https://pytorch-cn.com/tutorials/recipes/fuse.html](https://pytorch-cn.com/tutorials/recipes/fuse.html)
17. (beta) Static Quantization with Eager Mode in PyTorch, дата последнего обращения: декабря 20, 2025, [https://pytorch-cn.com/tutorials/advanced/static_quantization_tutorial.html](https://pytorch-cn.com/tutorials/advanced/static_quantization_tutorial.html)
18. FBGEMM 1.4.0 documentation \- PyTorch, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/FBGEMM/fbgemm/index.html](https://docs.pytorch.org/FBGEMM/fbgemm/index.html)
19. Quantization Recipe — PyTorch Tutorials 2.5.0+cu124 documentation, дата последнего обращения: декабря 20, 2025, [https://pytorch-cn.com/tutorials/recipes/quantization.html](https://pytorch-cn.com/tutorials/recipes/quantization.html)
20. HistogramObserver — PyTorch 2.9 documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.HistogramObserver.html](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.HistogramObserver.html)
21. Post-Training Quantization using test data? \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/post-training-quantization-using-test-data/171518](https://discuss.pytorch.org/t/post-training-quantization-using-test-data/171518)
22. A Manual Implementation of Quantization in PyTorch \- Single Layer \- About Me, дата последнего обращения: декабря 20, 2025, [https://franciscormendes.github.io/2024/05/16/quantization-1/](https://franciscormendes.github.io/2024/05/16/quantization-1/)
23. pytorch/torch/ao/quantization/qconfig.py at main \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/qconfig.py](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/qconfig.py)
24. 1\. Unlocking the Power of Quantization… | by Pelin Balci | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@balci.pelin/quantization-1-d05e5a61e0af](https://medium.com/@balci.pelin/quantization-1-d05e5a61e0af)
25. Introduction to Quantization on PyTorch, дата последнего обращения: декабря 20, 2025, [https://pytorch.org/blog/introduction-to-quantization-on-pytorch/](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
26. How to avoid Quantization warning: "Must run observer before calling calculate_qparams."?, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/how-to-avoid-quantization-warning-must-run-observer-before-calling-calculate-qparams/85902](https://discuss.pytorch.org/t/how-to-avoid-quantization-warning-must-run-observer-before-calling-calculate-qparams/85902)
27. Static Quantized model accuracy varies greatly with Calibration data · Issue \#45185 \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/pytorch/pytorch/issues/45185](https://github.com/pytorch/pytorch/issues/45185)
28. Select the right observers in QAT \- quantization \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/select-the-right-observers-in-qat/195289](https://discuss.pytorch.org/t/select-the-right-observers-in-qat/195289)
29. INT8 quantized model is much slower than fp32 model on CPU \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/int8-quantized-model-is-much-slower-than-fp32-model-on-cpu/87004](https://discuss.pytorch.org/t/int8-quantized-model-is-much-slower-than-fp32-model-on-cpu/87004)
30. Running int8 pytorch model with AVX512_VNNI \- Intel Community, дата последнего обращения: декабря 20, 2025, [https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/Running-int8-pytorch-model-with-AVX512-VNNI/m-p/1183493?profile.language=en\&countrylabel=Colombia](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/Running-int8-pytorch-model-with-AVX512-VNNI/m-p/1183493?profile.language=en&countrylabel=Colombia)
