# **Отчет об исследовании: Продвинутая отладка производительности глубокого обучения с использованием PyTorch Profiler**

## **Аннотация**

В современной экосистеме разработки систем искусственного интеллекта разрыв между теоретической производительностью аппаратного обеспечения (GPU/TPU) и реальной пропускной способностью моделей часто достигает порядков величины. Инструментарий torch.profiler представляет собой критически важный механизм для преодоления этого разрыва, предоставляя инженерам возможность наблюдать за асинхронным взаимодействием между хост-системой (CPU) и ускорителем (GPU). Данный отчет представляет собой исчерпывающее руководство по методологии профилирования, охватывающее теоретические основы гетерогенных вычислений, анатомию накладных расходов драйверов (overhead), анализ узких мест памяти и продвинутые стратегии оптимизации, такие как torch.compile и CUDA Graphs. На основе анализа десятков академических и технических источников 1, отчет детализирует процессы диагностики "узких мест" (bottlenecks), предлагая строгую таксономию состояний простоя GPU и стратегии их устранения для достижения максимальной утилизации кремния.

## ---

**Глава 1\. Теоретические основы производительности гетерогенных систем**

Для эффективной интерпретации данных, предоставляемых torch.profiler, необходимо фундаментальное понимание архитектуры исполнения, лежащей в основе современных фреймворков глубокого обучения. PyTorch, как и другие современные библиотеки, скрывает сложность управления аппаратным обеспечением за высокоуровневыми абстракциями, однако именно на стыке этих абстракций и "железа" возникают наиболее существенные потери производительности.

### **1.1. Дихотомия Хост-Устройство (Host-Device)**

Архитектура вычислений с ускорением базируется на модели разделения обязанностей между центральным процессором (Host) и графическим процессором (Device). Это не просто два разных процессора; это две независимые вычислительные системы с собственными пространствами памяти, тактовыми частотами и архитектурными парадигмами.

- **Хост (CPU):** Отвечает за оркестрацию. Он выполняет интерпретатор Python, управляет загрузкой данных, предпроцессингом и, что критически важно, формирует команды для GPU. CPU оптимизирован для последовательной обработки и сложной логики ветвления.
- **Устройство (GPU):** Представляет собой массивный параллельный процессор, оптимизированный для пропускной способности (throughput). Он способен выполнять тысячи потоков одновременно, но требует постоянного потока инструкций для поддержания загрузки.2

Связующим звеном выступает шина PCIe (обычно Gen4 или Gen5), которая, несмотря на высокую пропускную способность (до 64 ГБ/с для Gen4 x16), обладает ненулевой латентностью. Каждая передача данных или управляющей команды через этот интерфейс является "дорогой" операцией по сравнению с доступом к регистрам или L1-кэшу.

### **1.2. Асинхронная модель исполнения и очереди команд**

Ключевым механизмом, позволяющим нивелировать латентность шины PCIe и разницу в скоростях CPU и GPU, является асинхронность. Когда в коде PyTorch выполняется операция, например output \= model(input), выполнение на CPU не блокируется до завершения вычислений на GPU. Вместо этого происходит следующее:

1. **Сериализация:** PyTorch преобразует вызов Python-функции в последовательность инструкций CUDA.
2. **Диспетчеризация (Dispatch):** Инструкции помещаются в очередь команд (CUDA Stream), управляемую драйвером NVIDIA.
3. **Возврат управления:** CPU немедленно переходит к следующей строке кода, в то время как GPU асинхронно выбирает команды из очереди и исполняет их.

В идеальном сценарии CPU работает "на опережение", постоянно пополняя очередь команд, чтобы GPU никогда не простаивал. Это состояние называется **насыщением GPU** (GPU Saturation). Однако, если CPU не успевает генерировать команды с той скоростью, с которой GPU их исполняет, очередь пустеет, и GPU переходит в состояние простоя (Idle). Именно это состояние torch.profiler визуализирует как "пробелы" или "пустоты" на временной шкале, и именно оно является главной целью оптимизации.5

### **1.3. Латентность запуска ядра (Kernel Launch Latency)**

Фундаментальным ограничением асинхронной модели является время, необходимое для самого акта отправки команды. Это время, называемое "накладными расходами на запуск ядра" (kernel launch overhead), включает в себя время работы интерпретатора Python, диспетчера PyTorch (ATen), драйвера CUDA и физической передачи сигнала по шине PCIe.  
Согласно микро-бенчмаркам, минимальная латентность запуска пустого ядра (null kernel) составляет около **4-5 микросекунд** на Linux и может превышать **10 микросекунд** на Windows из\-за особенностей драйверной модели WDDM.2 Хотя 5 микросекунд кажутся пренебрежимо малыми, в контексте глубокого обучения они могут стать фатальными. Если выполнение самого вычислительного ядра (например, сложение двух небольших тензоров) занимает 1 микросекунду, то система тратит 80% времени на накладные расходы и лишь 20% на полезную работу. Этот феномен, известный как **"Launch-Bound"** (ограничение по запуску), является одной из самых сложных и распространенных проблем, диагностируемых с помощью профилировщика.7

## ---

**Глава 2\. Архитектура и конфигурация PyTorch Profiler**

Инструмент torch.profiler является основным средством наблюдения за описанными выше процессами. Он построен на базе библиотеки Kineto (разработанной Meta), которая объединяет высокоуровневые события Python с низкоуровневыми метриками GPU, полученными через CUPTI (CUDA Profiling Tools Interface).3

### **2.1. Контекстный менеджер и параметры захвата**

Профилирование инициируется через контекстный менеджер profile, который требует тщательной настройки для получения репрезентативных данных. Ошибки в конфигурации могут привести к получению "шумных" данных, отражающих не реальную производительность модели, а артефакты инициализации (JIT-компиляция, аллокация памяти).

#### **Таблица 2.1. Ключевые параметры конфигурации torch.profiler**

| Параметр       | Описание и Рекомендации                                                                                                                                            | Влияние на производительность                                           | Источник |
| :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------- | :------- |
| activities     | Список отслеживаемых событий. Обязательно включать ProfilerActivity.CPU и ProfilerActivity.CUDA для корреляции хоста и устройства.                                 | Среднее. Добавляет хуки в диспетчер ATen и активирует CUPTI.            | 1        |
| schedule       | Управляет циклом профилирования (wait, warmup, active). Критически важен для исключения фазы "прогрева" (warmup), когда кэши и JIT еще не стабилизировались.       | Низкое (логическое управление).                                         | 11       |
| record_shapes  | Записывает размерности входных тензоров для каждого оператора. Необходим для понимания, почему один и тот же оператор (напр., MatMul) работает с разной скоростью. | Высокое. Требует копирования метаданных тензоров.                       | 1        |
| profile_memory | Отслеживает аллокации и освобождение памяти. Позволяет строить временную шкалу потребления VRAM и диагностировать фрагментацию.                                    | Очень высокое. Может существенно замедлить выполнение.                  | 10       |
| with_stack     | Записывает Python-стек вызовов для каждого оператора. Позволяет связать низкоуровневый кернел CUDA с конкретной строкой в train.py.                                | Критическое. Значительно увеличивает размер трассы и накладные расходы. | 14       |

### **2.2. Стратегия планирования (Scheduling)**

Использование объекта torch.profiler.schedule является обязательным для корректного анализа. Типичная конфигурация выглядит следующим образом:

Python

schedule \= torch.profiler.schedule(  
 wait=1, \# Пропустить первый шаг (инициализация структур данных)  
 warmup=1, \# Прогрев (компиляция JIT-ядер, аллокация кэшей)  
 active=3, \# Запись данных (активная фаза)  
 repeat=1 \# Количество циклов записи  
)

Игнорирование фазы warmup приводит к тому, что профилировщик захватывает время компиляции ядер CUDA (PTX assembly) и первичную аллокацию памяти, что искажает представление о производительности в установившемся режиме (steady state).12

### **2.3. Инструментация кода**

Для повышения читаемости трассы рекомендуется использовать контекстный менеджер record_function для разметки логических блоков кода. Это позволяет группировать тысячи мелких операторов в понятные блоки, такие как "Forward Pass", "Loss Calculation", "Backward Pass", "Optimizer Step".

Python

with torch.profiler.profile(...) as prof:  
 for step, batch in enumerate(dataloader):  
 with torch.profiler.record_function("model_inference"):  
 output \= model(batch)  
 \#...  
 prof.step()

Без такой разметки анализ трассы превращается в поиск иглы в стоге сена среди тысяч вызовов aten::add и aten::mul.1

## ---

**Глава 3\. Феноменология узких мест: Анализ временной шкалы**

Визуализация трассы (обычно через TensorBoard или chrome://tracing) предоставляет наиболее интуитивный способ диагностики. Основной метод анализа — поиск "пробелов" (gaps) на временной шкале GPU. В полностью оптимизированной системе полоса GPU (GPU Timeline) должна выглядеть как сплошной кирпич активности без просветов. Любой пробел означает, что дорогостоящее оборудование простаивает.

### **3.1. Классификация состояний простоя**

Опираясь на данные профилирования 15, можно выделить три основных класса проблем, приводящих к низкой утилизации GPU:

1. **CPU-Bound (Ограничение процессором):** GPU выполняет работу быстрее, чем CPU успевает её подготовить.
   - _Признаки:_ Длительные пробелы между блоками ядер на GPU. На полосе CPU в это время видна высокая активность (длинные бары выполнения функций), не связанная с запуском ядер (например, DataLoader, аугментация изображений, сложная логика Python).
2. **Launch-Bound (Ограничение запуском):** CPU занят на 100%, но делает полезную работу, а тратит время на накладные расходы запуска тысяч мелких ядер.
   - _Признаки:_ Полоса GPU выглядит как "конфетти" или "пунктирная линия" — множество крошечных ядер с микроскопическими промежутками. Полоса CPU забита вызовами cudaLaunchKernel или диспетчером ATen.
3. **Memory-Bound (Ограничение памятью):** Ядра выполняются, но их длительность аномально велика по сравнению с количеством арифметических операций.
   - _Признаки:_ Длинные ядра на GPU, которые относятся к операциям пересылки данных (Memcpy) или элементарным операциям (aten::copy\_, aten::add), занимающим несоразмерно много времени.

### **3.2. Интерпретация таблицы key_averages**

Табличное представление данных является мощным инструментом количественного анализа. Важно различать метрики Self CPU Time и CPU Time Total.

- **Self CPU Time:** Время, проведенное внутри самой функции, исключая вызовы подфункций. Высокое значение здесь для cudaLaunchKernel — верный признак проблемы Launch-Bound.12
- **CPU Time Total:** Общее время выполнения функции от входа до выхода. Высокое значение для верхнеуровневых функций (например, model.forward) нормально, но если оно значительно превышает CUDA Time Total, это сигнал о том, что CPU является бутылочным горлышком.

#### **Таблица 3.2. Пример интерпретации метрик профилировщика**

| Событие / Оператор | Self CPU % | CUDA Time Total  | Диагноз                                                                        |
| :----------------- | :--------- | :--------------- | :----------------------------------------------------------------------------- |
| aten::conv2d       | \< 1%      | Высокое          | **Норма.** Тяжелая операция выполняется на GPU, CPU лишь запускает её.         |
| DataLoader         | Высокое    | 0%               | **CPU-Bound.** Процессор занят загрузкой данных, GPU ждет.                     |
| cudaLaunchKernel   | \> 30%     | Низкое (в сумме) | **Launch-Bound.** Слишком много мелких операций. Накладные расходы доминируют. |
| Memcpy HtoD        | \< 5%      | Высокое          | **PCIe Bottleneck.** Передача данных занимает больше времени, чем вычисления.  |

## ---

**Глава 4\. Проблема мелких ядер и накладные расходы (Launch Overhead)**

Одной из самых неочевидных проблем, выявляемых torch.profiler, является ситуация, когда мощный GPU работает на 10-20% своей мощности из\-за "смерти от тысячи порезов" (death by a thousand cuts). Это происходит, когда архитектура модели требует выполнения тысяч последовательных мелких операций.

### **4.1. Анатомия cudaLaunchKernel**

Когда в профилировщике оператор cudaLaunchKernel занимает верхние строчки по потреблению CPU, это означает, что процессор тратит всё свое время на "рукопожатие" с драйвером. Как обсуждалось в Главе 1, каждый запуск ядра стоит около 4-10 микросекунд.7  
Рассмотрим пример цикла LSTM или рекуррентной обработки, где на каждом шаге выполняется матричное умножение небольших векторов. Если размерность вектора мала (например, hidden size \= 64), то само умножение на GPU (ядро gemv) может занять 0.5 микросекунды. При накладных расходах в 5 микросекунд эффективность использования оборудования составляет:

$$\\text{Efficiency} \= \\frac{T\_{compute}}{T\_{compute} \+ T\_{launch}} \= \\frac{0.5}{0.5 \+ 5.0} \\approx 9\\%$$  
В таком режиме GPU проводит 91% времени в ожидании следующей команды. Профилировщик покажет это как огромное количество вызовов cudaLaunchKernel с высоким Self CPU Time.18

### **4.2. Влияние операционной системы и драйверов**

Исследования 2 показывают существенную разницу в накладных расходах между Linux и Windows. На Windows драйверная модель WDDM (Windows Display Driver Model) вводит дополнительную латентность из\-за сложной системы виртуализации GPU и планирования. Это может увеличивать overhead до 10-20 микросекунд. Именно поэтому для высокопроизводительных тренировок рекомендуется использовать Linux или режим TCC (Tesla Compute Cluster) на Windows, который обходит графическую подсистему WDDM.

### **4.3. Решения: Fusion и JIT**

Для борьбы с overhead необходимо уменьшить количество запусков ядер, не уменьшая объем полезной работы. Это достигается путем **слияния ядер (Kernel Fusion)**. Вместо запуска трех отдельных ядер для операций Mul \-\> Add \-\> ReLU, компилятор может сгенерировать одно ядро, которое выполняет все три операции за один проход по данным.

- **TorchScript / JIT:** Использование @torch.jit.script позволяет "запечь" последовательность операций в граф, который исполнителю проще оптимизировать.
- **torch.compile:** Новейший компилятор в PyTorch 2.0 использует backend Inductor для автоматической генерации слитных ядер Triton, часто устраняя overhead практически полностью.20 В профилировщике это отображается как исчезновение множества мелких aten:: операторов и появление крупных блоков с именами вроде triton_poi_fused\_....

## ---

**Глава 5\. Пропускная способность и Стратегия Батчинга (Batching)**

Если накладные расходы на запуск являются фиксированной величиной (константой), то самым простым способом уменьшить их относительное влияние является увеличение объема полезной работы в каждом запуске. Это подводит нас к концепции **размера батча (Batch Size)** как главного рычага оптимизации.

### **5.1. Амортизация накладных расходов**

Увеличение размера батча работает по принципу амортизации. Запуск ядра матричного умножения для батча размером 1 и для батча размером 64 стоит одинаково с точки зрения CPU (те же 5 мкс). Однако на GPU вычисления для батча 64 займут в 64 раза больше ресурсов (в идеале) или просто задействуют больше параллельных ядер CUDA, которые ранее простаивали.  
Экспериментальные данные 21 на примере конвейеров Hugging Face демонстрируют драматический рост пропускной способности:

- Batch Size 1: 187 итераций/сек (Огромное влияние overhead).
- Batch Size 64: 2478 итераций/сек (Overhead размазан по 64 примерам).

### **5.2. Кейс: Оптимизация Hugging Face Pipeline**

Библиотека transformers от Hugging Face предоставляет удобный API pipeline, который по умолчанию работает с batch_size=1. Это классическая ловушка для новичков.  
Профилирование стандартного пайплайна:

Python

classifier \= pipeline("sentiment-analysis")  
results \= classifier(dataset)

покажет трассу, полную пробелов. CPU тратит время на токенизацию одного предложения, запуск модели, получение результата, и только затем берется за следующее. GPU простаивает 80% времени.  
Оптимизированный подход с использованием batch_size:

Python

classifier \= pipeline("sentiment-analysis", batch_size=64, device=0)  
for out in classifier(KeyDataset(dataset, "text"), batch_size=64):  
 pass

Здесь профилировщик покажет плотную упаковку ядер. Более того, использование ChunkPipeline для задач типа QA позволяет обрабатывать длинные документы, разбивая их на куски, но сохраняя эффективность батчинга.21  
Однако существует предел. Увеличение батча линейно увеличивает потребление памяти (активации). Если выйти за пределы VRAM, возникнет ошибка OOM (Out Of Memory). Кроме того, при определенном размере батча GPU становится полностью насыщенным (Compute-Bound), и дальнейшее увеличение батча лишь увеличивает латентность (время отклика) без роста пропускной способности (throughput).23

## ---

**Глава 6\. Управление Памятью и OOM**

torch.profiler с флагом profile_memory=True предоставляет бесценную информацию о динамике потребления памяти. Это критически важно, так как многие модели ограничены не вычислительной мощностью, а пропускной способностью памяти (Memory Bandwidth Bound).

### **6.1. Временная шкала памяти (Memory Timeline)**

Визуализация памяти позволяет увидеть "зубчатую" структуру потребления.

- **Активации:** Память растет во время прямого прохода (Forward), так как промежуточные результаты сохраняются для вычисления градиентов.
- **Пик:** Максимум достигается в конце Forward pass, перед началом Backward.
- **Освобождение:** Во время обратного прохода (Backward) память освобождается по мере того, как градиенты вычислены и активации больше не нужны.
- **Градиенты и Оптимизатор:** В конце шага память снова подскакивает при обновлении весов оптимизатором (который может хранить копии моментов, как в Adam).

Анализ этой шкалы позволяет выявить утечки памяти (если "дно" графика растет от шага к шагу) и чрезмерную фрагментацию.13

### **6.2. Случайная синхронизация (Accidental Synchronization)**

Профилирование памяти часто выявляет скрытого врага производительности: неявную синхронизацию данных.  
Пример:

Python

print(f"Loss: {loss.item()}")

Вызов .item() требует переноса скаляра с GPU на CPU. Чтобы получить это значение, CPU должен остановить выполнение, дождаться завершения всех ядер на GPU, скопировать данные и только потом продолжить.  
На временной шкале профилировщика это выглядит как огромный провал активности GPU после вычисления loss, сопровождающийся событием Memcpy DtoH (Device to Host) на CPU. Избегание таких операций внутри цикла обучения (или выполнение их раз в N шагов) — простейшая и самая эффективная оптимизация.26

## ---

**Глава 7\. Продвинутые техники: CUDA Graphs и HTA**

Когда простые методы (батчинг, удаление синхронизаций) исчерпаны, в бой вступают тяжеловесные инструменты.

### **7.1. CUDA Graphs**

Технология CUDA Graphs, доступная в PyTorch, позволяет радикально решить проблему Launch Overhead. Вместо того чтобы CPU запускал каждое ядро по отдельности каждый шаг, вся последовательность запусков "записывается" в граф один раз.  
Во время исполнения CPU отправляет одну команду: "Запустить Граф". GPU выполняет всю цепочку ядер самостоятельно, без участия CPU.  
В профилировщике это выглядит как исчезновение тысяч вызовов cudaLaunchKernel. Вместо них на CPU появляется один вызов cudaGraphLaunch, а на GPU — плотный блок ядер без малейших зазоров. Это особенно эффективно для моделей с фиксированным графом вычислений и статичными размерами входных данных.4

### **7.2. Holistic Trace Analysis (HTA)**

Для анализа распределенных тренировок на кластерах GPU, стандартного просмотрщика Chrome Tracing может быть недостаточно. Библиотека Holistic Trace Analysis (HTA), разработанная на основе данных Kineto, позволяет программно анализировать трассы. Она может автоматически выявлять:

- Дисбаланс нагрузки между разными GPU (Stragglers).
- Анализ коммуникационных ядер (NCCL) и их перекрытие с вычислениями (Compute-Communication Overlap).
- Автоматический расчет метрик эффективности ядра (Kernel Duration Distribution).3

## ---

**Глава 8\. Операционная стратегия отладки (Playbook)**

На основе проанализированных материалов, предлагается следующий алгоритм действий для инженера по производительности:

1. **Базовая линия (Baseline):** Запустите "прогревочный" прогон модели. Включите профилировщик с schedule(wait=1, warmup=1, active=3).
2. **Макро-анализ:** Откройте трассу в TensorBoard.
   - Есть ли пробелы на GPU?
     - _Да, большие:_ Проверьте DataLoader (CPU-Bound) или синхронизации (.item()).
     - _Да, микроскопические, но много:_ Проверьте cudaLaunchKernel (Launch-Bound).
     - _Нет:_ Вы достигли насыщения (Compute-Bound).
3. **Анализ данных:**
   - Сортируйте key_averages по Self CPU Time. Если в топе cudaLaunchKernel — увеличивайте батч или используйте torch.compile.
   - Сортируйте по Self CUDA Time. Если в топе aten::copy\_ или Memcpy — оптимизируйте движение данных.
4. **Итеративная оптимизация:** Применяйте изменения по одному. Изменили num_workers \-\> Замерили. Включили pin_memory \-\> Замерили. Увеличили батч \-\> Замерили.

## ---

**Заключение**

Использование torch.profiler трансформирует процесс оптимизации из гадания на кофейной гуще в строгую инженерную дисциплину. Понимание физических ограничений интерфейсов (PCIe), накладных расходов драйверов и архитектуры памяти позволяет разработчику не просто "настраивать гиперпараметры", а целенаправленно устранять узкие места. Как показал анализ, переход от наивной реализации к оптимизированной (с использованием батчинга, фьюзинга и графов) может ускорить обучение и инференс в десятки раз, что в масштабах современных дата-центров эквивалентно экономии миллионов долларов и лет вычислительного времени.

### ---

**Список использованных источников (Citations)**

1 PyTorch Profiler Recipe (Huang)  
2 ICPP 2019 Paper on Kernel Overhead  
4 Massed Compute FAQ  
18 StackOverflow: cudaLaunchKernel  
21 Hugging Face Pipelines Docs  
22 KDNuggets: Optimizing HF Pipelines  
15 Intel GPA: CPU vs GPU Bound  
16 Eunomia: CPU/GPU Profiling Boundaries  
11 PyTorch Docs: Key Averages  
10 PyTorch Recipe: Execution Time  
14 PyTorch Beginner: Stack Traces  
12 Gao Hongnan: Profiling Blog  
10 PyTorch Recipe: Visualization  
13 PyTorch Blog: Memory Understanding  
27 Discuss PyTorch: Measuring Time  
20 PyTorch Docs: Torch Compile Profiling  
19 Discuss PyTorch: cudaLaunchKernel 99% CPU  
28 Discuss PyTorch: Interpreting Results  
9 Medium: Sub-millisecond latency  
5 NVIDIA Forums: Kernel Launch Latency Reasons  
7 NVIDIA Forums: Latency Measurement  
8 NVIDIA Forums: Windows vs Linux Latency  
23 Discuss PyTorch: Memory vs Batch Size  
6 NVIDIA Blog: Visualization of Overhead  
29 ICPP Poster: Kernel Launch Overheads  
3 PyTorch Blog: Holistic Trace Analysis  
17 Reddit: LocalLLaMA Profiling  
24 Hyperstack: Optimizing LLM Inference  
30 Arxiv: MoE Inference  
25 Databricks: LLM Inference Performance  
21 Hugging Face Pipelines Batching  
7 NVIDIA Forums Summary: Latency Stats

#### **Источники**

1. PyTorch Profiler — PyTorch Tutorials 1.8.1+cu102 documentation \- h-huang.github.io, дата последнего обращения: декабря 22, 2025, [https://h-huang.github.io/tutorials/recipes/recipes/profiler_recipe.html](https://h-huang.github.io/tutorials/recipes/recipes/profiler_recipe.html)
2. Understanding the Overheads of Launching CUDA Kernels, дата последнего обращения: декабря 22, 2025, [https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-abst.pdf](https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-abst.pdf)
3. PyTorch Trace Analysis for the Masses, дата последнего обращения: декабря 22, 2025, [https://pytorch.org/blog/trace-analysis-for-masses/](https://pytorch.org/blog/trace-analysis-for-masses/)
4. Can you explain the concept of CUDA kernel launch overhead and how it affects overall system performance? \- Massed Compute, дата последнего обращения: декабря 22, 2025, [https://massedcompute.com/faq-answers/?question=Can%20you%20explain%20the%20concept%20of%20CUDA%20kernel%20launch%20overhead%20and%20how%20it%20affects%20overall%20system%20performance?](https://massedcompute.com/faq-answers/?question=Can+you+explain+the+concept+of+CUDA+kernel+launch+overhead+and+how+it+affects+overall+system+performance?)
5. What are possible reasons of heavy kernel launch latency? \- CUDA Programming and Performance \- NVIDIA Developer Forums, дата последнего обращения: декабря 22, 2025, [https://forums.developer.nvidia.com/t/what-are-possible-reasons-of-heavy-kernel-launch-latency/287203](https://forums.developer.nvidia.com/t/what-are-possible-reasons-of-heavy-kernel-launch-latency/287203)
6. Understanding the Visualization of Overhead and Latency in NVIDIA Nsight Systems, дата последнего обращения: декабря 22, 2025, [https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)
7. Any way to measure the latency of a kernel launch? \- CUDA ..., дата последнего обращения: декабря 22, 2025, [https://forums.developer.nvidia.com/t/any-way-to-measure-the-latency-of-a-kernel-launch/221413](https://forums.developer.nvidia.com/t/any-way-to-measure-the-latency-of-a-kernel-launch/221413)
8. kernel launch latency \- CUDA Programming and Performance \- NVIDIA Developer Forums, дата последнего обращения: декабря 22, 2025, [https://forums.developer.nvidia.com/t/kernel-launch-latency/62455](https://forums.developer.nvidia.com/t/kernel-launch-latency/62455)
9. Building High-Performance CUDA Kernels for Low-Latency ML Inference: A Deep Technical Analysis | by Shreshthkapai | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@shreshthkapai/sub-millisecond-gpu-task-queue-breaking-pytorchs-latency-bottleneck-b6f3d3f2e895](https://medium.com/@shreshthkapai/sub-millisecond-gpu-task-queue-breaking-pytorchs-latency-bottleneck-b6f3d3f2e895)
10. PyTorch Profiler — PyTorch Tutorials 2.9.0+cu128 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
11. torch.profiler — PyTorch 2.9 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.pytorch.org/docs/stable/profiler.html](https://docs.pytorch.org/docs/stable/profiler.html)
12. PyTorch's Event And Profiler \- Omniverse, дата последнего обращения: декабря 22, 2025, [https://www.gaohongnan.com/operations/profiling/03_time_profiler.html](https://www.gaohongnan.com/operations/profiling/03_time_profiler.html)
13. Understanding GPU Memory 1: Visualizing All Allocations over Time \- PyTorch, дата последнего обращения: декабря 22, 2025, [https://pytorch.org/blog/understanding-gpu-memory-1/](https://pytorch.org/blog/understanding-gpu-memory-1/)
14. Profiling your PyTorch Module — PyTorch Tutorials 2.9.0+cu128 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.pytorch.org/tutorials/beginner/profiler.html](https://docs.pytorch.org/tutorials/beginner/profiler.html)
15. Identify Basic GPU-CPU Bound Scenarios \- Intel, дата последнего обращения: декабря 22, 2025, [https://www.intel.com/content/www/us/en/docs/gpa/cookbook/2022-4/identify-basic-gpu-cpu-bound-scenarios.html](https://www.intel.com/content/www/us/en/docs/gpa/cookbook/2022-4/identify-basic-gpu-cpu-bound-scenarios.html)
16. CPU and GPU Profiling Boundaries: What to Measure Where \- eunomia-bpf, дата последнего обращения: декабря 22, 2025, [https://eunomia.dev/others/cuda-tutorial/10-cpu-gpu-profiling-boundaries/](https://eunomia.dev/others/cuda-tutorial/10-cpu-gpu-profiling-boundaries/)
17. Profiling torch model: why is the GPU utilization so low? : r/LocalLLaMA \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1evdxuz/profiling_torch_model_why_is_the_gpu_utilization/](https://www.reddit.com/r/LocalLLaMA/comments/1evdxuz/profiling_torch_model_why_is_the_gpu_utilization/)
18. What is cudaLaunchKernel in pytorch profiler output \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/71328662/what-is-cudalaunchkernel-in-pytorch-profiler-output](https://stackoverflow.com/questions/71328662/what-is-cudalaunchkernel-in-pytorch-profiler-output)
19. cudaLaunchKernel takes 99% of CPU time \- vision \- PyTorch Forums, дата последнего обращения: декабря 22, 2025, [https://discuss.pytorch.org/t/cudalaunchkernel-takes-99-of-cpu-time/168307](https://discuss.pytorch.org/t/cudalaunchkernel-takes-99-of-cpu-time/168307)
20. Profiling to understand torch.compile performance — PyTorch 2.9 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html](https://docs.pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html)
21. Pipelines \- Hugging Face, дата последнего обращения: декабря 22, 2025, [https://huggingface.co/docs/transformers/en/main_classes/pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
22. 5 Tips for Building Optimized Hugging Face Transformer Pipelines \- KDnuggets, дата последнего обращения: декабря 22, 2025, [https://www.kdnuggets.com/5-tips-for-building-optimized-hugging-face-transformer-pipelines](https://www.kdnuggets.com/5-tips-for-building-optimized-hugging-face-transformer-pipelines)
23. Relationship between GPU Memory Usage and Batch Size \- quantization \- PyTorch Forums, дата последнего обращения: декабря 22, 2025, [https://discuss.pytorch.org/t/relationship-between-gpu-memory-usage-and-batch-size/132266](https://discuss.pytorch.org/t/relationship-between-gpu-memory-usage-and-batch-size/132266)
24. Optimise GPU Utilisation for LLM Inference: Batching with Llama2 vs Mixtral \- Hyperstack, дата последнего обращения: декабря 22, 2025, [https://www.hyperstack.cloud/technical-resources/tutorials/optimising-gpu-utilisation-for-llm-inference-llama2-vs-mixtral-series](https://www.hyperstack.cloud/technical-resources/tutorials/optimising-gpu-utilisation-for-llm-inference-llama2-vs-mixtral-series)
25. LLM Inference Performance Engineering: Best Practices | Databricks Blog, дата последнего обращения: декабря 22, 2025, [https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
26. Profiling with PyTorch \- Habana Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.habana.ai/en/latest/Profiling/Profiling_with_PyTorch.html](https://docs.habana.ai/en/latest/Profiling/Profiling_with_PyTorch.html)
27. How to measure time in PyTorch, дата последнего обращения: декабря 22, 2025, [https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964](https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964)
28. Interpreting profiler results \- deployment \- PyTorch Forums, дата последнего обращения: декабря 22, 2025, [https://discuss.pytorch.org/t/interpreting-profiler-results/187429](https://discuss.pytorch.org/t/interpreting-profiler-results/187429)
29. Understanding the Overheads of Launching CUDA Kernels, дата последнего обращения: декабря 22, 2025, [https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-moc.pdf](https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-moc.pdf)
30. MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs \- arXiv, дата последнего обращения: декабря 22, 2025, [https://arxiv.org/html/2411.11217v1](https://arxiv.org/html/2411.11217v1)
