# **Квест 19.2: Архитектура Возрождения — Полный Технический Анализ Интерактивных Голосовых Ассистентов в Терминале**

## **1\. Введение: Императив Техноманта**

В современной парадигме разработки искусственного интеллекта задача создания локального, полностью автономного голосового ассистента — часто именуемая в специализированных кругах как «Квест 19.2: Оживление» — представляет собой фундаментальный вызов, лежащий на стыке обработки сигналов, нейросетевого инференса и системного программирования. Данный отчет представляет собой исчерпывающий «гримуар» 1, описывающий процесс создания цифрового голема, способного к интерактивному взаимодействию в реальном времени через терминал.

В отличие от облачных решений, где латентность скрыта за сетевыми протоколами, локальное «возрождение» требует от инженера-техноманта глубокого понимания каждого такта процессора и каждого байта оперативной памяти. Мы переходим от статического исполнения скриптов к динамическому циклу жизни, где система, подобно живому организму, должна непрерывно слушать, фильтровать шум, вычленять смысл и реагировать, сохраняя при этом состояние контекста.

Анализ базируется на интеграции передовых открытых технологий: **Silero VAD** для детекции речи, **OpenAI Whisper** (в его оптимизированных ипостасях Faster-Whisper и WhisperX) для транскрипции, и **LLM** малого размера (таких как DistilGPT-2) для когнитивной обработки. Особое внимание уделяется архитектурным решениям, позволяющим избежать «проклятия бесконечного цикла» 2 и утечек памяти, свойственных длительным сессиям инференса на Python.3 Цель данного документа — предоставить детальную техническую карту для реализации ассистента, чья реакция и стабильность не уступают, а в аспектах приватности и автономности превосходят проприетарные аналоги.

## ---

**2\. Акустический Интерфейс: Сбор и Первичная Обработка Сигнала**

Фундаментом любого голосового ассистента является способность корректно воспринимать физическую реальность через аудиопоток. В контексте «оживления в терминале» это означает работу с низкоуровневыми аудиоинтерфейсами, буферизацией и преобразованием форматов данных в реальном времени. Ошибки на этом этапе фатальны: если голем «слышит» искаженный сигнал, никакая мощь нейронных сетей не сможет восстановить смысл.

### **2.1 Выбор Библиотеки Ввода-Вывода: PyAudio против SoundDevice**

В экосистеме Python исторически сложилось противостояние двух основных библиотек для работы с аудиопотоками: **PyAudio** и **SoundDevice**. Анализ их архитектуры показывает критические различия, влияющие на стабильность долгоживущих ассистентов.

**PyAudio**, являющаяся прямой оберткой над C-библиотекой PortAudio, долгое время была стандартом де\-факто. Она предоставляет гранулярный контроль над параметрами потока, однако её использование сопряжено с рядом архаичных проблем. Во-первых, библиотека работает с сырыми байтовыми строками, что требует дополнительных вычислительных затрат на конвертацию данных в форматы с плавающей точкой, необходимые для современных нейросетей.4 Во-вторых, PyAudio известна своей нестабильностью при блокирующих операциях и сложностью в управлении потоками вывода (stdout/stderr), что часто приводит к засорению терминала служебной информацией C-уровня, разрушая чистоту интерфейса «техноманта».5 Более того, проект характеризуется спорадической поддержкой, что создает риски совместимости с новыми версиями Python (например, 3.10+).5

**SoundDevice**, напротив, представляет собой эволюционный скачок. Эта библиотека также базируется на PortAudio, но изначально спроектирована для интеграции с экосистемой **NumPy**.7 Это архитектурное решение имеет решающее значение для Квеста 19.2. Поскольку модели VAD (Silero) и ASR (Whisper) ожидают на входе тензоры (массивы float32), использование SoundDevice позволяет получать данные непосредственно в нужном формате, минуя дорогостоящие операции копирования и преобразования типов в памяти.4 Кроме того, SoundDevice предлагает более надежный механизм callback-функций, который позволяет организовать неблокирующий захват аудио, критически важный для обеспечения отзывчивости интерфейса ассистента.

| Характеристика         | PyAudio                       | SoundDevice                         | Влияние на Квест 19.2                                                         |
| :--------------------- | :---------------------------- | :---------------------------------- | :---------------------------------------------------------------------------- |
| **Формат данных**      | Bytes (требует struct.unpack) | NumPy Arrays (float32)              | SoundDevice устраняет латентность конвертации, ускоряя препроцессинг для VAD. |
| **Управление памятью** | Ручное                        | Автоматическое (через буферы NumPy) | Снижение риска утечек памяти при длительной работе (Infinite Loop).           |
| **Обработка ошибок**   | C-style (Segmentation faults) | Python Exceptions                   | Более простая отладка «падений» ассистента в интерактивном режиме.            |
| **Интеграция**         | Низкая (требует оберток)      | Высокая (scipy, librosa)            | Прямая совместимость с математическим аппаратом VAD детекторов.               |

_Анализ показывает, что для задач реального времени, требующих высокой производительности и интеграции с нейросетями, SoundDevice является безальтернативным выбором, обеспечивающим необходимую «гигиену» кода и данных._

### **2.2 Физика Дискретизации: Проблема Частоты 16 кГц**

Следующим критическим параметром является частота дискретизации (sample rate). Модели, обученные на огромных корпусах речи (Whisper, Silero), стандартизированы под частоту **16,000 Гц** (16 кГц).10 Этот выбор не случаен: согласно теореме Котельникова (Найквиста-Шеннона), для корректной передачи человеческой речи, спектр которой редко превышает 8 кГц, частоты дискретизации в 16 кГц достаточно. Использование более высоких частот (44.1 или 48 кГц) не только не повышает точность распознавания, но и кратно увеличивает нагрузку на VAD и ASR, замедляя инференс.12

Однако, аппаратное обеспечение современных микрофонов часто фиксировано на частоте 48 кГц (стандарт DVD). Попытка передать такой сигнал в модель, ожидающую 16 кГц, приведет к эффекту «замедленной съемки» (pitch shift) и полной потере семантики.

**Стратегии Ресемплирования:**

1. **Аппаратный запрос:** Идеальный сценарий — запросить у драйвера устройства поток 16 кГц через параметры SoundDevice. Если устройство поддерживает нативную децимацию, это снимает нагрузку с CPU.13
2. **Программное ресемплирование:** Если аппаратная поддержка отсутствует, необходим алгоритм ресемплирования в реальном времени. Библиотека librosa предоставляет высококачественные алгоритмы (например, Kaiser window), но они могут быть слишком медленными для real-time loop.14 В контексте низколатентного ассистента рекомендуется использовать быстрые реализации из scipy.signal.resample или встроенные возможности soxr (если они доступны через SoundDevice), чтобы минимизировать задержку между получением чанка и его обработкой.16

### **2.3 Кольцевой Буфер: Математика Непрерывности**

Одной из главных проблем интерактивных систем является «проблема первого слога». Человек начинает говорить до того, как система детектирует речь. Если запись начинается строго по триггеру VAD, первые 200-300 миллисекунд (атака звука, первый согласный) теряются, что критично для коротких команд («Да», «Нет», «Стоп»).

Решением служит структура данных, известная как **Кольцевой Буфер (Ring Buffer)**.18 Это очередь фиксированного размера (например, collections.deque с параметром maxlen), которая хранит последние $N$ секунд аудиопотока. В режиме ожидания (SILENCE) система непрерывно пишет данные в этот буфер, автоматически удаляя устаревшие данные. Как только VAD детектирует голос, содержимое кольцевого буфера _мгновенно_ переносится в активный буфер записи, обеспечивая наличие контекста _до_ момента срабатывания детектора.19

- **Расчет размера буфера:** Для надежного захвата начала фразы рекомендуется хранить от 0.5 до 1.0 секунды аудио. При частоте 16 кГц и размере чанка 512 сэмплов, это соответствует очереди длиной примерно 30-60 чанков.20 Использование меньшего буфера (менее 100 мс) часто приводит к обрезанию взрывных согласных, что снижает точность Whisper.

## ---

**3\. Привратник: Архитектура Детекции Голосовой Активности (VAD)**

В цикле «Оживления» VAD (Voice Activity Detection) выполняет роль привратника. Без него тяжелая модель транскрипции (Whisper) будет пытаться распознать шум вентилятора или стук клавиатуры, что приведет к галлюцинациям и перегреву оборудования. Эффективный VAD должен быть быстрым, точным и устойчивым к шуму.

### **3.1 Сравнительный Анализ: Silero против WebRTC**

Исследование доступных решений выявляет два доминирующих подхода: классический статистический (WebRTC) и современный нейросетевой (Silero).

**WebRTC VAD** базируется на моделях гауссовых смесей (GMM). Он чрезвычайно быстр и легок, принимая решения на основе фреймов по 10-30 мс.10 Однако его природа делает его чувствительным к нестационарным шумам (например, резким звукам) и он часто дает ложные срабатывания или, наоборот, пропускает тихую речь.

**Silero VAD**, выбранный в качестве стандарта для данного Квеста, представляет собой глубокую нейронную сеть, обученную на огромном массиве данных (более 6000 языков).22

- **Архитектура:** Silero использует механизм внимания (Attention) или LSTM для анализа контекста, что позволяет ему отличать человеческую речь от механического шума с высокой точностью.
- **Производительность:** Несмотря на нейросетевую природу, Silero VAD чрезвычайно эффективен. Обработка одного чанка аудио (30+ мс) занимает менее 1 мс на одном ядре CPU, что делает его идеальным для выполнения в основном цикле программы (Main Loop) без блокировки интерфейса.23
- **Метрики:** Сравнительные тесты показывают, что Silero VAD значительно превосходит WebRTC по метрике AUC (Area Under Curve) в условиях зашумленности, обеспечивая баланс между чувствительностью и специфичностью.19

В сравнении с проприетарными решениями типа **Cobra (Picovoice)**, Silero выигрывает за счет открытости (MIT License) и отсутствия необходимости в облачных ключах или валидации лицензии, что соответствует духу создания независимого «Голема».22

### **3.2 Логика Машины Состояний (State Machine)**

Простая бинарная классификация (речь/тишина) от VAD недостаточна для построения диалога. Необходима реализация **Машины Состояний**, управляющей переходами между режимами ассистента.

1. **IDLE (Ожидание):** Аудио пишется в кольцевой буфер. VAD анализирует каждый чанк. Если вероятность речи превышает порог (threshold, обычно 0.5) в течение нескольких последовательных чанков (например, 3-5), происходит триггер speech_start.19
2. **LISTENING (Запись):** Данные из кольцевого буфера сбрасываются в линейный буфер фразы. Новые чанки добавляются туда же.
3. **PAUSE (Пауза):** Если VAD показывает отсутствие речи, система не должна немедленно обрывать запись. Человеческая речь прерывиста. Вводится понятие min_silence_duration (обычно 500-1000 мс).25
4. **PROCESSING (Обработка):** Только если длительность тишины превысила min_silence_duration, запись считается завершенной (speech_end). Буфер передается в Whisper, а система переходит в состояние обработки, блокируя новый ввод или (в продвинутых версиях) продолжая слушать для перебивания (barge-in).26

Такая логика предотвращает фрагментацию фраз и обеспечивает естественность взаимодействия, позволяя пользователю делать паузы для размышления без потери контекста.

## ---

**4\. Писарь: Автоматическое Распознавание Речи (ASR)**

После того как «Привратник» (VAD) выделил полезный сигнал, в дело вступает «Писарь» — модель автоматического распознавания речи. Выбор и оптимизация этой модели определяют не только точность понимания, но и воспринимаемую задержку (latency) системы.

### **4.1 Архитектура Whisper: Возможности и Ограничения**

Модель **Whisper** от OpenAI, основанная на архитектуре Transformer (Encoder-Decoder), является золотым стандартом для задач ASR общего назначения.27 Обученная на 680,000 часов слабо размеченных данных, она демонстрирует феноменальную устойчивость к акцентам, техническому жаргону и фоновому шуму.

Однако, «ванильная» реализация Whisper имеет архитектурные особенности, затрудняющие real-time использование:

- **Окно внимания:** Whisper оптимизирован для обработки 30-секундных фрагментов аудио. При подаче коротких команд (2-3 секунды) модель вынуждена либо дополнять (pad) входные данные до 30 секунд, либо использовать менее эффективные пути инференса, что создает накладные расходы.11
- **Галлюцинации тишины:** Известный баг Whisper — при подаче абсолютно тихих сегментов (если VAD сработал ложно) модель может начать генерировать повторяющиеся фразы или случайный текст, пытаясь «найти» смысл в шуме.2 Это требует жесткой фильтрации пустых транскрипций на уровне логики приложения.

### **4.2 Оптимизация Инференса: Faster-Whisper и WhisperX**

Для достижения интерактивной скорости (Latnecy \< 500 мс) стандартная реализация PyTorch часто оказывается слишком тяжелой. Анализ рекомендует использование оптимизированных движков.

- **Faster-Whisper:** Данная реализация переносит архитектуру Whisper на движок **CTranslate2**, специализированный на инференсе Трансформеров. Использование квантования весов (INT8) позволяет сократить потребление памяти в 2-4 раза и ускорить выполнение до 4 раз без существенной потери точности.30 Это критически важно для запуска ассистента на CPU или слабых GPU.
- **WhisperX:** Добавляет возможности пакетной обработки (batching) и выравнивания фонем. Хотя пакетная обработка более актуальна для транскрипции файлов, оптимизации памяти в WhisperX также полезны для потокового режима.19

Выбор размера модели (Model Size):  
Для терминального ассистента существует компромисс между интеллектом и скоростью:

- **Tiny / Base:** Мгновенная реакция, но возможны ошибки в сложных терминах. Идеальны для командного управления («Запусти сервер», «Открой vim»).
- **Small / Medium:** «Золотая середина». Достаточная точность для диктовки текста и сложных запросов, приемлемая скорость на современном CPU.
- **Large-v3:** Требует GPU (минимум 4-8 ГБ VRAM). Обеспечивает максимальную точность, но вносит заметную задержку, которая может разрушить магию «живого» диалога.28

## ---

**5\. Разум: Локальные Языковые Модели (LLM) и Генерация Ответа**

«Оживление» подразумевает наличие интеллекта. Полученный текст должен быть осмыслен, и на него должен быть дан ответ. Чтобы сохранить статус «локального голема», мы отказываемся от API (OpenAI/Claude) в пользу локальных моделей.

### **5.1 Дистилляция Знаний: DistilGPT-2**

В условиях ограниченных ресурсов (памяти и вычислительной мощности) использование гигантов вроде LLaMA-70B невозможно. Анализ указывает на **DistilGPT-2** как на кандидата для базовой логики. Это дистиллированная версия GPT-2, содержащая всего 82 миллиона параметров.32

- **Преимущества:** Экстремально высокая скорость инференса и низкое потребление памяти (менее 1 ГБ).
- **Недостатки:** Ограниченное контекстное окно (1024 токена) и склонность к потере нити разговора в длинных диалогах.33
- **Применение:** DistilGPT-2 отлично подходит для задач классификации интентов (Zero-Shot Classification) или простых диалоговых скриптов (ChatBench), но для сложной генеративной работы может потребоваться использование квантованных версий более современных моделей (например, Phi-2 или TinyLlama) через llama.cpp.

### **5.2 Zero-Shot Классификация Интентов**

Для управления ассистентом часто эффективнее не генерировать свободный текст, а классифицировать намерение пользователя. Модель **Bart-Large-MNLI** (или её дистиллированная версия valhalla/distilbart-mnli-12-3) позволяет реализовать Zero-Shot классификацию. Ассистент может определять, является ли фраза командой («открой браузер»), вопросом («какая погода») или светской беседой, без предварительного обучения на конкретных фразах.34 Это позволяет создавать жесткую логику управления поверх гибкого понимания языка.

## ---

**6\. Голос: Синтез Речи (TTS) и Проблема Прерывания**

Завершающий этап цикла — вокализация ответа. Здесь также важен баланс между качеством и скоростью.

- **gTTS (Google Text-to-Speech):** Простой вариант, но требует подключения к интернету, что нарушает концепцию полной автономности. Кроме того, имеет заметную задержку на сетевой запрос.35
- **Локальные TTS:** Для сохранения духа «Квеста» рекомендуются локальные решения, такие как **pyttsx3** (использует системные голоса) или более современные нейросетевые модели типа **Kokoro** или **VITS** (через Piper TTS). Они обеспечивают приемлемое качество при нулевой сетевой задержке.

Механизм Прерывания (Barge-in):  
Высший пилотаж в создании голосового ассистента — возможность перебить его. Если пользователь начинает говорить во время ответа ассистента, система должна это обнаружить.  
Это реализуется путем продолжения работы VAD во время синтеза речи. Если в процессе воспроизведения ответа VAD детектирует новый голос пользователя, поток воспроизведения (SoundDevice Output Stream) должен быть немедленно остановлен (stream.stop()), буферы очищены, и система должна перейти в состояние LISTENING.26 Это требует сложной многопоточной координации.

## ---

**7\. Архитектурный Синтез: Интерактивный Цикл**

Сборка всех компонентов воедино требует решения проблем конкурентности и управления памятью в Python.

### **7.1 Конкурентность: Asyncio против Threading**

Глобальная блокировка интерпретатора (GIL) в Python делает истинную параллельность на потоках (Threading) невозможной для CPU-bound задач (инференс нейросетей). Однако аудио ввод-вывод является I/O-bound задачей.

- **Рекомендуемая архитектура:**
  1. **Аудио-поток (Thread):** SoundDevice запускается в отдельном потоке через callback. Он заполняет потокобезопасную очередь (queue.Queue) сырыми данными.37
  2. **VAD-петля (Main Loop / Asyncio):** Основной цикл забирает данные из очереди, прогоняет через Silero VAD (который достаточно быстр, чтобы не блокировать цикл надолго) и управляет машиной состояний.39
  3. **Тяжелые вычисления (Process/Executor):** Инференс Whisper и LLM лучше выносить в отдельные процессы (multiprocessing) или использовать run_in_executor в asyncio, чтобы длительное "мышление" голема не останавливало обработку входящего аудио и работу VAD.41

### **7.2 Управление Памятью и Утечки**

Длительная работа PyTorch-моделей в цикле while True часто приводит к постепенному исчерпанию оперативной памяти (Memory Leak). Анализ показывает основные причины и методы борьбы:

- **Накопление графов:** PyTorch по умолчанию строит граф вычислений для обратного распространения ошибки. При инференсе это не нужно и ест память. Использование контекстного менеджера with torch.inference_mode(): (предпочтительнее устаревшего no_grad()) отключает эту функциональность, экономя память и ускоряя работу.42
- **Сборка мусора:** Явное удаление тензоров (del tensor) и вызов gc.collect() после каждого цикла диалога является хорошей практикой для предотвращения фрагментации памяти.3

## ---

**8\. Безопасность и Тестирование: Защита Голема**

Даже локальный ассистент уязвим. Внедрение промптов (Prompt Injection) через аудио может заставить ассистента выполнить нежелательные действия в терминале (например, rm \-rf /).

### **8.1 Регрессионное Тестирование (End-to-End)**

Для обеспечения надежности и безопасности необходимо внедрить автоматизированное тестирование. Ручная проверка голосом неэффективна и невоспроизводима.  
Методология:

1. **Генерация тестовых данных:** Использование TTS (например, gTTS) для создания набора аудиофайлов с командами, включая граничные случаи (тишина, шум, быстрый темп) и атаки (инъекции).46
2. **Инъекция в пайплайн:** Вместо микрофона, данные из wav-файлов подаются непосредственно в функцию обработки чанков. Это позволяет тестировать логику VAD и ASR в детерминированной среде.47
3. **Валидация:** Сравнение транскрипции Whisper с эталонным текстом (расчет WER \- Word Error Rate) и проверка реакции LLM на запрещенные команды.49

## ---

**9\. Гримуар: Протокол Реализации**

Ниже представлен алгоритмический план (псевдокод), синтезирующий все исследованные компоненты в единую структуру «Живого Терминала».

**Инициализация:**

1. Загрузка Silero VAD (ONNX).22
2. Загрузка Faster-Whisper (INT8, CPU/GPU).30
3. Инициализация RingBuffer (0.5-1.0 сек).19
4. Инициализация SoundDevice InputStream (16kHz, float32, blocksize=512).9

**Бесконечный Цикл (The Loop):**

1. Получение чанка audio_chunk из потока.
2. Добавление audio_chunk в RingBuffer.
3. prob \= VAD(audio_chunk).
4. **Если состояние SILENCE:**
   - Если prob \> 0.5 (триггер):
     - Состояние \-\> **LISTENING**.
     - Копирование RingBuffer в ActiveBuffer (решение проблемы первого слога).
5. **Если состояние LISTENING:**
   - Добавление audio_chunk в ActiveBuffer.
   - Если prob \< 0.5:
     - Инкремент счетчика тишины.
     - Если счетчик \> MAX_SILENCE (конец фразы):
       - Состояние \-\> **PROCESSING**.
       - text \= Whisper.transcribe(ActiveBuffer).
       - Очистка ActiveBuffer и сброс счетчиков.
       - Если text не пустой и не галлюцинация:
         - response \= LLM(text).
         - TTS.speak(response) (с проверкой на прерывание).
       - Состояние \-\> **SILENCE**.
       - gc.collect() (профилактика утечек).

## ---

**10\. Заключение**

Квест 19.2 демонстрирует, что создание полноценного голосового ассистента в терминале — это не магия, а результат строгой инженерной интеграции. Ключ к успеху лежит в правильном выборе инструментов: **SoundDevice** для низкоуровневой работы с данными, **Silero** для эффективной фильтрации, **Faster-Whisper** для быстрой транскрипции и **Кольцевого Буфера** для обеспечения временнóй целостности речи.

Реализация данной архитектуры позволяет создать «Голема», который не зависит от облачных серверов, уважает приватность пользователя и реагирует с задержкой, приближенной к человеческой. Это и есть истинное «Оживление» в терминале — превращение кода в собеседника.

---

**Анализ Источников и Ссылок:**

- **VAD (Точность и Сравнение):**.19
- **Аудио Буферизация и I/O:**.7
- **Оптимизация Whisper:**.11
- **LLM и Инференс:**.32
- **Тестирование и Безопасность:**.46

#### **Источники**

1. дата последнего обращения: декабря 21, 2025, [https://huggingface.co/zdouble/model/resolve/7763ed9c051f6e77f0ecca4d605f08cb37a563bf/billions_of_all_in_one.yaml?download=true](https://huggingface.co/zdouble/model/resolve/7763ed9c051f6e77f0ecca4d605f08cb37a563bf/billions_of_all_in_one.yaml?download=true)
2. Stops working after long gap with no speech? · openai whisper · Discussion \#29 \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/openai/whisper/discussions/29](https://github.com/openai/whisper/discussions/29)
3. Memory Leak Debugging and Common Causes \- PyTorch Forums, дата последнего обращения: декабря 21, 2025, [https://discuss.pytorch.org/t/memory-leak-debugging-and-common-causes/67339](https://discuss.pytorch.org/t/memory-leak-debugging-and-common-causes/67339)
4. Playing and Recording Sound in Python, дата последнего обращения: декабря 21, 2025, [https://realpython.com/playing-and-recording-sound-python/](https://realpython.com/playing-and-recording-sound-python/)
5. It's there a viable alternative to pyaudio? : r/Python \- Reddit, дата последнего обращения: декабря 21, 2025, [https://www.reddit.com/r/Python/comments/rmei4f/its_there_a_viable_alternative_to_pyaudio/](https://www.reddit.com/r/Python/comments/rmei4f/its_there_a_viable_alternative_to_pyaudio/)
6. What's a good sound recording library? : r/Python \- Reddit, дата последнего обращения: декабря 21, 2025, [https://www.reddit.com/r/Python/comments/3k11g5/whats_a_good_sound_recording_library/](https://www.reddit.com/r/Python/comments/3k11g5/whats_a_good_sound_recording_library/)
7. TOV: "Both PyAudio and SoundDevice a…" \- Fosstodon, дата последнего обращения: декабря 21, 2025, [https://fosstodon.org/@textovervideo/113765050321100138](https://fosstodon.org/@textovervideo/113765050321100138)
8. Play and Record Sound with Python — python-sounddevice, version 0.5.3, дата последнего обращения: декабря 21, 2025, [https://python-sounddevice.readthedocs.io/](https://python-sounddevice.readthedocs.io/)
9. Build A Voice Recorder Using Python \- Analytics Vidhya, дата последнего обращения: декабря 21, 2025, [https://www.analyticsvidhya.com/blog/2021/10/build-a-voice-recorder-using-python/](https://www.analyticsvidhya.com/blog/2021/10/build-a-voice-recorder-using-python/)
10. How can I do real-time voice activity detection in Python? \- Stack Overflow, дата последнего обращения: декабря 21, 2025, [https://stackoverflow.com/questions/60832201/how-can-i-do-real-time-voice-activity-detection-in-python](https://stackoverflow.com/questions/60832201/how-can-i-do-real-time-voice-activity-detection-in-python)
11. Speech-to-Text Latency: How to Read Vendor Claims and Minimize STT Delays, дата последнего обращения: декабря 21, 2025, [https://picovoice.ai/blog/speech-to-text-latency/](https://picovoice.ai/blog/speech-to-text-latency/)
12. Optimise OpenAI Whisper API: Audio Format, Sampling Rate and Quality \- DEV Community, дата последнего обращения: декабря 21, 2025, [https://dev.to/mxro/optimise-openai-whisper-api-audio-format-sampling-rate-and-quality-29fj](https://dev.to/mxro/optimise-openai-whisper-api-audio-format-sampling-rate-and-quality-29fj)
13. Presets — librosa 0.11.0 documentation, дата последнего обращения: декабря 21, 2025, [http://librosa.org/doc/0.11.0/auto_examples/plot_presets.html](http://librosa.org/doc/0.11.0/auto_examples/plot_presets.html)
14. librosa.load — librosa 0.11.0 documentation, дата последнего обращения: декабря 21, 2025, [https://librosa.org/doc/main/generated/librosa.load.html](https://librosa.org/doc/main/generated/librosa.load.html)
15. librosa.load — librosa 0.9.1 documentation, дата последнего обращения: декабря 21, 2025, [https://librosa.org/doc-playground/main/generated/librosa.load.html](https://librosa.org/doc-playground/main/generated/librosa.load.html)
16. Default sample rate for librosa.core.load is not the native sample rate \#509 \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/librosa/librosa/issues/509](https://github.com/librosa/librosa/issues/509)
17. Why resample on load? \- librosa blog, дата последнего обращения: декабря 21, 2025, [https://librosa.org/blog/2019/07/17/resample-on-load/](https://librosa.org/blog/2019/07/17/resample-on-load/)
18. Notes on continuous (ring) buffer reading \- Development \- VCV Community, дата последнего обращения: декабря 21, 2025, [https://community.vcvrack.com/t/notes-on-continuous-ring-buffer-reading/24264](https://community.vcvrack.com/t/notes-on-continuous-ring-buffer-reading/24264)
19. How to Implement High-Speed Voice Recognition in Chatbot Systems with WhisperX & Silero-VAD | by Aiden Koh | Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@aidenkoh/how-to-implement-high-speed-voice-recognition-in-chatbot-systems-with-whisperx-silero-vad-cdd45ea30904](https://medium.com/@aidenkoh/how-to-implement-high-speed-voice-recognition-in-chatbot-systems-with-whisperx-silero-vad-cdd45ea30904)
20. Silero VAD plugin \- LiveKit docs, дата последнего обращения: декабря 21, 2025, [https://docs.livekit.io/agents/logic-structure/turns/vad/](https://docs.livekit.io/agents/logic-structure/turns/vad/)
21. Quality benchmarks between audiotok / webrtcvad / silero-vad · Issue \#68 \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/wiseman/py-webrtcvad/issues/68](https://github.com/wiseman/py-webrtcvad/issues/68)
22. Silero VAD: pre-trained enterprise-grade Voice Activity Detector \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)
23. Voice Activity Detection (VAD): The Complete 2025 Guide to Speech Detection, дата последнего обращения: декабря 21, 2025, [https://picovoice.ai/blog/complete-guide-voice-activity-detection-vad/](https://picovoice.ai/blog/complete-guide-voice-activity-detection-vad/)
24. Choosing the Best Voice Activity Detection in 2025: Cobra vs Silero vs WebRTC VAD, дата последнего обращения: декабря 21, 2025, [https://picovoice.ai/blog/best-voice-activity-detection-vad-2025/](https://picovoice.ai/blog/best-voice-activity-detection-vad-2025/)
25. Voice activity detection (VAD) | OpenAI API, дата последнего обращения: декабря 21, 2025, [https://platform.openai.com/docs/guides/realtime-vad](https://platform.openai.com/docs/guides/realtime-vad)
26. I built a Local AI Voice Assistant with Ollama \+ gTTS with interruption \- Reddit, дата последнего обращения: декабря 21, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1k4b5xl/i_built_a_local_ai_voice_assistant_with_ollama/](https://www.reddit.com/r/LocalLLaMA/comments/1k4b5xl/i_built_a_local_ai_voice_assistant_with_ollama/)
27. Whisper \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/docs/transformers/en/model_doc/whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper)
28. openai/whisper-large-v3 \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
29. Whisper-Tiny \- Qualcomm AI Hub, дата последнего обращения: декабря 21, 2025, [https://aihub.qualcomm.com/mobile/models/whisper_tiny](https://aihub.qualcomm.com/mobile/models/whisper_tiny)
30. Faster Whisper transcription with CTranslate2 \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
31. 5 Ways to Speed Up Whisper Transcription \- Modal, дата последнего обращения: декабря 21, 2025, [https://modal.com/blog/faster-transcription](https://modal.com/blog/faster-transcription)
32. distilbert/distilgpt2 \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/distilbert/distilgpt2](https://huggingface.co/distilbert/distilgpt2)
33. microsoft/chatbench-distilgpt2 \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/microsoft/chatbench-distilgpt2](https://huggingface.co/microsoft/chatbench-distilgpt2)
34. valhalla/distilbart-mnli-12-3 \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/valhalla/distilbart-mnli-12-3](https://huggingface.co/valhalla/distilbart-mnli-12-3)
35. gTTS — gTTS documentation, дата последнего обращения: декабря 21, 2025, [https://gtts.readthedocs.io/](https://gtts.readthedocs.io/)
36. Google text-to-speech (gTTS): Transforming text into voice | Speechify, дата последнего обращения: декабря 21, 2025, [https://speechify.com/blog/gtts/](https://speechify.com/blog/gtts/)
37. silero-vad/examples/microphone_and_webRTC_integration/microphone_and_webRTC_integration.py at master · snakers4/silero-vad \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/snakers4/silero-vad/blob/master/examples/microphone_and_webRTC_integration/microphone_and_webRTC_integration.py](https://github.com/snakers4/silero-vad/blob/master/examples/microphone_and_webRTC_integration/microphone_and_webRTC_integration.py)
38. pyaudio-streaming-examples.ipynb \- snakers4/silero-vad \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb](https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb)
39. Asyncio Vs Threading In Python \- GeeksforGeeks, дата последнего обращения: декабря 21, 2025, [https://www.geeksforgeeks.org/python/asyncio-vs-threading-in-python/](https://www.geeksforgeeks.org/python/asyncio-vs-threading-in-python/)
40. Asyncio vs Threads in Python: The Ultimate Showdown for Interview Mastery, дата последнего обращения: декабря 21, 2025, [https://piyushsonawane.medium.com/asyncio-vs-threads-in-python-the-ultimate-showdown-for-interview-mastery-feffeba322e2](https://piyushsonawane.medium.com/asyncio-vs-threads-in-python-the-ultimate-showdown-for-interview-mastery-feffeba322e2)
41. python \- multiprocessing vs multithreading vs asyncio \- Stack Overflow, дата последнего обращения: декабря 21, 2025, [https://stackoverflow.com/questions/27435284/multiprocessing-vs-multithreading-vs-asyncio](https://stackoverflow.com/questions/27435284/multiprocessing-vs-multithreading-vs-asyncio)
42. Inference in PyTorch, what do the wrappers mean? What's best? \- Zach Mueller, дата последнего обращения: декабря 21, 2025, [https://muellerzr.github.io/blog/PyTorchInference.html](https://muellerzr.github.io/blog/PyTorchInference.html)
43. Efficient PyTorch: Tensor Memory Format Matters, дата последнего обращения: декабря 21, 2025, [https://pytorch.org/blog/tensor-memory-format-matters/](https://pytorch.org/blog/tensor-memory-format-matters/)
44. PyTorch \`torch.no_grad\` vs \`torch.inference_mode\` \- Stack Overflow, дата последнего обращения: декабря 21, 2025, [https://stackoverflow.com/questions/69543907/pytorch-torch-no-grad-vs-torch-inference-mode](https://stackoverflow.com/questions/69543907/pytorch-torch-no-grad-vs-torch-inference-mode)
45. Memory leaks at inference \- PyTorch Forums, дата последнего обращения: декабря 21, 2025, [https://discuss.pytorch.org/t/memory-leaks-at-inference/85108](https://discuss.pytorch.org/t/memory-leaks-at-inference/85108)
46. How to Evaluate Voice Assistant Pipelines From End to End | TELUS Digital, дата последнего обращения: декабря 21, 2025, [https://www.telusdigital.com/insights/data-and-ai/article/how-to-evaluate-voice-assistant-pipelines](https://www.telusdigital.com/insights/data-and-ai/article/how-to-evaluate-voice-assistant-pipelines)
47. How to Test Voice Recognition in 4 Steps With Perfecto, дата последнего обращения: декабря 21, 2025, [https://www.perfecto.io/blog/test-voice-recognition-perfecto](https://www.perfecto.io/blog/test-voice-recognition-perfecto)
48. Voice Application Testing: Tools and Frameworks for the Conversational Age \- Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@antonyberlin2003/voice-application-testing-tools-and-frameworks-for-the-conversational-age-06bdf97276c5](https://medium.com/@antonyberlin2003/voice-application-testing-tools-and-frameworks-for-the-conversational-age-06bdf97276c5)
49. Continuous Speech Recognition Testing \- Cyara, дата последнего обращения: декабря 21, 2025, [https://cyara.com/blog/continous-speech-recognition-testing/](https://cyara.com/blog/continous-speech-recognition-testing/)
50. Testing voice/chat agents for prompt injection attempts : r/AIToolTesting \- Reddit, дата последнего обращения: декабря 21, 2025, [https://www.reddit.com/r/AIToolTesting/comments/1np6wxf/testing_voicechat_agents_for_prompt_injection/](https://www.reddit.com/r/AIToolTesting/comments/1np6wxf/testing_voicechat_agents_for_prompt_injection/)
51. Voice Activity Detection \- A Lazy Data Science Guide \- Mohit Mayank, дата последнего обращения: декабря 21, 2025, [http://mohitmayank.com/a_lazy_data_science_guide/audio_intelligence/voice_activity_detection/](http://mohitmayank.com/a_lazy_data_science_guide/audio_intelligence/voice_activity_detection/)
52. Two important libraries used for audio processing and streaming in Python \- Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@venn5708/two-important-libraries-used-for-audio-processing-and-streaming-in-python-d3b718a75904](https://medium.com/@venn5708/two-important-libraries-used-for-audio-processing-and-streaming-in-python-d3b718a75904)
53. Faster Whisper Transcription: How to Maximize Performance for Real-Time Audio-to-Text, дата последнего обращения: декабря 21, 2025, [https://www.cerebrium.ai/articles/faster-whisper-transcription-how-to-maximize-performance-for-real-time-audio-to-text](https://www.cerebrium.ai/articles/faster-whisper-transcription-how-to-maximize-performance-for-real-time-audio-to-text)
