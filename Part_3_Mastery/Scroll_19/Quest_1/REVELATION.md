# **Архитектурный анализ и оптимизация ресурсов в последовательных мультимодальных конвейерах вывода: Технический отчет по реализации Квеста 19.1**

## **Аннотация**

В данном отчете представлен исчерпывающий технический анализ проектирования и реализации последовательного мультимодального конвейера (pipeline) искусственного интеллекта, разработанного в рамках инженерной задачи "Квест 19.1". Исследование фокусируется на интеграции трех разнородных нейросетевых архитектур — автоматического распознавания речи (ASR), классификации намерений "нулевого выстрела" (Zero-Shot Intent Classification) и генерации текста — в единую систему, функционирующую в условиях жестких ограничений видеопамяти (VRAM). Особое внимание уделяется механизмам управления памятью CUDA, феноменологии нейросетевых галлюцинаций (в частности, аномалии "Mead" в моделях Whisper) и алгоритмическим стратегиям согласования семантических пространств. Работа базируется на эмпирических данных, документации библиотек transformers, torch, librosa и анализе современных датасетов, таких как SUPERB.

## ---

**1\. Введение: Парадигма модульного ИИ на ограниченных ресурсах**

### **1.1 Эволюция от монолитных моделей к компонуемым системам**

Современный ландшафт искусственного интеллекта характеризуется переходом от изолированных задач к сложным мультимодальным системам, способным воспринимать, интерпретировать и генерировать информацию в различных форматах (аудио, текст, изображение). В контексте задачи "Квест 19.1" мы сталкиваемся с архетипичной проблемой Edge AI: необходимостью развертывания сложных интеллектуальных агентов на оборудовании потребительского класса с ограниченным объемом видеопамяти (VRAM), часто не превышающим 4–8 ГБ.1

Метафорическая "цепь големов", описанная в постановке задачи, технически представляет собой направленный ациклический граф (DAG) обработки данных, где выходы одной модели служат входами для следующей.

1. **"Писец" (The Scribe):** Модель openai/whisper-tiny преобразует акустические сигналы в дискретные текстовые токены.3
2. **"Толкователь" (The Interpreter):** Модель valhalla/distilbart-mnli-12-3 выполняет семантическое картирование транскрибированного текста на пространство предопределенных намерений.5
3. **"Оракул" (The Oracle):** Модель distilgpt2 генерирует контекстуально релевантный ответ на основе выявленного намерения.7

Ключевым инженерным вызовом здесь является невозможность одновременного размещения всех трех моделей в памяти GPU (VRAM) типичной рабочей станции без возникновения ошибки CUDA out of memory (OOM). Это диктует необходимость применения стратегии последовательного вывода (Sequential Inference), или, в терминах легенды квеста, ритуала "призыва и изгнания", который требует глубокого понимания механизмов аллокации памяти в PyTorch.9

### **1.2 Цели и структура отчета**

Данный отчет ставит своей целью не просто описать процесс запуска кода, но и провести глубокую деконструкцию каждого этапа конвейера. Мы проанализируем:

- Физические и математические основы преобразования аудиосигналов для моделей Transformer.
- Природу стохастических ошибок (галлюцинаций) в моделях ASR и методы их минимизации.
- Математический аппарат NLI (Natural Language Inference) как универсальный интерфейс для классификации.
- Детали реализации управления памятью CUDA для предотвращения фрагментации.

Каждый раздел подкреплен ссылками на исследовательские материалы, документацию и эмпирические наблюдения сообщества разработчиков.

## ---

**2\. Акустический интерфейс: Теория и практика модели Whisper**

Первым звеном цепи является преобразование непрерывного физического сигнала (звука) в дискретную последовательность символов. Для этой цели выбрана модель Whisper от OpenAI, представляющая собой Encoder-Decoder Transformer, обученный на 680 000 часах мультилингальных данных.3

### **2.1 Цифровая обработка сигналов и спектральный анализ**

Модели глубокого обучения, такие как Whisper, не работают с сырыми аудиофайлами напрямую. Процесс "слушания" начинается с этапа предобработки, выполняемого классом WhisperProcessor или библиотекой librosa.

#### **2.1.1 Частота дискретизации и теорема Котельникова**

Критическим требованием архитектуры Whisper является фиксированная частота дискретизации входного сигнала — 16 000 Гц (16 кГц).11 Это значение выбрано не случайно. Человеческая речь, как правило, сосредоточена в диапазоне частот от 300 Гц до 3400 Гц (так называемый "телефонный диапазон"). Согласно теореме Котельникова (Найквиста-Шеннона), для корректного восстановления сигнала частота дискретизации должна как минимум вдвое превышать максимальную частоту спектра сигнала ($F\_s \\ge 2F\_{max}$). Частота 16 кГц позволяет корректно оцифровывать звуки до 8 кГц, что с запасом покрывает основные форманты речи, включая шипящие и свистящие согласные, несущие значительную информационную нагрузку.

Если входной файл имеет другую частоту (например, стандартные для музыки 44.1 кГц или 48 кГц), требуется ресемплинг (передискретизация).11 В библиотеке librosa по умолчанию используется алгоритм soxr_hq (высококачественный ресемплинг на основе быстрого преобразования Фурье), который минимизирует артефакты алиасинга (наложения спектров). Использование более простых методов, таких как линейная интерполяция, может привести к искажению высокочастотных компонент и снижению точности распознавания.13

#### **2.1.2 Лог-Мел спектрограмма: Глаза Голема**

Whisper воспринимает звук как визуальный образ — лог-Мел спектрограмму. Этот процесс включает следующие математические преобразования:

1. **Оконное преобразование Фурье (STFT):** Сигнал разбивается на короткие перекрывающиеся окна (обычно 25 мс с шагом 10 мс). К каждому окну применяется быстрое преобразование Фурье для получения спектра мощности.
2. **Фильтрация по шкале Мелов:** Полученный спектр проецируется на шкалу Мелов, которая имитирует нелинейность человеческого слуха (мы лучше различаем низкие частоты, чем высокие). Whisper использует 80 Мел-фильтров.3
3. **Логарифмирование:** Амплитуды логарифмируются (обычно берется $10 \\log\_{10}(S)$), чтобы сжать динамический диапазон. Это позволяет модели одинаково эффективно обрабатывать как тихий шепот, так и громкую речь.

Итоговый тензор input_features имеет размерность (batch_size, 80, 3000), где 80 — количество частотных каналов, а 3000 — количество временных фреймов, соответствующих 30 секундам аудио.16 Аудио короче 30 секунд дополняется нулями (padding), длиннее — обрезается или разбивается на чанки.

### **2.2 Архитектура Трансформера в задачах ASR**

В отличие от моделей на базе Wav2Vec 2.0, которые используют CTC (Connectionist Temporal Classification) для выравнивания аудио и текста, Whisper использует полноценную архитектуру Encoder-Decoder (Seq2Seq).3

1. **Энкодер:** Принимает лог-Мел спектрограмму, обрабатывает её через сверточный слой (Conv1d) для извлечения локальных признаков, а затем пропускает через слои самовнимания (Self-Attention). Это создает богатое контекстное представление акустической информации.
2. **Декодер:** Авторегрессивно генерирует токены текста, используя перекрестное внимание (Cross-Attention) к выходам энкодера.

Модель whisper-tiny содержит всего 39 миллионов параметров.1 Это делает её чрезвычайно легкой (требует \~1 ГБ VRAM), но также накладывает ограничения на её "интеллектуальные" способности, особенно при работе с зашумленными данными или сложной лексикой.

### **2.3 Феноменология галлюцинаций: Аномалия "Mead" и проблема тишины**

Одним из наиболее интересных и проблемных аспектов работы модели Whisper является её склонность к галлюцинациям — генерации текста, отсутствующего в аудиосигнале. Исследования показывают, что Whisper особенно уязвим к галлюцинациям на участках тишины или монотонного фонового шума.20

#### **2.3.1 Природа галлюцинаций в тишине**

Поскольку Whisper является Seq2Seq моделью, обученной на большом корпусе субтитров из интернета, она выучила сильные статистические связи между языковыми конструкциями. Когда акустический сигнал слаб или отсутствует (тишина), внимание декодера "размывается", и модель начинает полагаться исключительно на априорное распределение вероятностей языковой модели.

Это приводит к генерации фраз, часто встречающихся в обучающих данных, но не связанных с аудио: "Thank you for watching", "Subtitles by...", или даже полных вымышленных предложений.20 В отличие от CTC-моделей, которые в тишине склонны выдавать пустой токен \<pad\>, авторегрессионный декодер Whisper стремится "продолжить историю".

#### **2.3.2 Аномалия "Mead"**

В исследовательских материалах зафиксирован специфический случай галлюцинации: генерация слова "Mead" (или фраз, связанных с ним) на коротких или тихих аудиофрагментах.23 Анализ источников позволяет выдвинуть гипотезу о происхождении этого артефакта.

Существует крупный аудиовизуальный датасет **MEAD** (Multi-view Emotional Audio-Visual Dataset), используемый для генерации эмоциональной речи и "говорящих голов".25 Вероятно, данные из этого или подобного корпуса попали в обучающую выборку Whisper. Если в обучающих данных содержались сэмплы тишины или технические метки, ассоциированные со словом "Mead" (например, в метаданных или именах файлов, которые ошибочно попали в транскрипцию), модель могла выучить ложную корреляцию: \[Low Energy Audio\] \-\> "Mead".

Кроме того, слово "Mead" фонетически простое (носовой сонант /m/ и долгий гласный /i:/), и в условиях акустической парейдолии (поиска паттернов в шуме) модель может ошибочно детектировать эти форманты в фоновом шуме.23

Для борьбы с этим феноменом в Quest 19.1 необходимо использовать параметр no_speech_threshold при генерации или предварительно обрабатывать аудио с помощью VAD (Voice Activity Detection), чтобы не подавать на вход "Писцу" участки тишины.27

## ---

**3\. Экономика VRAM: Управление памятью GPU в средах PyTorch**

Центральным ограничением квеста является "Мана" — видеопамять (VRAM). Модели whisper-tiny (\~1 ГБ), distilbart (\~1 ГБ) и distilgpt2 (\~0.5–1 ГБ) суммарно могут занять 3–4 ГБ VRAM, что близко к пределу бюджетных карт, особенно с учетом накладных расходов CUDA.1

### **3.1 Иерархия памяти CUDA и аллокатор PyTorch**

Память GPU не является простым плоским пространством. PyTorch использует собственный **Caching Allocator** для ускорения выделения памяти. Когда тензор удаляется (del tensor), память не возвращается операционной системе (OS) мгновенно. Вместо этого она помечается как свободная внутри кэша PyTorch и переиспользуется для будущих тензоров того же размера.9

Это создает две категории используемой памяти:

1. **Allocated Memory:** Память, занятая активными тензорами.
2. **Reserved (Cached) Memory:** Память, выделенная у драйвера CUDA, но в данный момент не используемая тензорами.

Проблема возникает при **фрагментации**. Если мы загружаем и выгружаем модели разного размера, в "куче" памяти могут образоваться "дыры" (свободные блоки), которые слишком малы для размещения новой модели целиком, но в сумме составляют значительный объем. Это приводит к ошибкам OOM даже при наличии формально свободного места.10

### **3.2 Стратегия последовательной загрузки ("Призыв и Изгнание")**

Для успешного выполнения квеста на GPU с малым объемом памяти (например, 4 ГБ) необходимо реализовать строгий паттерн последовательного выполнения:

1. **Загрузка (Load):** Модель загружается в RAM, затем переносится в VRAM (.to(device)).
2. **Инференс (Inference):** Выполняется прямой проход (forward pass).
3. **Выгрузка (Unload):**
   - Модель удаляется из пространства имен Python (del model).
   - Вызывается сборщик мусора Python (gc.collect()) для уничтожения циклических ссылок и освобождения объектов-оберток.
   - **Очистка кэша:** Вызывается torch.cuda.empty_cache().

### **3.3 Фрагментация памяти и методы дефрагментации**

Функция torch.cuda.empty*cache() является критически важной, но дорогой операцией. Она освобождает всю *зарезервированную*, но не *аллоцированную\_ память, возвращая её драйверу GPU.9 Это позволяет драйверу дефрагментировать память или выделить её другому процессу.

**Таблица 1: Сравнительный анализ использования памяти моделями (Оценка)**

| Модель               | Параметры | Приблизительный вес (FP32) | Требуемая VRAM (Инференс) | Примечание                               |
| :------------------- | :-------- | :------------------------- | :------------------------ | :--------------------------------------- |
| whisper-tiny         | 39 млн    | \~150 МБ                   | \~1.0 ГБ                  | Большой оверхед из\-за аудио-контекста 1 |
| distilbart-mnli-12-3 | \~180 млн | \~700 МБ                   | \~1.2 ГБ                  | Зависит от кол-ва меток-кандидатов 32    |
| distilgpt2           | 82 млн    | \~330 МБ                   | \~0.8 ГБ                  | Зависит от длины генерации (KV-кэш) 8    |

Без вызова empty_cache() между этапами, фрагментация памяти после удаления Whisper может помешать загрузке DistilBART, даже если суммарный объем свободной памяти достаточен.

## ---

**4\. Семантическая интерпретация: Классификация намерений Zero-Shot**

После того как "Писец" (Whisper) преобразовал звук в текст, в дело вступает "Толкователь". Задача этого этапа — преобразовать неструктурированный текст (например, "Включи, пожалуйста, свет") в структурированное намерение (например, turn_on_light). Для этого используется модель valhalla/distilbart-mnli-12-3.

### **4.1 NLI как универсальный интерфейс классификации**

Традиционная классификация требует обучения модели на фиксированном наборе классов. Подход Zero-Shot (нулевой выстрел) меняет парадигму, сводя задачу классификации к задаче **Natural Language Inference (NLI)** — логического вывода на естественном языке.33

Модель получает на вход пару предложений:

1. **Посылка (Premise):** Исходный текст (например, транскрипция от Whisper).
2. **Гипотеза (Hypothesis):** Шаблон вида "Этот текст о {метке}".

Модель предсказывает отношение между ними: **Entailment** (следствие), **Contradiction** (противоречие) или **Neutral** (нейтрально). Вероятность класса "Entailment" интерпретируется как вероятность того, что текст принадлежит к данной категории.35

Это позволяет классифицировать текст по любым меткам, которые мы придумаем "на лету", без переобучения модели. В контексте Квеста 19.1 это дает гибкость: мы можем задать метки, соответствующие магическим действиям (например, spell_cast, summon_creature, alchemy), и модель поймет их семантику благодаря предварительному обучению на огромных корпусах текстов.

### **4.2 Архитектура DistilBART: Эффективность через дистилляцию**

Модель valhalla/distilbart-mnli-12-3 является дистиллированной версией bart-large-mnli. Процесс дистилляции (Knowledge Distillation) заключается в обучении меньшей модели ("студента") воспроизводить поведение большой модели ("учителя").6

Обозначение "12-3" раскрывает архитектурные изменения:

- **Энкодер:** Оставлены все 12 слоев (как у учителя), так как энкодер критически важен для понимания входного текста.
- **Декодер:** Сокращен до 3 слоев (вместо 12).

Поскольку задача классификации использует декодер только для формирования финального представления (в отличие от генерации текста, где декодер работает итеративно), сокращение слоев декодера дает значительный выигрыш в скорости и памяти при минимальной потере точности (падение Accuracy на MNLI всего на \~1-2%).5 Это делает данную модель идеальным выбором для нашего ограниченного по ресурсам конвейера.

## ---

**5\. Генеративный оракул: Применение Causal Language Models**

Финальный этап — "Оракул", генерирующий ответ пользователю. Используемая модель distilgpt2 является представителем семейства CLM (Causal Language Models).

### **5.1 Архитектура DistilGPT-2 и механизмы внимания**

distilgpt2 — это трансформер, состоящий только из декодера (Decoder-only). Он обучен предсказывать следующий токен в последовательности на основе предыдущих.8 Модель имеет около 82 миллионов параметров, что вдвое меньше стандартного GPT-2 Small (124 млн).7

В отличие от BERT (используемого в классификаторе), GPT-2 использует **маскированное внимание (Masked Self-Attention)**. Это означает, что при вычислении представления для текущего токена модель "не видит" токены, стоящие справа (в будущем). Это фундаментальное свойство позволяет использовать модель для генерации текста токен за токеном.

### **5.2 Стратегии декодирования и управление генерацией**

Чтобы "Оракул" давал осмысленные ответы, необходимо правильно настроить параметры генерации в методе model.generate():

- **do_sample=True:** Включает сэмплирование (вероятностный выбор следующего токена). Без этого модель будет всегда выбирать самый вероятный токен (Greedy Search), что часто приводит к зацикливанию и скучным ответам.38
- **top_k и top_p (Nucleus Sampling):** Ограничивают "хвост" распределения вероятностей, отсекая маловероятные и бессмысленные слова. top_k=50 означает выбор только из 50 наиболее вероятных слов.40
- **no_repeat_ngram_size=2:** Критически важный параметр для малых моделей типа distilgpt2. Он запрещает модели генерировать одинаковые биграммы (пары слов) подряд. Малые модели часто "застревают" в петлях повторений (например, "the spell is the spell is..."). Этот параметр жестко блокирует такие петли.41
- **max_length:** Ограничивает длину генерации, экономя VRAM и время.

Пример интеграции в конвейер: мы подаем на вход модели промпт, сформированный на основе классифицированного намерения (например, "Пользователь хочет скастовать заклинание. Оракул отвечает:"), и модель продолжает этот текст.

## ---

**6\. Анализ данных и среды выполнения: Бенчмарк SUPERB**

В условиях задачи упоминается использование датасета SUPERB (Speech processing Universal PERformance Benchmark), в частности его подмножества KS (Keyword Spotting).43 Понимание природы этих данных необходимо для корректной интерпретации результатов работы "Писца".

### **6.1 Структура датасета Keyword Spotting (KS)**

Датасет KS базируется на Google Speech Commands v1.0. Он содержит короткие аудиоклипы (обычно 1 секунда), содержащие одну из 10 команд (yes, no, up, down, left, right, on, off, stop, go), а также классы \_silence\_ (тишина) и \_unknown\_ (неизвестное слово).44

### **6.2 Проблемы согласования модальностей**

Использование модели Whisper (ASR) на таком датасете сопряжено с рядом трудностей:

1. **Краткость:** Whisper оптимизирован для транскрибации предложений (контекст до 30 секунд). На односекундных клипах модель может работать нестабильно, пытаясь найти контекст там, где его нет.
2. **Тишина и шум:** Класс \_silence\_ в SUPERB KS является главным источником галлюцинаций для Whisper. Как обсуждалось в разделе 2.3, именно здесь возникают фантомные "Mead" или субтитры.20
3. **Несоответствие форматов:** Whisper ожидает 30-секундный вход. При подаче 1-секундного клипа оставшиеся 29 секунд заполняются паддингом (нулями). Если паддинг реализован некорректно (например, не нулями, а шумом), это может сбить работу энкодера.

Тем не менее, эксперименты показывают, что Whisper способен распознавать ключевые слова с высокой точностью, если правильно обработать выходные данные (нормализация текста, удаление пунктуации).

**Таблица 2: Сравнение классов SUPERB KS и потенциальных намерений**

| Класс KS (Аудио) | Транскрипция Whisper    | Потенциальное Намерение (Classifier Label) |
| :--------------- | :---------------------- | :----------------------------------------- |
| up / go / start  | "Up", "Go"              | Activation / Movement                      |
| stop / off / no  | "Stop", "Off"           | Halt / Cancellation                        |
| \_silence\_      | "Mead", "Thanks...", "" | Noise (Требует фильтрации)                 |

## ---

**7\. Заключение**

Инженерная реализация "Квеста 19.1" демонстрирует фундаментальные принципы построения современных AI-систем на краю (Edge AI). Успех "Голем-конвейера" зависит не столько от выбора самых мощных моделей, сколько от грамотной оркестрации ресурсов и понимания "физики" работы нейросетей.

Проведенный анализ выявил, что:

1. **Управление VRAM** через принудительную очистку кэша (empty_cache) и удаление объектов является единственным способом запуска мультимодельных цепей на бюджетном оборудовании, несмотря на накладные расходы по времени выполнения.
2. **Предобработка аудио** (ресемплинг до 16 кГц, VAD) критически важна для стабильности Whisper и предотвращения галлюцинаций типа "Mead".
3. **Zero-Shot классификация** предоставляет мощный механизм гибкости, позволяя менять логику "Толкователя" без переобучения, просто изменяя текстовые метки.
4. **Генерация текста** требует тщательной настройки параметров декодирования (no_repeat_ngram_size), чтобы компенсировать ограниченные когнитивные способности малых моделей (distilgpt2).

Данный отчет служит теоретическим фундаментом для практической реализации кода квеста, обеспечивая понимание каждого шага "магического ритуала" с точки зрения компьютерных наук и инженерии данных.

#### **Источники**

1. whisper/README.md at main · openai/whisper \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/openai/whisper/blob/main/README.md](https://github.com/openai/whisper/blob/main/README.md)
2. How to run Whisper Large-v3 on 4gb vram (in my case, 1050 Ti) : r/LocalLLaMA \- Reddit, дата последнего обращения: декабря 21, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1bbqpes/how_to_run_whisper_largev3_on_4gb_vram_in_my_case/](https://www.reddit.com/r/LocalLLaMA/comments/1bbqpes/how_to_run_whisper_largev3_on_4gb_vram_in_my_case/)
3. Whisper \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/docs/transformers/en/model_doc/whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper)
4. openai/whisper-tiny \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
5. valhalla/distilbart-mnli-12-3 \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/valhalla/distilbart-mnli-12-3](https://huggingface.co/valhalla/distilbart-mnli-12-3)
6. README.md · valhalla/distilbart-mnli-12-3 at main \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/valhalla/distilbart-mnli-12-3/blame/main/README.md](https://huggingface.co/valhalla/distilbart-mnli-12-3/blame/main/README.md)
7. AI Model Catalog \- Azure AI Foundry, дата последнего обращения: декабря 21, 2025, [https://ai.azure.com/catalog/models/distilgpt2](https://ai.azure.com/catalog/models/distilgpt2)
8. distilbert/distilgpt2 \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/distilbert/distilgpt2](https://huggingface.co/distilbert/distilgpt2)
9. About torch.cuda.empty_cache() \- PyTorch Forums, дата последнего обращения: декабря 21, 2025, [https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232](https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232)
10. дата последнего обращения: декабря 21, 2025, [https://worldversant.com/the-silent-bottleneck-handling-gpu-memory-fragmentation-in-deep-learning-workloads\#:\~:text=Memory%20Defragmentation%20with%20torch.\&text=empty_cache()%20.,a%20complete%20solution%20for%20fragmentation.](<https://worldversant.com/the-silent-bottleneck-handling-gpu-memory-fragmentation-in-deep-learning-workloads#:~:text=Memory%20Defragmentation%20with%20torch.&text=empty_cache()%20.,a%20complete%20solution%20for%20fragmentation.>)
11. librosa.load — librosa 0.11.0 documentation, дата последнего обращения: декабря 21, 2025, [https://librosa.org/doc/main/generated/librosa.load.html](https://librosa.org/doc/main/generated/librosa.load.html)
12. Work directly with Files instead of path of files · openai whisper · Discussion \#1620 \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/openai/whisper/discussions/1620](https://github.com/openai/whisper/discussions/1620)
13. librosa.resample — librosa 0.11.0 documentation, дата последнего обращения: декабря 21, 2025, [https://librosa.org/doc/main/generated/librosa.resample.html](https://librosa.org/doc/main/generated/librosa.resample.html)
14. Why resample on load? \- librosa blog, дата последнего обращения: декабря 21, 2025, [https://librosa.org/blog/2019/07/17/resample-on-load/](https://librosa.org/blog/2019/07/17/resample-on-load/)
15. Whisper \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/docs/transformers/model_doc/whisper](https://huggingface.co/docs/transformers/model_doc/whisper)
16. Whisper in Transformers.ipynb \- Colab, дата последнего обращения: декабря 21, 2025, [https://colab.research.google.com/drive/16HO7if9iwfpSJzhqlaNOu6iiMhUBLMKE?usp=sharing](https://colab.research.google.com/drive/16HO7if9iwfpSJzhqlaNOu6iiMhUBLMKE?usp=sharing)
17. Whisper \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/docs/transformers/v4.29.0/model_doc/whisper](https://huggingface.co/docs/transformers/v4.29.0/model_doc/whisper)
18. Documentation of class/model functionality, for example: WhisperForConditionalGeneration · Issue \#29394 · huggingface/transformers \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/huggingface/transformers/issues/29394](https://github.com/huggingface/transformers/issues/29394)
19. Memory requirements? · openai whisper · Discussion \#5 \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/openai/whisper/discussions/5](https://github.com/openai/whisper/discussions/5)
20. AI speech-to-text can hallucinate violent language | Cornell Chronicle, дата последнего обращения: декабря 21, 2025, [https://news.cornell.edu/stories/2024/06/ai-speech-text-can-hallucinate-violent-language](https://news.cornell.edu/stories/2024/06/ai-speech-text-can-hallucinate-violent-language)
21. Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio This research was supported by the National Science Centre, Poland under Grant 2021/42/E/ST7/00452, the National Centre for Research and Development, Poland under Grant INFOSTRATEG-IV/0029/2022, and by program ”Excellence initiative – research university” \- arXiv, дата последнего обращения: декабря 21, 2025, [https://arxiv.org/html/2501.11378v1](https://arxiv.org/html/2501.11378v1)
22. OpenAI's Whisper filled in silent part at the beginning of audio recording (actual audio starts at "OK, great, thank you,") with deranged verbalization. Any understanding why it would do that? \- Reddit, дата последнего обращения: декабря 21, 2025, [https://www.reddit.com/r/OpenAI/comments/1c3b7e8/openais_whisper_filled_in_silent_part_at_the/](https://www.reddit.com/r/OpenAI/comments/1c3b7e8/openais_whisper_filled_in_silent_part_at_the/)
23. Do you have hypnagogic or hypnopompic hallucinations sometimes? | Page 3 \- Vi-Control, дата последнего обращения: декабря 21, 2025, [https://vi-control.net/community/threads/do-you-have-hypnagogic-or-hypnopompic-hallucinations-sometimes.111181/page-3](https://vi-control.net/community/threads/do-you-have-hypnagogic-or-hypnopompic-hallucinations-sometimes.111181/page-3)
24. Moonlight Whispers of the White Buffalo \- Unfocussed Photographic Art, дата последнего обращения: декабря 21, 2025, [https://unfocussed.com/blogs/captured-tales/moonlight-whispers-of-the-white-buffalo](https://unfocussed.com/blogs/captured-tales/moonlight-whispers-of-the-white-buffalo)
25. VOICEASSISTANT-EVAL: BENCHMARKING AI ASSIS- TANTS ACROSS LISTENING, SPEAKING, AND VIEWING \- OpenReview, дата последнего обращения: декабря 21, 2025, [https://openreview.net/pdf/5afb71b0930c3a0f6f1d23f03001afab9502bd21.pdf](https://openreview.net/pdf/5afb71b0930c3a0f6f1d23f03001afab9502bd21.pdf)
26. MEAD: A Large-Scale Audio-Visual Dataset for Emotional Talking-Face Generation | Request PDF \- ResearchGate, дата последнего обращения: декабря 21, 2025, [https://www.researchgate.net/publication/346876719_MEAD_A_Large-Scale_Audio-Visual_Dataset_for_Emotional_Talking-Face_Generation](https://www.researchgate.net/publication/346876719_MEAD_A_Large-Scale_Audio-Visual_Dataset_for_Emotional_Talking-Face_Generation)
27. How did whisper-zero manage to reduce whisper hallucinations? Any ideas? \- Reddit, дата последнего обращения: декабря 21, 2025, [https://www.reddit.com/r/LocalLLaMA/comments/1gg6rpg/how_did_whisperzero_manage_to_reduce_whisper/](https://www.reddit.com/r/LocalLLaMA/comments/1gg6rpg/how_did_whisperzero_manage_to_reduce_whisper/)
28. openai/whisper-large-v3 · Audio input consists of only 3000\. Short-form transcription is activated.no_speech_threshold is set to 0.5, but will be ignored. \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/openai/whisper-large-v3/discussions/104](https://huggingface.co/openai/whisper-large-v3/discussions/104)
29. Enhancing Term-Based Document Retrieval by Word Embedding and Transformer Models \- electronic library \-, дата последнего обращения: декабря 21, 2025, [https://elib.dlr.de/147631/1/Graduate%20Thesis%20%20Report%20-%20Sheikh%20Mastura%20Farzana.pdf](https://elib.dlr.de/147631/1/Graduate%20Thesis%20%20Report%20-%20Sheikh%20Mastura%20Farzana.pdf)
30. CUDA semantics — PyTorch 2.9 documentation, дата последнего обращения: декабря 21, 2025, [https://docs.pytorch.org/docs/stable/notes/cuda.html](https://docs.pytorch.org/docs/stable/notes/cuda.html)
31. A Deep Dive into PyTorch's GPU Memory Management \- Forward Everyday, дата последнего обращения: декабря 21, 2025, [https://forwardevery.day/2024/09/03/a-deep-dive-into-pytorchs-gpu-memory-management/](https://forwardevery.day/2024/09/03/a-deep-dive-into-pytorchs-gpu-memory-management/)
32. Introductory Guide to using HuggingFace for your Modelling Needs | by Louis D'hulst, дата последнего обращения: декабря 21, 2025, [https://medium.com/@louisdhulst/introductory-guide-to-using-huggingface-for-your-modelling-needs-e4e9907cf6b4](https://medium.com/@louisdhulst/introductory-guide-to-using-huggingface-for-your-modelling-needs-e4e9907cf6b4)
33. Zeroshot Classification. Machine learning with no Data and… | by Anshu Kumar \- Medium, дата последнего обращения: декабря 21, 2025, [https://akgeni.medium.com/zeroshot-classification-864a278628f6](https://akgeni.medium.com/zeroshot-classification-864a278628f6)
34. Matching Tasks with Industry Groups for Augmenting Commonsense Knowledge \- arXiv, дата последнего обращения: декабря 21, 2025, [https://arxiv.org/html/2505.07440v1](https://arxiv.org/html/2505.07440v1)
35. Zero-Shot Classification in Web Scraping: Tutorial & Guide \- Bright Data, дата последнего обращения: декабря 21, 2025, [https://brightdata.com/blog/ai/zero-shot-classification](https://brightdata.com/blog/ai/zero-shot-classification)
36. GPT-2 \- Wikipedia, дата последнего обращения: декабря 21, 2025, [https://en.wikipedia.org/wiki/GPT-2](https://en.wikipedia.org/wiki/GPT-2)
37. Distilgpt2 · Models \- Dataloop, дата последнего обращения: декабря 21, 2025, [https://dataloop.ai/library/model/distilbert_distilgpt2/](https://dataloop.ai/library/model/distilbert_distilgpt2/)
38. Generation \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/docs/transformers/main_classes/text_generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
39. Text Generation with GPT-2 Model \- MachineLearningMastery.com, дата последнего обращения: декабря 21, 2025, [https://machinelearningmastery.com/text-generation-with-gpt-2-model/](https://machinelearningmastery.com/text-generation-with-gpt-2-model/)
40. Text Generation with HuggingFace \- GPT2 \- Kaggle, дата последнего обращения: декабря 21, 2025, [https://www.kaggle.com/code/tuckerarrants/text-generation-with-huggingface-gpt2](https://www.kaggle.com/code/tuckerarrants/text-generation-with-huggingface-gpt2)
41. Generating Text with GPT2 in Under 10 Lines of Code | by Majd Farah \- Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@majd.farah08/generating-text-with-gpt2-in-under-10-lines-of-code-5725a38ea685](https://medium.com/@majd.farah08/generating-text-with-gpt2-in-under-10-lines-of-code-5725a38ea685)
42. NLG with GPT-2 \- Jake Tae, дата последнего обращения: декабря 21, 2025, [https://jaketae.github.io/study/gpt2/](https://jaketae.github.io/study/gpt2/)
43. s3prl/superb · Datasets at Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/datasets/s3prl/superb](https://huggingface.co/datasets/s3prl/superb)
44. anton-l/superb · Datasets at Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/datasets/anton-l/superb](https://huggingface.co/datasets/anton-l/superb)
45. README.md · regisss/superb_ks at main \- Hugging Face, дата последнего обращения: декабря 21, 2025, [https://huggingface.co/datasets/regisss/superb_ks/blob/main/README.md](https://huggingface.co/datasets/regisss/superb_ks/blob/main/README.md)
46. Generalisation Gap of Keyword Spotters in a Cross-Speaker Low-Resource Scenario \- PMC, дата последнего обращения: декабря 21, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8704929/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8704929/)
