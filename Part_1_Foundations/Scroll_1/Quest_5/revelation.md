# **Демистификация черного ящика: Глубокий инженерный анализ архитектуры ручного инференса модели DistilBERT**

## **Введение: От абстракции к контролю**

В современной экосистеме обработки естественного языка (NLP) наблюдается тенденция к демократизации доступа к сложным нейросетевым архитектурам. Высокоуровневые библиотеки, такие как Hugging Face Transformers, предоставляют абстракцию pipeline, которая часто воспринимается разработчиками как «универсальный амулет», способный решить любую задачу классификации или генерации одной строкой кода. Однако, по мере роста требований к производительности производственных систем, эта абстракция превращается из ускорителя разработки в техническое ограничение. «Квест 1.5: Заглянуть в разум Голема» символизирует критический этап в созревании инженера машинного обучения: переход от потребления готовых API к прямому управлению механикой инференса.

В данном отчете представлен исчерпывающий технический анализ жизненного цикла ручного инференса на примере модели distilbert-base-uncased-finetuned-sst-2-english. Мы деконструируем процесс предсказания на атомарные составляющие: токенизацию (пре-процессинг), прямой проход через вычислительный граф (инференс) и интерпретацию логитов (пост-процессинг). Особое внимание уделяется математической природе «сырых» выходов нейросети, оптимизации работы с памятью через контекстные менеджеры PyTorch и архитектурным отличиям дистиллированных моделей. Этот анализ необходим для инженеров, стремящихся оптимизировать латентность, внедрять кастомную логику батчинга и эффективно развертывать модели в условиях ограниченных ресурсов.

## **Часть I. Анатомия инференса: Пределы абстракции Pipeline**

Стандартный конвейер инференса часто ошибочно воспринимается как монолитная операция, магическим образом преобразующая текст в семантическую метку. В действительности же это строго детерминированная последовательность тензорных операций и трансформаций данных. Понимание этой последовательности является предпосылкой для перехода от "пользователя" к "архитектору" системы.

### **1.1 Технические ограничения абстракции Pipeline**

Функция pipeline спроектирована как фасад, скрывающий сложность загрузки весов, токенизации и пост-обработки. Хотя это эффективно для быстрого прототипирования, в высоконагруженных системах такой подход вносит существенные накладные расходы. Обобщенная логика пайплайна инициализирует пути исполнения, которые могут быть не оптимизированы для конкретного оборудования или размера батча, выполняя избыточные проверки типов и копирование данных, которых можно избежать в ручном цикле.1 Более того, в сложных архитектурах, таких как векторные базы данных, этапы токенизации и инференса часто разнесены во времени и пространстве (например, токенизация происходит на этапе индексации, а инференс — при запросе), что делает монолитный pipeline неприменимым. Наконец, абстракция скрывает точную локализацию ошибок: при возникновении сбоя разработчику сложно определить, произошла ли ошибка на этапе кодирования спецсимволов, аллокации GPU-памяти или маппинга меток.3

### **1.2 Трехступенчатая архитектура ручного управления**

Ручной инференс требует явного управления тремя отдельными этапами, что превращает инженера из пассивного наблюдателя в активного оператора потока данных. Этот процесс можно представить в виде следующей таблицы, описывающей трансформацию данных на каждом этапе:

| Этап                    | Компонент ("Сущность")   | Входные данные       | Выходные данные                          | Основная операция                             |
| :---------------------- | :----------------------- | :------------------- | :--------------------------------------- | :-------------------------------------------- |
| **1\. Пре-процессинг**  | AutoTokenizer ("Толмач") | Сырой текст (String) | Тензоры (input_ids, attention_mask)      | Кодирование WordPiece, добавление спецтокенов |
| **2\. Инференс**        | AutoModel ("Голем")      | Тензоры (PyTorch/TF) | Объект SequenceClassifierOutput (Логиты) | Матричное умножение, Attention, Feed-Forward  |
| **3\. Пост-процессинг** | Логика Python            | Логиты (Float)       | Метка класса (String)                    | Argmax / Softmax, маппинг через id2label      |

## **Часть II. Механика пре-процессинга: Двигатель токенизации**

Первым актом ритуала является призыв "Толмача", технически именуемого токенизатором. В контексте трансформерных моделей токенизация — это не просто разбиение строки по пробелам, а сложный алгоритм сжатия информации, балансирующий между размером словаря и семантической выразительностью.

### **2.1 Алгоритм WordPiece и эффективность словаря**

Модель distilbert-base-uncased использует алгоритм токенизации WordPiece, унаследованный от архитектуры BERT. В отличие от традиционной токенизации по пробелам, которая приводит к созданию огромных разреженных словарей, WordPiece разбивает слова на подслова (subwords). Это позволяет модели оперировать фиксированным словарем (обычно 30 522 токена для BERT-семейства), представляя теоретически бесконечное количество слов. Распространенные слова, такие как "token", остаются едиными единицами, в то время как редкие или морфологически сложные слова разбиваются на составные части (например, "tokenization" может стать \['token', '\#\#iza', '\#\#tion'\]). Префикс \#\# указывает на то, что подслово является продолжением предыдущего токена.4

Спецификация uncased в названии модели указывает на то, что весь текст приводится к нижнему регистру перед токенизацией. Это снижает размерность задачи (модели не нужно учить, что "Apple" и "apple" семантически близки), но жертвует смысловыми нюансами, где капитализация имеет значение (например, "Bush" как фамилия президента и "bush" как растение).4

### **2.2 Директива return_tensors='pt' и управление памятью**

В скрипте ручного инференса токенизатор вызывается со специфическим аргументом, который диктует формат выходных данных:

Python

runes \= translator(my_phrase, return_tensors="pt")

Аргумент return_tensors="pt" является критически важной директивой. По умолчанию токенизаторы Hugging Face возвращают стандартные списки Python (Lists of Lists). Хотя списки гибки, они крайне неэффективны для масштабных матричных операций, требуемых нейросетью, так как не гарантируют непрерывного размещения данных в памяти. Флаг pt приказывает токенизатору немедленно конвертировать результаты в тензоры PyTorch (torch.Tensor), размещая их в непрерывном блоке памяти, оптимизированном для процессорных инструкций (CPU) или вычислений на графическом ускорителе (GPU).6

Отсутствие этого флага является распространенным источником ошибок типа AttributeError или TypeError при передаче данных в модель, так как AutoModel ожидает на входе объекты, поддерживающие тензорные операции (например, .size(), .device), которых нет у обычных списков.7 Кроме того, использование 'pt' автоматически добавляет размерность батча (batch dimension), превращая вектор длины $L$ в матрицу размера $1 \\times L$. Это критически важно, так как архитектура трансформеров ожидает на входе именно пакет данных, даже если этот пакет состоит из одного предложения.8

### **2.3 Структура выходных данных токенизатора**

Переменная runes, являющаяся результатом работы токенизатора, представляет собой объект BatchEncoding (наследуемый от словаря), содержащий два ключевых тензора:

#### **2.3.1 input_ids: Числовые руны**

Этот тензор содержит целые числа, отображающие каждое подслово на его индекс в словаре модели. Важной деталью является автоматическое добавление специальных токенов, которые играют структурную роль в архитектуре BERT:

- \*\* (ID 101):\*\* Добавляется в начало каждой последовательности. В задачах классификации (Sequence Classification) вектор скрытого состояния, соответствующий этому токену на выходе последнего слоя, используется как агрегированное представление всего предложения. Именно этот вектор подается на полносвязный слой классификатора.8
- \*\* (ID 102):\*\* Добавляется в конец последовательности, обозначая границу предложения.

#### **2.3.2 attention_mask: Механизм фокусировки**

Этот бинарный тензор служит инструкцией для механизма внимания (Self-Attention) внутри модели. В случае обработки пакета предложений разной длины, более короткие предложения дополняются (padding) специальным токеном \`\` (обычно ID 0\) до длины самого длинного предложения в батче. Маска внимания присваивает значение 1 реальным токенам и 0 токенам-заполнителям.

Механизм работы маски заключается в модификации матрицы внимания перед применением функции Softmax. Значения (dot products), соответствующие замаскированным позициям (0), заменяются на минус бесконечность ($-\\infty$). При применении Softmax ($e^{-\\infty} \\approx 0$) вес внимания для этих токенов становится нулевым, что предотвращает влияние "мусорных" токенов-заполнителей на формирование семантического представления реальных слов.10 Ошибки в формировании маски внимания при ручном батчинге часто приводят к неверным предсказаниям, так как модель начинает "учитывать пустоту".12

## **Часть III. Архитектура инференса: Внутри DistilBERT**

После подготовки данных ("пищи") процесс переходит к этапу "кормления" — прямому проходу (forward pass) через AutoModelForSequenceClassification. Именно здесь происходят основные вычислительные затраты.

### **3.1 Физика дистилляции: Ученик и Учитель**

Используемая модель distilbert-base-uncased является результатом процесса "knowledge distillation" (дистилляции знаний), где компактная модель-"студент" (DistilBERT) обучается воспроизводить поведение массивной модели-"учителя" (BERT).4 Архитектурно DistilBERT уменьшает количество слоев трансформера в два раза (с 12 до 6), удаляет эмбеддинги типов токенов (token_type_ids) и слой пулинга, сохраняя при этом остальную архитектуру идентичной.

Процесс обучения студента управляется тройной функцией потерь (triple loss objective), что обеспечивает высокую точность при сниженных ресурсах:

1. **Distillation Loss (Потеря дистилляции):** Модель обучается минимизировать дивергенцию Кульбака-Лейблера между распределением вероятностей, выдаваемым учителем, и собственным распределением. Это заставляет студента выучивать не просто "правильный ответ", но и "структуру неопределенности" учителя (dark knowledge).4
2. **Masked Language Modeling (MLM):** Классическая задача BERT по восстановлению скрытых токенов, обеспечивающая понимание языкового контекста.
3. **Cosine Embedding Loss:** Модель обучается генерировать векторы скрытых состояний, косинусно близкие к векторам учителя. Это выравнивает внутренние представления моделей в векторном пространстве.

Результатом является модель, которая на 40% меньше по размеру и на 60% быстрее в инференсе, чем BERT, при сохранении 97% производительности на задачах типа SST-2.4 Слой классификации, добавляемый классом AutoModelForSequenceClassification, представляет собой линейную проекцию вектора \`\` на пространство классов (размерность 2: Positive/Negative).13

### **3.2 Управление вычислительным графом: torch.no_grad() vs torch.inference_mode()**

В скрипте используется контекстный менеджер with torch.no_grad():. Это не просто синтаксический сахар, а фундаментальный механизм управления памятью и вычислительным графом PyTorch.

По умолчанию PyTorch строит динамический вычислительный граф (DAG) для всех операций с тензорами, у которых атрибут requires_grad=True. Этот граф хранит историю операций и промежуточные значения активаций, необходимые для вычисления градиентов при обратном распространении ошибки (backpropagation). В режиме инференса (предсказания) обратное распространение не выполняется, поэтому хранение графа является пустой тратой видеопамяти (VRAM) и процессорного времени.

- **torch.no_grad():** Явно отключает движок автограда (autograd engine). Это предотвращает аллокацию памяти под градиенты, позволяя использовать существенно большие размеры батчей и ускоряя вычисления.14
- **torch.inference_mode():** Введенный в PyTorch 1.9 более строгий и эффективный режим. В дополнение к отключению градиентов, он отключает отслеживание представлений (view tracking) и счетчики версий тензоров. Это позволяет добиться дополнительного ускорения (на 5-10%) и снижения накладных расходов Python. Документация PyTorch рекомендует использовать именно inference_mode для сценариев чистого инференса, однако no_grad остается стандартом в большинстве унаследованных систем.16

Игнорирование этих менеджеров при инференсе является классической ошибкой, приводящей к "утечкам памяти" (memory leaks). Граф вычислений будет накапливаться в памяти с каждым проходом, пока приложение не исчерпает доступную RAM/VRAM и не упадет с ошибкой OOM (Out Of Memory).18

### **3.3 Проблема утечек памяти при инференсе**

Даже при использовании torch.no_grad(), инженеры часто сталкиваются с утечками памяти, если неправильно обрабатывают выходные данные. Например, накопление тензоров логитов в списке Python для последующего анализа (например, all_logits.append(outputs.logits)) может привести к удержанию ссылок на весь вычислительный граф, если тензоры не были явно отсоединены. Хотя no_grad предотвращает создание графа, лучшей практикой является явное перемещение тензоров на CPU и конвертация в NumPy или простые типы Python, если они больше не нужны для вычислений на GPU: outputs.logits.detach().cpu().numpy().18

## **Часть IV. Природа логитов: Сырые мысли Голема**

Самый концептуально сложный момент ручного инференса — интерпретация выходных данных. Модель возвращает не вероятности и не метки классов, а **логиты** (logits). В приведенном примере это тензор вида \[\[-2.4, 2.6\]\].

### **4.1 Математическая и геометрическая интерпретация**

Логиты — это сырые, ненормализованные результаты линейного преобразования последнего слоя нейросети: $z \= Wx \+ b$, где $x$ — входной вектор (эмбеддинг \`\`), $W$ — матрица весов, $b$ — вектор смещения.  
Геометрически логит можно интерпретировать как расстояние (со знаком) от классифицируемого объекта до разделяющей гиперплоскости в многомерном пространстве признаков.

- Значение 0 означает, что объект находится прямо на границе принятия решений (максимальная неопределенность).
- Большое положительное значение (например, 2.6) означает, что объект находится далеко в глубине области, соответствующей данному классу.
- Отрицательное значение (например, \-2.4) означает удаление в противоположную сторону.  
  В отличие от вероятностей, ограниченных диапазоном $$, логиты определены на всей числовой прямой $(-\\infty, \+\\infty)$. Это свойство делает их более информативными для обучения, так как они сохраняют информацию о "силе" уверенности даже в областях, где вероятности насыщаются (vanishing gradients).20

### **4.2 Почему модели возвращают логиты, а не вероятности**

Существуют две основные причины, по которым инженеры и архитекторы моделей предпочитают работать с логитами:

1. **Численная стабильность:** Преобразование логитов в вероятности требует функции Softmax, которая использует экспоненцирование ($e^x$). Для очень больших или очень маленьких значений $x$ это может привести к переполнению (overflow) или потере значимости (underflow) чисел с плавающей точкой. Многие функции потерь (например, CrossEntropyLoss в PyTorch) ожидают на входе именно логиты, чтобы внутри себя применить оптимизированную версию LogSoftmax (функция log-sum-exp), которая математически эквивалентна, но вычислительно более стабильна.22
2. **Вычислительная эффективность:** Для задачи жесткой классификации (выбора одного победителя) вычисление точных вероятностей избыточно. Функция экспоненты — одна из самых дорогих арифметических операций для процессора. Работа с логитами позволяет избежать этих затрат.

### **4.3 Теория принятия решений: Argmax против Softmax**

В финальном этапе скрипта используется операция argmax. Рассмотрим математическое обоснование этого выбора.

- **Softmax:** Преобразует вектор $z$ в вектор вероятностей $\\sigma(z)\_i \= \\frac{e^{z\_i}}{\\sum\_{j} e^{z\_j}}$.
- **Монотонность:** Экспонента $e^x$ является строго монотонно возрастающей функцией. Это означает, что если $z\_A \> z\_B$, то и $e^{z\_A} \> e^{z\_B}$, и следовательно $P(A) \> P(B)$.
- **Инженерный вывод:** Порядок элементов (ранжирование) в векторе логитов строго совпадает с порядком в векторе вероятностей. Индекс максимального логита _всегда_ будет совпадать с индексом максимальной вероятности.

Следовательно, применение argmax непосредственно к логитам дает тот же самый результат (класс-победитель), что и применение argmax к вероятностям, но экономит вычислительный ресурс, исключая шаг Softmax. В системах с высокой пропускной способностью (High-Frequency Trading, Real-Time Bidding), где обрабатываются миллионы запросов в секунду, исключение "лишних" трансцендентных операций дает измеримый выигрыш в латентности.24 Softmax необходим только тогда, когда нужна _калиброванная уверенность_ (например, "мы уверены на 80%"), а не просто факт классификации.

## **Часть V. Пост-процессинг и семантика SST-2**

Финальный аккорд — превращение безликого индекса в человекочитаемый вердикт.

### **5.1 Контекст набора данных SST-2**

Модель finetuned-sst-2 была дообучена на наборе данных Stanford Sentiment Treebank (SST-2). Изначально этот датасет, созданный на основе обзоров Rotten Tomatoes, был революционным, так как содержал разметку тональности не только для полных предложений, но и для каждого узла в синтаксическом дереве разбора (parse tree). Это позволяло рекурсивным нейросетям (RNTN) учить композициональность языка (например, как слово "не" инвертирует тональность слова "плохо" в фразе "не плохо").26  
В версии SST-2, используемой для бенчмарка GLUE и обучения данной модели, метки упрощены до бинарных: 0 (негативный) и 1 (позитивный). Нейтральные фразы исключены, а метки даны только для полных предложений. Это упрощение создает четкую границу принятия решений, но лишает модель способности выражать амбивалентность.28

### **5.2 Конфигурация id2label**

Связь между индексом 1 и словом POSITIVE не зашита в архитектуру нейросети; это метаданные. Эти данные хранятся в файле конфигурации модели (config.json) в словаре id2label.  
В скрипте мы обращаемся к golem.config.id2label\[strongest_thought_id\]. Это паттерн надежного программирования: вместо хардкода (if id \== 1: print("Positive")), мы полагаемся на внутреннюю схему модели. Если завтра мы заменим модель на ту, что классифицирует эмоции (0: 'Sad', 1: 'Joy', 2: 'Anger'), код продолжит работать корректно без изменений логики, автоматически подхватывая новые метки.29

## **Часть VI. Продвинутые инженерные паттерны: За пределами одного запроса**

Хотя "Квест 1.5" фокусируется на обработке одной фразы, освоение ручного инференса открывает двери к паттернам, необходимым для масштабирования систем искусственного интеллекта.

### **6.1 Ручной батчинг и пропускная способность (Throughput)**

Главное преимущество отказа от pipeline — возможность реализации эффективного батчинга (пакетной обработки). Графические процессоры (GPU) являются массивно-параллельными устройствами. Обработка одного предложения на GPU неэффективна — это сродни перевозке одной коробки на грузовом поезде.  
При ручном инференсе инженер может собрать список из, например, 32 предложений:

Python

batch \= \["Фраза 1", "Фраза 2",..., "Фраза 32"\]  
inputs \= tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

Критическим моментом здесь является стратегия padding. В pipeline часто используется статическая длина (например, добивание до 512 токенов), что приводит к огромному количеству холостых вычислений над нулями. В ручном режиме можно использовать **динамический паддинг** (padding=True или padding='longest'), когда длина батча выравнивается по самому длинному предложению _внутри этого батча_. Если все предложения в батче короткие (10-15 слов), тензор будет размером $32 \\times 15$, а не $32 \\times 512$. Это снижает вычислительную сложность с $O(N^2)$ до фактической длины, ускоряя инференс в десятки раз.30

### **6.2 Управление устройствами и типичные ошибки**

Ручной инференс требует явного управления размещением тензоров. По умолчанию тензоры создаются в оперативной памяти (CPU). Для ускорения их необходимо переместить на GPU:

Python

device \= "cuda" if torch.cuda.is_available() else "cpu"  
model.to(device)  
inputs \= {k: v.to(device) for k, v in inputs.items()}

Одной из самых частых ошибок является "Device Mismatch Error" (RuntimeError: Expected all tensors to be on the same device), возникающая, когда модель находится на GPU, а входные тензоры — на CPU (или наоборот). Абстракция pipeline скрывает это управление, но при ручном контроле инженер обязан гарантировать когерентность устройств.9

### **6.3 Производительность: Pipeline против vLLM и оптимизированных серверов**

Сравнение производительности показывает, что для высоконагруженных оффлайн-задач даже ручной инференс на PyTorch уступает специализированным решениям. Бенчмарки показывают, что библиотеки типа **vLLM** (использующие PagedAttention и непрерывный батчинг) или **TGI** (Text Generation Inference) могут превосходить стандартный Hugging Face инференс в разы по пропускной способности.32 Однако, понимание механики, описанной в этом отчете (токены, логиты, маски внимания), является фундаментом для настройки и этих продвинутых систем. Например, параметры max_num_batched_tokens в vLLM напрямую апеллируют к пониманию того, как длина последовательности и размер батча влияют на аллокацию памяти, изученному нами на этапе анализа тензоров input_ids.33

### **6.4 Метрики Латентности (Latency) и Пропускной способности (Throughput)**

При проектировании систем инференса важно различать эти две метрики, так как они часто находятся в конфликте.

- **Латентность (Time To First Token / End-to-End Latency):** Время обработки одного запроса. Критично для онлайн-чатов. Здесь выгоден малый размер батча (часто 1). Ручной инференс позволяет минимизировать накладные расходы Python, улучшая эту метрику.33
- **Пропускная способность (Tokens per Second):** Количество обработанных данных в единицу времени. Критично для оффлайн-аналитики (например, анализ тональности архива новостей за год). Здесь выгоден максимальный размер батча, заполняющий всю память GPU.

## **Заключение: Искусство "Белого ящика"**

Переход от pipeline к AutoModel — это больше, чем смена синтаксиса; это фундаментальный сдвиг в инженерном мышлении. Пользователь pipeline относится к ИИ как к потребительскому продукту — черному ящику, магически выдающему ответы. Инженер, владеющий ручным инференсом, видит ИИ как вычислительную систему — серию операций линейной алгебры, поддающуюся профилированию, отладке и оптимизации.

"Квест 1.5" демонстрирует, что "разум Голема" — это не абстрактное психологическое пространство, а осязаемое векторное поле. Его "мысли" — это логиты, его "инстинкты" — это веса, а его "чувства" — это токенизаторы. Понимая природу этих компонентов, инженер получает власть не просто использовать модель, но формировать её поведение, встраивать в жесткие рамки архитектурных требований и извлекать максимум из доступного кремния. Магия, как оказалось, — это просто математика, и понимание этой математики дает абсолютный контроль.

---

**Таблица 1\. Сравнительный анализ методов инференса**

| Характеристика             | pipeline(...)               | Ручной AutoModel                      | vLLM / TGI                                 |
| :------------------------- | :-------------------------- | :------------------------------------ | :----------------------------------------- |
| **Сложность кода**         | Низкая (1-2 строки)         | Средняя (10-20 строк)                 | Высокая (требует отдельного сервера)       |
| **Гибкость**               | Низкая (стандартные задачи) | Высокая (любая логика)                | Ограничена API сервера                     |
| **Отладка**                | Затруднена ("Черный ящик")  | Прозрачна (доступ ко всем тензорам)   | Через логи сервера                         |
| **Производительность**     | Низкая (накладные расходы)  | Средняя/Высокая (зависит от инженера) | Экстремальная (оптимизированные ядра CUDA) |
| **Сценарий использования** | Прототипы, обучение         | Кастомные сервисы, сложная логика     | High-load продакшн                         |

Источники данных (Data Sources):  
4 Hugging Face Model Card: distilbert-base-uncased  
34 Hugging Face Model Files: distilbert-base-uncased-finetuned-sst-2-english  
24 Hugging Face Model Card: Uses & Limitations  
1 Hugging Face Docs: Pipeline Tutorial (Batching strategies)  
20 Medium: Logits vs Probabilities (Numerical stability)  
22 GeeksforGeeks: Softmax vs SoftmaxCrossEntropy  
21 Medium: Understanding Softmax & Logits Geometric interpretation  
25 StackExchange: Logits vs Scaled Probabilities  
16 PyTorch Docs: Inference Mode mechanics  
14 StackOverflow: torch.no_grad() usage benefits  
17 StackOverflow: no_grad vs inference_mode comparison  
6 Hugging Face Docs: Tokenizer return_tensors='pt'  
8 StackOverflow: Why use return_tensors='pt'  
26 YouTube: Stanford Sentiment Treebank Explained  
28 GitHub Issues: SST-2 label mapping details  
4 Hugging Face: DistilBERT Architecture & Distillation Loss  
5 Hugging Face: DistilBERT Config & Tokenization details  
29 Hugging Face: Config JSON (id2label dictionary)  
2 Discuss Hugging Face: Pipeline vs Model Generate overhead  
3 Discuss Hugging Face: Debugging pipeline failures  
13 Reddit: AutoModel vs AutoModelForSequenceClassification difference  
7 GitHub Issues: Tokenizer return types in map()  
15 Medium: Inference in PyTorch (Memory usage)  
35 Discuss PyTorch: Inference Mode safety checks  
32 Medium: vLLM vs Hugging Face benchmarks  
18 Discuss PyTorch: Memory leak debugging  
19 Discuss PyTorch: Memory leaks at inference (detach() usage)  
10 Blog: What are Attention Masks?  
11 Hugging Face Course: Attention masks and padding  
30 McCormickML: Smart Batching Tutorial (Dynamic Padding)  
32 Medium: vLLM scalability analysis  
33 Databricks: LLM Inference Performance Engineering (TTFT, TPOT)  
9 Hugging Face Docs: Accelerate Troubleshooting (Mismatched shapes)  
31 StackOverflow: Mismatched tensor size error debugging

#### **Источники**

1. Pipeline \- Hugging Face, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/docs/transformers/en/pipeline_tutorial](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
2. Difference between pipeline and model.generate? \- Transformers \- Hugging Face Forums, дата последнего обращения: ноября 23, 2025, [https://discuss.huggingface.co/t/difference-between-pipeline-and-model-generate/35015](https://discuss.huggingface.co/t/difference-between-pipeline-and-model-generate/35015)
3. Pipeline vs model.generate() \- Beginners \- Hugging Face Forums, дата последнего обращения: ноября 23, 2025, [https://discuss.huggingface.co/t/pipeline-vs-model-generate/26203](https://discuss.huggingface.co/t/pipeline-vs-model-generate/26203)
4. distilbert/distilbert-base-uncased \- Hugging Face, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
5. DistilBERT \- Hugging Face, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/docs/transformers/en/model_doc/distilbert](https://huggingface.co/docs/transformers/en/model_doc/distilbert)
6. Tokenizer — transformers 3.5.0 documentation \- Hugging Face, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/transformers/v3.5.1/main_classes/tokenizer.html](https://huggingface.co/transformers/v3.5.1/main_classes/tokenizer.html)
7. Why return_tensors='pt' doesn't work？ · Issue \#7291 · huggingface/datasets \- GitHub, дата последнего обращения: ноября 23, 2025, [https://github.com/huggingface/datasets/issues/7291](https://github.com/huggingface/datasets/issues/7291)
8. Why we use return_tensors \= "pt" during tokenization? \- Stack Overflow, дата последнего обращения: ноября 23, 2025, [https://stackoverflow.com/questions/78095157/why-we-use-return-tensors-pt-during-tokenization](https://stackoverflow.com/questions/78095157/why-we-use-return-tensors-pt-during-tokenization)
9. Troubleshoot \- Hugging Face, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/docs/accelerate/en/basic_tutorials/troubleshooting](https://huggingface.co/docs/accelerate/en/basic_tutorials/troubleshooting)
10. What Are Attention Masks? \- Luke Salamone's Blog, дата последнего обращения: ноября 23, 2025, [https://blog.lukesalamone.com/posts/what-are-attention-masks/](https://blog.lukesalamone.com/posts/what-are-attention-masks/)
11. Handling multiple sequences \- Hugging Face LLM Course, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/learn/llm-course/en/chapter2/5](https://huggingface.co/learn/llm-course/en/chapter2/5)
12. Handling multiple sequences \- Medium, дата последнего обращения: ноября 23, 2025, [https://medium.com/@danushidk507/handling-multiple-sequences-553aed538dbd](https://medium.com/@danushidk507/handling-multiple-sequences-553aed538dbd)
13. what is the difference between AutoModelForCausalLM and AutoModel? : r/huggingface, дата последнего обращения: ноября 23, 2025, [https://www.reddit.com/r/huggingface/comments/1bv1kfk/what_is_the_difference_between/](https://www.reddit.com/r/huggingface/comments/1bv1kfk/what_is_the_difference_between/)
14. What is the use of torch.no_grad in pytorch? \- Data Science Stack Exchange, дата последнего обращения: ноября 23, 2025, [https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch](https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch)
15. Inference in PyTorch: Understanding the Wrappers and Choosing the Best \- Medium, дата последнего обращения: ноября 23, 2025, [https://medium.com/@whyamit404/inference-in-pytorch-understanding-the-wrappers-and-choosing-the-best-d4f16fbde960](https://medium.com/@whyamit404/inference-in-pytorch-understanding-the-wrappers-and-choosing-the-best-d4f16fbde960)
16. inference_mode — PyTorch 2.9 documentation, дата последнего обращения: ноября 23, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
17. PyTorch \`torch.no_grad\` vs \`torch.inference_mode\` \- Stack Overflow, дата последнего обращения: ноября 23, 2025, [https://stackoverflow.com/questions/69543907/pytorch-torch-no-grad-vs-torch-inference-mode](https://stackoverflow.com/questions/69543907/pytorch-torch-no-grad-vs-torch-inference-mode)
18. Memory leak (no_grad) when accumulating dataloader outputs \- autograd \- PyTorch Forums, дата последнего обращения: ноября 23, 2025, [https://discuss.pytorch.org/t/memory-leak-no-grad-when-accumulating-dataloader-outputs/175775](https://discuss.pytorch.org/t/memory-leak-no-grad-when-accumulating-dataloader-outputs/175775)
19. Memory leaks at inference \- PyTorch Forums, дата последнего обращения: ноября 23, 2025, [https://discuss.pytorch.org/t/memory-leaks-at-inference/85108](https://discuss.pytorch.org/t/memory-leaks-at-inference/85108)
20. Logits vs. Probabilities: Understanding Neural Network Outputs Clearly \- Illuri Sandeep, дата последнего обращения: ноября 23, 2025, [https://illuri-sandeep5454.medium.com/logits-vs-probabilities-understanding-neural-network-outputs-clearly-0e86a4256a0e](https://illuri-sandeep5454.medium.com/logits-vs-probabilities-understanding-neural-network-outputs-clearly-0e86a4256a0e)
21. From Logits to Probabilities: Understanding Softmax in Neural Networks | by Deepankar Singh | AI-Enthusiast | Medium, дата последнего обращения: ноября 23, 2025, [https://medium.com/ai-enthusiast/from-logits-to-probabilities-understanding-softmax-in-neural-networks-3ebea2e95cfe](https://medium.com/ai-enthusiast/from-logits-to-probabilities-understanding-softmax-in-neural-networks-3ebea2e95cfe)
22. Difference Between Softmax and Softmax_Cross_Entropy_With_Logits \- GeeksforGeeks, дата последнего обращения: ноября 23, 2025, [https://www.geeksforgeeks.org/data-science/difference-between-softmax-and-softmaxcrossentropywithlogits/](https://www.geeksforgeeks.org/data-science/difference-between-softmax-and-softmaxcrossentropywithlogits/)
23. What are logits? What is the difference between softmax and softmax_cross_entropy_with_logits? \- Stack Overflow, дата последнего обращения: ноября 23, 2025, [https://stackoverflow.com/questions/34240703/what-are-logits-what-is-the-difference-between-softmax-and-softmax-cross-entrop](https://stackoverflow.com/questions/34240703/what-are-logits-what-is-the-difference-between-softmax-and-softmax-cross-entrop)
24. distilbert/distilbert-base-uncased-finetuned-sst-2-english \- Hugging Face, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
25. Should I use the logits or the scaled probabilities from them to extract my predictions?, дата последнего обращения: ноября 23, 2025, [https://stats.stackexchange.com/questions/260933/should-i-use-the-logits-or-the-scaled-probabilities-from-them-to-extract-my-pred](https://stats.stackexchange.com/questions/260933/should-i-use-the-logits-or-the-scaled-probabilities-from-them-to-extract-my-pred)
26. Stanford Sentiment Treebank (SST) \- YouTube, дата последнего обращения: ноября 23, 2025, [https://www.youtube.com/watch?v=3U-UkCmJQu4](https://www.youtube.com/watch?v=3U-UkCmJQu4)
27. Stanford Sentiment Treebank v2 (SST2) \- Kaggle, дата последнего обращения: ноября 23, 2025, [https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2](https://www.kaggle.com/datasets/atulanandjha/stanford-sentiment-treebank-v2-sst2)
28. Add Stanford Sentiment Treebank (SST) · Issue \#1934 · huggingface/datasets \- GitHub, дата последнего обращения: ноября 23, 2025, [https://github.com/huggingface/datasets/issues/1934](https://github.com/huggingface/datasets/issues/1934)
29. config.json · distilbert/distilbert-base-uncased-finetuned-sst-2-english at main, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blame/main/config.json](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blame/main/config.json)
30. Smart Batching Tutorial \- Speed Up BERT Training \- Chris McCormick, дата последнего обращения: ноября 23, 2025, [https://mccormickml.com/2020/07/29/smart-batching-tutorial/](https://mccormickml.com/2020/07/29/smart-batching-tutorial/)
31. Mismatched tensor size error when generating text with beam_search (huggingface library), дата последнего обращения: ноября 23, 2025, [https://stackoverflow.com/questions/67221901/mismatched-tensor-size-error-when-generating-text-with-beam-search-huggingface](https://stackoverflow.com/questions/67221901/mismatched-tensor-size-error-when-generating-text-with-beam-search-huggingface)
32. vLLM vs Hugging Face for High-Performance LLM Inference | by Ali Shafique | Medium, дата последнего обращения: ноября 23, 2025, [https://medium.com/@alishafique3/vllm-vs-hugging-face-for-high-performance-offline-llm-inference-2d953b4fb3b4](https://medium.com/@alishafique3/vllm-vs-hugging-face-for-high-performance-offline-llm-inference-2d953b4fb3b4)
33. LLM Inference Performance Engineering: Best Practices | Databricks Blog, дата последнего обращения: ноября 23, 2025, [https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
34. distilbert/distilbert-base-uncased-finetuned-sst-2-english at main \- Hugging Face, дата последнего обращения: ноября 23, 2025, [https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/main](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/main)
35. PyTorch \`torch.no_grad\` vs \`torch.inference_mode\` \- autograd, дата последнего обращения: ноября 23, 2025, [https://discuss.pytorch.org/t/pytorch-torch-no-grad-vs-torch-inference-mode/134099](https://discuss.pytorch.org/t/pytorch-torch-no-grad-vs-torch-inference-mode/134099)
