# **Отчет об исследовании архитектуры MLOps: Реализация слепого A/B тестирования для моделей анализа тональности**

## **1\. Введение: Парадигма эксперимента в промышленном машинном обучении**

Переход моделей машинного обучения из изолированных исследовательских сред (Jupyter Notebooks, локальные серверы) в высоконагруженные промышленные системы (Production) представляет собой один из наиболее критических этапов жизненного цикла MLOps. Статистические метрики, полученные на валидационных наборах данных (offline metrics) — такие как точность (Accuracy), F1-мера или ROC-AUC — часто оказываются недостаточными предикторами поведения модели в реальных условиях.1 Это расхождение обусловлено стохастической природой пользовательского ввода, дрейфом данных (data drift) и жесткими требованиями к задержке ответа (latency), которые невозможно полноценно симулировать в оффлайн-режиме.  
В рамках выполнения Квеста 30.2 «Слепое Испытание» была поставлена задача разработать и теоретически обосновать архитектуру системы для проведения «слепого» A/B тестирования двух версий моделей анализа тональности (Sentiment Analysis). Термин «слепое» в данном контексте подразумевает, что конечный пользователь не осведомлен о том, какая именно версия модели обрабатывает его запрос, что исключает предвзятость в пользовательском поведении и позволяет собрать объективные данные о качестве работы алгоритмов.2  
Данный отчет представляет собой исчерпывающее руководство по реализации такой системы. В нем рассматривается интеграция микросервиса на базе **FastAPI**, системы сбора метрик **Prometheus** и платформы визуализации **Grafana**. Особое внимание уделяется сравнительному анализу двух архитектур трансформеров: легковесной модели **DistilBERT** и робастной модели **RoBERTa**, а также методологии их параллельного развертывания с использованием механизмов маршрутизации трафика и липких сессий (sticky sessions).

### **1.1 Актуальность онлайн-экспериментов**

В современной практике MLOps утвердилось мнение, что «модель не готова, пока она не протестирована в бою». Оффлайн-метрики демонстрируют способность модели запоминать и обобщать обучающую выборку, но A/B тестирование — это проверка способности модели влиять на бизнес-метрики (Overall Evaluation Criteria — OEC).1 Например, модель с более высокой точностью может иметь настолько высокую задержку инференса, что это приведет к оттоку пользователей, делая ее внедрение экономически нецелесообразным. Слепое A/B тестирование позволяет найти баланс между качеством предсказаний и эксплуатационными характеристиками.3

## ---

**2\. Теоретический базис и сравнительный анализ стратегий развертывания**

Прежде чем переходить к технической реализации, необходимо четко разграничить применяемые стратегии развертывания, так как выбор стратегии диктует архитектурные требования.

### **2.1 Таксономия стратегий деплоя в MLOps**

Существует несколько подходов к обновлению моделей, каждый из которых решает специфические задачи:

- **Canary Deployment (Канареечный релиз):** Эта стратегия ориентирована на минимизацию рисков технического характера. Новая версия модели (Canary) раскатывается на малый процент аудитории (например, 5-10%). Основная цель — выявить сбои, утечки памяти или критические ошибки (5xx errors) до того, как они затронут всех пользователей. Если метрики стабильны, трафик постепенно переключается полностью. Сравнение качества моделей здесь вторично.4
- **Blue/Green Deployment:** Подразумевает наличие двух идентичных контуров (Blue — текущий, Green — новый). Переключение трафика происходит мгновенно и полностью. Это обеспечивает возможность быстрого отката (rollback), но не позволяет проводить одновременное сравнение эффективности моделей на живом трафике.4
- **Shadow Deployment (Теневое развертывание):** Новая модель получает копию реального трафика, но ее ответы не возвращаются пользователю. Это позволяет проверить производительность и корректность работы «вхолостую». Недостатком является удвоение нагрузки на инфраструктуру без получения обратной связи от пользователей.7
- **A/B Testing (Сплит-тестирование):** Стратегия, выбранная для данного исследования. Две или более версий модели работают параллельно, обслуживая разные сегменты пользователей. Цель — статистически значимое сравнение бизнес-метрик или качества продукта. Именно этот подход позволяет ответить на вопрос: «Какая модель лучше понимает пользователя?».1

### **2.2 Методология «Слепого» теста**

В контексте машинного обучения «слепое» тестирование требует реализации механизма **Sticky Sessions** (липких сессий). В отличие от веб\-дизайна, где можно менять цвет кнопки при каждой перезагрузке страницы, в NLP-задачах непоследовательность ответов модели может разрушить пользовательский опыт. Если пользователь отправляет текст и получает оценку «Позитивно», а через секунду при повторной отправке того же текста получает «Нейтрально» (из-за переключения на другую модель), доверие к сервису падает. Следовательно, архитектура должна гарантировать, что конкретный пользователь (или сессия) на протяжении всего эксперимента привязан к одной версии модели (варианту A или B).8

## ---

**3\. Сравнительный анализ архитектур моделей: Претенденты**

Для эксперимента были выбраны две модели семейства BERT, представляющие собой классический компромисс между скоростью (Performance) и качеством (Accuracy).

### **3.1 Модель A (Challenger): DistilBERT**

В качестве легковесного претендента выступает архитектура **DistilBERT**, конкретно вариант distilbert-base-uncased-finetuned-sst-2-english.

- **Архитектурные особенности:** DistilBERT является результатом процесса дистилляции знаний (knowledge distillation), где «студент» (DistilBERT) обучается повторять поведение «учителя» (BERT). В результате модель имеет на 40% меньше параметров (\~66 млн) за счет сокращения количества слоев с 12 до 6, при сохранении размерности скрытого пространства.10
- **Профиль производительности:** Основное преимущество DistilBERT — скорость. На CPU инференс этой модели выполняется в среднем на 60% быстрее, чем у полного BERT, что делает её идеальным кандидатом для высоконагруженных систем реального времени.12
- **Обучающая выборка (SST-2):** Данная версия дообучена на наборе данных Stanford Sentiment Treebank (SST-2). Это _бинарный_ датасет, содержащий только позитивные и негативные метки. Это создает фундаментальное ограничение: модель не умеет классифицировать нейтральные высказывания, принудительно относя их к одному из полюсов.14

### **3.2 Модель B (Control/High-Quality): RoBERTa**

В качестве тяжеловесной, но более точной альтернативы выбрана **RoBERTa** (Robustly Optimized BERT Pretraining Approach), вариант cardiffnlp/twitter-roberta-base-sentiment-latest.

- **Архитектурные особенности:** RoBERTa сохраняет архитектуру BERT-base (12 слоев, \~125 млн параметров), но кардинально меняет процесс предобучения. Исключена задача предсказания следующего предложения (NSP), увеличена длина последовательностей и размер батча, а маскирование токенов происходит динамически при каждой эпохе обучения. Это позволяет модели лучше улавливать контекст.11
- **Профиль производительности:** За качество приходится платить ресурсами. RoBERTa требует примерно на 20% больше оперативной памяти и выполняется значительно медленнее DistilBERT (инференс может занимать 100-150 мс против 50-60 мс у DistilBERT на аналогичном оборудовании).17
- **Обучающая выборка (TweetEval):** Модель обучена на \~124 миллионах твитов. Это дает ей критическое преимущество в понимании сленга, эмодзи и неформальной лексики. Важно, что она поддерживает _тернарную_ классификацию (Негатив, Нейтрально, Позитив), что делает её более гибкой для анализа реальной человеческой речи.18

### **3.3 Гипотеза эксперимента**

Ключевая гипотеза A/B теста формулируется следующим образом: _«Использование модели RoBERTa повысит качество распознавания нейтральных и саркастичных комментариев (Customer Satisfaction), однако увеличит среднюю задержку ответа (Latency). Тест должен показать, оправдывает ли прирост качества увеличение расходов на инфраструктуру и возможную потерю части пользователей из\-за долгого ожидания»_.3

## ---

**4\. Инженерная реализация уровня приложений (FastAPI)**

Центральным элементом системы является микросервис на Python с использованием фреймворка **FastAPI**. Выбор обусловлен его асинхронной природой, позволяющей эффективно обрабатывать I/O-операции и интегрироваться с современными MLOps инструментами.20

### **4.1 Управление жизненным циклом моделей**

Загрузка ML-моделей — ресурсоемкая операция. Критически важно не загружать веса моделей внутри обработчика запроса (request handler), так как это приведет к многосекундным задержкам на каждый вызов.  
Реализация использует механизм lifespan (или события startup в старых версиях FastAPI) для инициализации глобальных объектов пайплайнов Hugging Face. При старте приложения в память загружаются оба объекта: pipeline("sentiment-analysis", model="distilbert...") и pipeline("sentiment-analysis", model="roberta..."). Это увеличивает потребление RAM (Cold Start), но обеспечивает минимальную задержку при обработке запросов (Hot Path).21

### **4.2 Маршрутизация трафика: Middleware Pattern**

Для реализации логики A/B тестирования применяется паттерн **Middleware**. Это позволяет отделить бизнес-логику (предикшн) от логики экспериментов (распределение трафика).  
Алгоритм работы Middleware:

1. **Перехват запроса:** Middleware перехватывает каждый входящий HTTP-запрос до того, как он достигнет эндпоинта.
2. **Идентификация сессии:** Система проверяет наличие cookie с идентификатором model_session.
   - Если cookie есть, извлекается назначенная группа (например, model_a или model_b).
   - Если cookie нет, генерируется новая сессия.
3. **Взвешенное случайное распределение:** Для новой сессии происходит выбор модели. Использование простого random.choice недостаточно для сложных сценариев. Рекомендуется использовать random.choices (доступен в Python 3.6+), который поддерживает весовые коэффициенты. Это позволяет реализовать не только распределение 50/50, но и канареечные релизы (например, 90/10).23  
   Python  
   \# Пример логики взвешенного выбора  
   selected_model \= random.choices(  
    population=\['distilbert', 'roberta'\],  
    weights=\[0.5, 0.5\],  
    k=1  
   )

4. **Внедрение контекста:** Выбранная модель сохраняется в объекте request.state, делая её доступной для основного обработчика.
5. **Фиксация сессии:** При формировании ответа Middleware добавляет заголовок Set-Cookie, закрепляя выбор за пользователем на заданный срок (TTL).9

### **4.3 Проблема Stateless-клиентов и Хеширование**

В сценариях межсервисного взаимодействия (Service-to-Service), где клиенты могут не поддерживать cookies, полагаться на них нельзя. В таких случаях применяется детерминированное хеширование устойчивого идентификатора (например, user_id или IP-адреса).  
Формула маршрутизации:

$$Variant \= HASH(UserID) \\mod 100$$

Если результат $\< 50$, назначается Модель A; иначе — Модель B. Этот подход гарантирует, что один и тот же user_id всегда попадет на одну и ту же версию модели, даже при перезагрузке серверов, без необходимости хранения состояния.8

### **4.4 Обработка различий в выходных данных**

Поскольку модели имеют разные форматы выходных меток (DistilBERT: бинарный, RoBERTa: тернарный), слой приложения должен выполнять нормализацию. Результаты RoBERTa могут быть либо огрублены (Нейтральный \-\> Позитивный/Негативный в зависимости от порога score), либо передаваться «как есть» для более глубокой аналитики. В рамках «слепого» теста рекомендуется возвращать сырые метки и нормализовывать их уже на этапе аналитики в Grafana.27

## ---

**5\. Инженерная реализация наблюдаемости (Prometheus)**

Без качественной телеметрии A/B тестирование бесполезно. Система мониторинга должна отвечать на вопросы о трафике, задержках и распределении предсказаний.

### **5.1 Дизайн метрик и проблема кардинальности**

Использование библиотеки prometheus-client требует тщательного проектирования меток (labels). Распространенной ошибкой является добавление user_id или session_id в качестве метки. В Prometheus это приводит к взрывному росту кардинальности (High Cardinality), создавая миллионы уникальных временных рядов, что неминуемо обрушивает базу данных временных рядов (TSDB).29  
Оптимальный набор метрик для задачи:

1. **Counter:** sentiment_requests_total
   - **Метки:** model_version (distilbert/roberta), status (success/error), sentiment_label (positive/negative/neutral).
   - **Назначение:** Позволяет сравнивать объем трафика (проверка корректности сплита) и распределение ответов моделей.
2. **Histogram:** sentiment_inference_seconds
   - **Метки:** model_version.
   - **Бакеты (Buckets):** Должны быть настроены экспоненциально в диапазоне от 0.01с до 5.0с, чтобы захватить как быстрые ответы DistilBERT, так и возможные «хвосты» RoBERTa.
   - **Назначение:** Вычисление перцентилей задержки (P50, P95, P99).32
3. **Gauge:** model_memory_usage_bytes (опционально)
   - **Назначение:** Отслеживание потребления RAM каждым процессом/контейнером для оценки стоимости инфраструктуры.33

### **5.2 Экспорт метрик**

Для интеграции FastAPI с Prometheus используется библиотека prometheus-fastapi-instrumentator. Она автоматически оборачивает эндпоинты и экспонирует стандартные метрики. Однако для A/B теста необходимо вручную инструментировать код бизнес-логики (Custom Instrumentation), чтобы фиксировать специфические события (например, какую именно метку предсказала модель).20

### **5.3 Сетевая топология и DNS (Docker Compose)**

Все компоненты (API, Prometheus, Grafana) оркестрируются через Docker Compose. Важный аспект — разрешение имен (DNS Resolution). Prometheus настраивается на скрейпинг (сбор данных) с сервиса API по его имени в сети Docker (например, http://api_service:8000/metrics).  
Файл конфигурации prometheus.yml определяет интервал сбора (scrape_interval). Для A/B тестов, где важна динамика, рекомендуется интервал 5–15 секунд.35

## ---

**6\. Логика запросов и аналитика (PromQL)**

Язык запросов PromQL позволяет преобразовывать сырые временные ряды в аналитические инсайты.

### **6.1 Анализ распределения трафика**

Для подтверждения того, что алгоритм балансировки работает корректно (например, 50/50), используется запрос скорости (rate) запросов:

Фрагмент кода

sum by (model_version) (rate(sentiment_requests_total\[5m\]))

Этот запрос суммирует скорость запросов в секунду (RPS) для каждой версии модели. График должен показывать две линии, идущие близко друг к другу. Существенное расхождение сигнализирует о баге в Middleware или проблеме с хешированием.37

### **6.2 Сравнительный анализ задержек (Latency P99)**

Сравнение средней задержки часто скрывает проблемы. Для оценки пользовательского опыта (UX) критически важен 99-й перцентиль (P99) — время, за которое обрабатываются 99% запросов.

Фрагмент кода

histogram_quantile(0.99, sum(rate(sentiment_inference_seconds_bucket\[5m\])) by (le, model_version))

Ожидается, что P99 для RoBERTa будет значительно выше, чем для DistilBERT. Если разница превышает допустимый порог (SLA), внедрение RoBERTa может быть заблокировано, несмотря на качество.39

### **6.3 Коэффициент ошибок (Error Rate)**

Важно отслеживать, не приводит ли новая модель к росту технических ошибок (таймаутов, OOM):

Фрагмент кода

sum(rate(sentiment_requests_total{status=\~"5.."}\[5m\])) by (model_version)  
/  
sum(rate(sentiment_requests_total\[5m\])) by (model_version)

Этот запрос показывает долю ошибочных запросов относительно общего трафика для каждой модели.38

### **6.4 Семантическое расхождение (Sentiment Skew)**

Наиболее интересный бизнесовый инсайт дает сравнение распределения предсказанных классов.

Фрагмент кода

sum by (sentiment_label, model_version) (sentiment_requests_total)

Анализ результатов этого запроса покажет фундаментальные различия моделей. Например, DistilBERT может классифицировать 60% сообщений как Positive, тогда как RoBERTa — только 40%, относя остальные 20% к Neutral. Это количественное выражение «слепоты» DistilBERT к нейтральным эмоциям.41

## ---

**7\. Стратегия визуализации (Grafana)**

Grafana выступает финальным интерфейсом для интерпретации результатов эксперимента. Правильная конфигурация панелей критична для предотвращения ложных выводов.

### **7.1 Панель статистики (Stat Panel) и Трансформации**

При отображении суммарного количества запросов часто возникает ошибка: использование функции Last для метрики типа Counter. Поскольку Counter постоянно растет, Last покажет общее число запросов с момента старта сервера, а не за выбранный период времени.  
Правильный подход: использование функции increase() в PromQL или применение трансформаций в Grafana.

- **Трансформация "Reduce" (Series to Rows):** Позволяет свернуть временной ряд в одно число (Total, Max, Min) для отображения в Stat Panel.
- **Расчет разницы:** С помощью трансформаций можно вычислить дельту между показателями модели A и B прямо в интерфейсе, отобразив, например, «RoBERTa на 15% медленнее».42

### **7.2 Визуализация временных рядов (Time Series)**

Для метрик задержки используется панель Time Series. Рекомендуется отображать одновременно P50 (медиана) и P99 для обеих моделей (итого 4 линии). Это позволяет визуально оценить стабильность работы. Если линии RoBERTa имеют частые пики (spikes), это может свидетельствовать о проблемах со сборкой мусора (GC) или конкуренцией за CPU из\-за тяжеловесности модели.44

### **7.3 Сложенные столбчатые диаграммы (Stacked Bar Chart)**

Для анализа тональности идеально подходят сложенные бары. По оси X — время, по оси Y — количество запросов, цвет сегмента — предсказанная тональность. Разделение графиков по model_version (через Row repeating или две отдельные панели) позволяет мгновенно увидеть разницу в «мнении» моделей об одном и том же потоке трафика.46

## ---

**8\. Интерпретация результатов и синтез данных**

Предполагаемые результаты эксперимента, основанные на технических характеристиках архитектур, формируют базу для принятия решений.

### **8.1 Анализ производительности (Performance)**

Согласно бенчмаркам, DistilBERT демонстрирует инференс в районе **60-70 мс** на стандартном CPU, в то время как RoBERTa показывает **100-120 мс**.12 В условиях высокой нагрузки (High Concurrency) эта разница может масштабироваться нелинейно. Если приложение чувствительно к задержке (например, чат-бот), увеличение времени отклика в 2 раза может быть неприемлемым. Визуализация в Grafana подтвердит это расхождение через метрику sentiment_inference_seconds.

### **8.2 Анализ качества (Quality)**

Ключевой инсайт эксперимента будет заключаться в обработке нейтральных сообщений.

- _Сценарий:_ Пользователь пишет «Посылка пришла вовремя, но упаковка помята».
- _DistilBERT (SST-2):_ Скорее всего, классифицирует как **Negative** (из-за слова «помята») или **Positive** (из-за «вовремя»), с низкой уверенностью. Бинарная природа SST-2 заставляет модель поляризовать мнение.
- _RoBERTa (TweetEval):_ С высокой вероятностью классифицирует как **Neutral**, так как обучалась на твитах, где смешанные эмоции — норма.

Если бизнес-цель — выявлять _ярость_ клиентов (критический негатив), то «ложно-негативные» срабатывания DistilBERT могут быть полезны (лучше перебдеть). Если цель — точная аналитика бренда, то RoBERTa даст более чистую картину, отсеяв информационный шум в категорию «Нейтрально».19

### **8.3 Экономическая эффективность**

Метрика model_memory_usage_bytes покажет, что контейнер с RoBERTa потребляет на 20-30% больше памяти. В облачной инфраструктуре (AWS/GCP) это может означать необходимость перехода на более дорогой инстанс (например, с t3.medium на t3.large). Отчет должен сопоставить: стоит ли прирост точности в 5-7% увеличения счета за инфраструктуру на 20-50%?.3

## ---

**9\. Заключение и стратегические рекомендации**

Реализация Квеста 30.2 демонстрирует, что A/B тестирование ML-моделей — это не просто переключение трафика, а комплексная инженерная задача, затрагивающая все уровни стека: от выбора архитектуры нейросети до настройки TCP-сокетов в Docker Compose.  
**Итоговые рекомендации:**

1. **Для задач поддержки клиентов (Customer Support):** Рекомендуется внедрение **RoBERTa**. Способность корректно интерпретировать нейтральные и смешанные отзывы перевешивает затраты на задержку. Риск ошибочной классификации нейтрального отзыва как негативного (что свойственно DistilBERT) может привести к перегрузке операторов поддержки.
2. **Для задач мониторинга трендов (Trend Monitoring):** Если система анализирует миллионы сообщений в секунду для выявления общих трендов, **DistilBERT** является предпочтительным выбором. Его пропускная способность выше, а на больших числах статистическая погрешность бинарной классификации нивелируется.
3. **Обязательность Sticky Sessions:** Эксперимент подтверждает, что без привязки сессии данные метрик становятся зашумленными. Реализация middleware с поддержкой cookies или детерминированного хеширования является обязательным архитектурным требованием.
4. **Визуализация:** Настройка дашбордов Grafana с правильным использованием функций rate() и histogram_quantile() — единственный способ получить объективную картину. Опора на «средние» значения недопустима.

Таким образом, система «слепого» A/B тестирования превращает процесс выбора модели из гадания на кофейной гуще оффлайн-метрик в точный инженерный расчет, основанный на реальных данных и бизнес-приоритетах.

#### **Источники**

1. A/B Testing for ML Models: Best Practices \- Statsig, дата последнего обращения: декабря 22, 2025, [https://www.statsig.com/perspectives/ab-testing-ml-models-best-practices](https://www.statsig.com/perspectives/ab-testing-ml-models-best-practices)
2. A practical guide to A/B Testing in MLOps with Kubernetes and Seldon Core \- Take Control of ML and AI Complexity, дата последнего обращения: декабря 22, 2025, [https://www.seldon.io/a-practical-guide-to-a-b-testing-in-mlops-with-kubernetes-and-seldon-core/](https://www.seldon.io/a-practical-guide-to-a-b-testing-in-mlops-with-kubernetes-and-seldon-core/)
3. Machine Learning Fundamentals: a/b testing tutorial \- DEV Community, дата последнего обращения: декабря 22, 2025, [https://dev.to/devopsfundamentals/machine-learning-fundamentals-ab-testing-tutorial-1hha](https://dev.to/devopsfundamentals/machine-learning-fundamentals-ab-testing-tutorial-1hha)
4. Demystifying A/B Testing, Canary Testing, and Blue-Green Deployments | by Mehak Adlakha | Women in Technology | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/womenintechnology/demystifying-a-b-testing-canary-testing-and-blue-green-deployments-dda325e62290](https://medium.com/womenintechnology/demystifying-a-b-testing-canary-testing-and-blue-green-deployments-dda325e62290)
5. Canary vs. A/B release strategy \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/62092338/canary-vs-a-b-release-strategy](https://stackoverflow.com/questions/62092338/canary-vs-a-b-release-strategy)
6. Model Deployment Strategies, дата последнего обращения: декабря 22, 2025, [https://neptune.ai/blog/model-deployment-strategies](https://neptune.ai/blog/model-deployment-strategies)
7. Shadow deployment vs. canary release of machine learning models | JFrog ML \- Qwak, дата последнего обращения: декабря 22, 2025, [https://www.qwak.com/post/shadow-deployment-vs-canary-release-of-machine-learning-models](https://www.qwak.com/post/shadow-deployment-vs-canary-release-of-machine-learning-models)
8. Traffic Splits Aren't True A/B Testing for Machine Learning Models | by Başak Tuğçe Eskili | Marvelous MLOps | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/marvelous-mlops/traffic-splits-arent-true-a-b-testing-for-machine-learning-models-62f77d10c993](https://medium.com/marvelous-mlops/traffic-splits-arent-true-a-b-testing-for-machine-learning-models-62f77d10c993)
9. Getting Started \- FastAPI Sessions, дата последнего обращения: декабря 22, 2025, [https://jordanisaacs.github.io/fastapi-sessions/guide/getting_started/](https://jordanisaacs.github.io/fastapi-sessions/guide/getting_started/)
10. Distilbert: A Smaller, Faster, and Distilled BERT \- Zilliz Learn, дата последнего обращения: декабря 22, 2025, [https://zilliz.com/learn/distilbert-distilled-version-of-bert](https://zilliz.com/learn/distilbert-distilled-version-of-bert)
11. Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT | by Victor Sanh | HuggingFace | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/huggingface/smaller-faster-cheaper-lighter-introducing-dilbert-a-distilled-version-of-bert-8cf3380435b5](https://medium.com/huggingface/smaller-faster-cheaper-lighter-introducing-dilbert-a-distilled-version-of-bert-8cf3380435b5)
12. Improving QA Efficiency with DistilBERT: Fine-Tuning and Inference on mobile Intel CPUs, дата последнего обращения: декабря 22, 2025, [https://arxiv.org/html/2505.22937](https://arxiv.org/html/2505.22937)
13. What differences in inference speed and memory usage might you observe between different Sentence Transformer architectures (for example, BERT-base vs DistilBERT vs RoBERTa-based models)? \- Milvus, дата последнего обращения: декабря 22, 2025, [https://milvus.io/ai-quick-reference/what-differences-in-inference-speed-and-memory-usage-might-you-observe-between-different-sentence-transformer-architectures-for-example-bertbase-vs-distilbert-vs-robertabased-models](https://milvus.io/ai-quick-reference/what-differences-in-inference-speed-and-memory-usage-might-you-observe-between-different-sentence-transformer-architectures-for-example-bertbase-vs-distilbert-vs-robertabased-models)
14. distilbert/distilbert-base-uncased-finetuned-sst-2-english \- Hugging Face, дата последнего обращения: декабря 22, 2025, [https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
15. Distilbert Base Uncased Finetuned Sst 2 English · Models \- Dataloop, дата последнего обращения: декабря 22, 2025, [https://dataloop.ai/library/model/distilbert_distilbert-base-uncased-finetuned-sst-2-english/](https://dataloop.ai/library/model/distilbert_distilbert-base-uncased-finetuned-sst-2-english/)
16. Everything you need to know about ALBERT, RoBERTa, and DistilBERT, дата последнего обращения: декабря 22, 2025, [https://towardsdatascience.com/everything-you-need-to-know-about-albert-roberta-and-distilbert-11a74334b2da/](https://towardsdatascience.com/everything-you-need-to-know-about-albert-roberta-and-distilbert-11a74334b2da/)
17. BERT inference throughput deathmatch: BERT, RoBERTa & DistilBERT | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@niksa.jakovljevic/bert-inference-throughput-deathmatch-eaaf505b3804](https://medium.com/@niksa.jakovljevic/bert-inference-throughput-deathmatch-eaaf505b3804)
18. cardiffnlp/twitter-roberta-base-sentiment-latest \- Hugging Face, дата последнего обращения: декабря 22, 2025, [https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
19. twitter-roberta-base-sentiment-latest | AI Model Details \- AIModels.fyi, дата последнего обращения: декабря 22, 2025, [https://www.aimodels.fyi/models/huggingFace/twitter-roberta-base-sentiment-latest-cardiffnlp](https://www.aimodels.fyi/models/huggingFace/twitter-roberta-base-sentiment-latest-cardiffnlp)
20. trallnag/prometheus-fastapi-instrumentator: Instrument your ... \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/trallnag/prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
21. Sentiment Analysis Using Pre-trained models and Transformer | by TANISH SHARMA, дата последнего обращения: декабря 22, 2025, [https://medium.com/@sharma.tanish096/sentiment-analysis-using-pre-trained-models-and-transformer-28e9b9486641](https://medium.com/@sharma.tanish096/sentiment-analysis-using-pre-trained-models-and-transformer-28e9b9486641)
22. Middleware \- FastAPI, дата последнего обращения: декабря 22, 2025, [https://fastapi.tiangolo.com/tutorial/middleware/](https://fastapi.tiangolo.com/tutorial/middleware/)
23. How to get weighted random choice in Python \- GeeksforGeeks, дата последнего обращения: декабря 22, 2025, [https://www.geeksforgeeks.org/python/how-to-get-weighted-random-choice-in-python/](https://www.geeksforgeeks.org/python/how-to-get-weighted-random-choice-in-python/)
24. How to Get Weighted Random Choice in Python? \- Tutorials Point, дата последнего обращения: декабря 22, 2025, [https://www.tutorialspoint.com/how-to-get-weighted-random-choice-in-python](https://www.tutorialspoint.com/how-to-get-weighted-random-choice-in-python)
25. Session in FastAPI \- python \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/68099561/session-in-fastapi](https://stackoverflow.com/questions/68099561/session-in-fastapi)
26. growthbook/growthbook-python: Powerful A/B Testing for Python Apps \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/growthbook/growthbook-python](https://github.com/growthbook/growthbook-python)
27. Twitter Roberta Base Sentiment Latest · Models \- Dataloop, дата последнего обращения: декабря 22, 2025, [https://dataloop.ai/library/model/cardiffnlp_twitter-roberta-base-sentiment-latest/](https://dataloop.ai/library/model/cardiffnlp_twitter-roberta-base-sentiment-latest/)
28. Sentiment Analysis with 10 Transformers \- Kaggle, дата последнего обращения: декабря 22, 2025, [https://www.kaggle.com/code/yaminh/sentiment-analysis-with-10-transformers](https://www.kaggle.com/code/yaminh/sentiment-analysis-with-10-transformers)
29. Labels | client_python \- Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.github.io/client_python/instrumenting/labels/](https://prometheus.github.io/client_python/instrumenting/labels/)
30. Prometheus-client : how to specify a list of labels \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/73785963/prometheus-client-how-to-specify-a-list-of-labels](https://stackoverflow.com/questions/73785963/prometheus-client-how-to-specify-a-list-of-labels)
31. Reduce metrics costs by filtering collected and forwarded metrics | Grafana Cloud documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana-cloud/cost-management-and-billing/analyze-costs/reduce-costs/metrics-costs/client-side-filtering/](https://grafana.com/docs/grafana-cloud/cost-management-and-billing/analyze-costs/reduce-costs/metrics-costs/client-side-filtering/)
32. Metric types \- Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/concepts/metric_types/](https://prometheus.io/docs/concepts/metric_types/)
33. Python Monitoring with Prometheus (Beginner's Guide) | Better Stack Community, дата последнего обращения: декабря 22, 2025, [https://betterstack.com/community/guides/monitoring/prometheus-python-metrics/](https://betterstack.com/community/guides/monitoring/prometheus-python-metrics/)
34. Getting Started: Monitoring a FastAPI App with Grafana and Prometheus \- A Step-by-Step Guide \- DEV Community, дата последнего обращения: декабря 22, 2025, [https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-grafana-and-prometheus-a-step-by-step-guide-3fbn](https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-grafana-and-prometheus-a-step-by-step-guide-3fbn)
35. Prometheus with Docker Compose: Guide & Examples \- Spacelift, дата последнего обращения: декабря 22, 2025, [https://spacelift.io/blog/prometheus-docker-compose](https://spacelift.io/blog/prometheus-docker-compose)
36. Configuration \- Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/prometheus/latest/configuration/configuration/](https://prometheus.io/docs/prometheus/latest/configuration/configuration/)
37. A Comprehensive Guide to Grouping and Functions in PromQL | by Suchita Sharma, дата последнего обращения: декабря 22, 2025, [https://medium.com/@suchitasharma1106/a-comprehensive-guide-to-grouping-and-functions-in-promql-cc3c438be320](https://medium.com/@suchitasharma1106/a-comprehensive-guide-to-grouping-and-functions-in-promql-cc3c438be320)
38. Query examples | Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/prometheus/latest/querying/examples/](https://prometheus.io/docs/prometheus/latest/querying/examples/)
39. Introduction to PromQL, the Prometheus query language | Grafana Labs, дата последнего обращения: декабря 22, 2025, [https://grafana.com/blog/2020/02/04/introduction-to-promql-the-prometheus-query-language/](https://grafana.com/blog/2020/02/04/introduction-to-promql-the-prometheus-query-language/)
40. Query functions \- Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/prometheus/latest/querying/functions/](https://prometheus.io/docs/prometheus/latest/querying/functions/)
41. Querying basics \- Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/prometheus/latest/querying/basics/](https://prometheus.io/docs/prometheus/latest/querying/basics/)
42. Transform data | Grafana documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/query-transform-data/transform-data/](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/query-transform-data/transform-data/)
43. Calculation types | Grafana documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/query-transform-data/calculation-types/](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/query-transform-data/calculation-types/)
44. Time series | Grafana documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/time-series/](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/time-series/)
45. Stat Panel doesn't like Last · Issue \#30007 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/grafana/grafana/issues/30007](https://github.com/grafana/grafana/issues/30007)
46. Time series transformed reduced row bar size is not adjustable \- Grafana, дата последнего обращения: декабря 22, 2025, [https://community.grafana.com/t/time-series-transformed-reduced-row-bar-size-is-not-adjustable/143505](https://community.grafana.com/t/time-series-transformed-reduced-row-bar-size-is-not-adjustable/143505)
