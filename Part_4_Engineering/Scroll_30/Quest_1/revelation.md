# **Архитектурная валидация и стратегия реализации нагрузочного тестирования API: Глубокий анализ подхода «Квест 30.1»**

## **Аннотация**

В условиях современной разработки программного обеспечения, ориентированной на микросервисную архитектуру и непрерывную поставку (Continuous Delivery), обеспечение производительности API становится критически важной задачей, равнозначной функциональной корректности. Данный отчет представляет собой исчерпывающее исследование и валидацию подхода к нагрузочному тестированию, предложенного в рамках задачи «Квест 30.1: Проверка на Прочность». Документ детально рассматривает реализацию системы тестирования с использованием инструментария Locust, анализирует архитектурные решения по контейнеризации и оркестрации, а также определяет стратегии интеграции в CI/CD-конвейеры. Особое внимание уделяется сравнению механизмов генерации нагрузки, оптимизации Docker-образов, проблемам сетевого взаимодействия в виртуализированных средах и построению наблюдаемых (observable) систем анализа метрик. Отчет предназначен для архитекторов, DevOps-инженеров и специалистов по обеспечению качества (QA), стремящихся построить масштабируемую и воспроизводимую инфраструктуру тестирования производительности.

## ---

**Глава 1\. Теоретический базис и методология тестирования производительности**

### **1.1. Эволюция парадигм: от GUI к Code-as-Configuration**

Исторически нагрузочное тестирование ассоциировалось с тяжеловесными инструментами корпоративного класса, такими как HP LoadRunner, которые требовали использования проприетарных языков скриптов и сложных графических интерфейсов для моделирования сценариев. Однако с развитием практик DevOps и SRE (Site Reliability Engineering) произошел сдвиг парадигмы в сторону подхода «Code-as-Configuration» (код как конфигурация). В этом контексте инструмент Locust занимает уникальную нишу, предоставляя возможность описывать поведение пользователей (User Behavior) с использованием чистого Python-кода.1  
Этот подход фундаментально меняет процесс разработки тестов. Сценарии нагрузки перестают быть статичными XML-конфигурациями (как в JMeter) и становятся полноценными программными модулями, поддерживающими наследование, инкапсуляцию и повторное использование кода. Это позволяет применять к тестовым сценариям те же стандарты качества, что и к основному коду продукта: код-ревью, линтинг, версионирование в Git и модульное тестирование самой логики нагрузки.1

### **1.2. Моделирование нагрузки: Открытые и закрытые системы**

При проектировании нагрузочных тестов критически важно понимать различие между открытыми (Open Workload) и закрытыми (Closed Workload) моделями нагрузки. Locust по своей архитектуре реализует **закрытую модель**, где новые запросы генерируются только после завершения предыдущих (с учетом времени ожидания wait_time). Количество виртуальных пользователей фиксировано или изменяется по расписанию, но каждый пользователь работает в цикле: «отправил запрос \-\> дождался ответа \-\> подумал \-\> отправил следующий».3  
Это имеет глубокие последствия для валидации API. В закрытой модели, если тестируемая система (System Under Test — SUT) начинает деградировать и время отклика растет, частота отправки новых запросов (RPS — Requests Per Second) автоматически снижается, так как пользователи дольше ждут ответов. Это создает эффект «обратного давления» (backpressure), который может маскировать реальные проблемы масштабируемости, предотвращая полный отказ системы. Для сравнения, открытые модели (реализуемые инструментами вроде WRK или Vegeta) продолжают отправлять запросы с заданной частотой независимо от состояния сервера, что быстрее приводит к переполнению очередей. Понимание этого нюанса позволяет инженерам использовать Locust для реалистичной эмуляции поведения живых пользователей, но требует использования специфических техник (например, constant_pacing) для тестирования пропускной способности (Throughput).4

### **1.3. Закон Литтла в контексте API**

Фундаментальным уравнением для валидации результатов нагрузочного теста является Закон Литтла (Little's Law), который гласит:

$$L \= \\lambda \\times W$$

Где:

- $L$ — среднее количество запросов в системе (concurrency).
- $\\lambda$ — средняя скорость поступления запросов (throughput/RPS).
- $W$ — среднее время обработки запроса (latency).

Применительно к Locust, это уравнение позволяет планировать необходимые ресурсы. Если целевая нагрузка составляет 1000 RPS ($\\lambda$), а ожидаемое время отклика API — 200 мс ($W \= 0.2$ с), то для создания такой нагрузки необходимо поддерживать минимум $1000 \\times 0.2 \= 200$ одновременно активных запросов ($L$). Учитывая время на «размышление» пользователя (think time, $Z$), формула трансформируется для расчета количества виртуальных пользователей ($N$):

$$N \= \\lambda \\times (W \+ Z)$$

Использование этой формулы позволяет валидировать корректность настройки теста: если при 1000 пользователях и времени ожидания 1 секунда мы не получаем ожидаемые 500 RPS (при отклике 1с), это сигнализирует о наличии узкого места либо в генераторе нагрузки, либо в SUT.5

## ---

**Глава 2\. Архитектура реализации на Locust: Глубокий анализ компонентов**

### **2.1. Конкурентная модель: Greenlets против Threads**

Одной из ключевых особенностей Locust, обеспечивающей его высокую производительность, является использование событийной модели (event-based) на базе библиотеки gevent. В отличие от инструментов, использующих системные потоки (OS Threads) для каждого виртуального пользователя (например, классический JMeter), Locust использует «гринлеты» (greenlets) — легковесные корутины, управляемые в пользовательском пространстве (userspace).1  
Системные потоки имеют значительные накладные расходы на контекстное переключение (context switching) и потребление памяти (стек потока). Это ограничивает количество потоков на одной машине несколькими тысячами. Гринлеты же имеют минимальный оверхед, позволяя запускать десятки тысяч виртуальных пользователей на одном ядре процессора. Переключение между гринлетами происходит кооперативно во время операций ввода-вывода (I/O), что идеально подходит для нагрузочного тестирования HTTP API, где большую часть времени процесс ожидает ответа от сети.

### **2.2. Сравнительный анализ: HttpUser и FastHttpUser**

Выбор базового класса пользователя является одним из самых важных архитектурных решений при написании locustfile.

#### **2.2.1. HttpUser: Гибкость и совместимость**

Класс HttpUser базируется на популярной библиотеке python-requests. Это обеспечивает максимальное удобство разработки, так как API requests знаком большинству Python-разработчиков. Он поддерживает полную функциональность управления сессиями, cookies, SSL-сертификатами и прокси «из коробки».4 Однако requests является синхронной библиотекой, и, несмотря на манки-патчинг (monkey patching) со стороны gevent, она создает значительную нагрузку на CPU генератора из\-за создания тяжелых объектов Python для каждого запроса и ответа. Это ограничивает максимальный RPS с одного ядра (обычно до 1000-1500 RPS в зависимости от сложности логики).

#### **2.2.2. FastHttpUser: Производительность и эффективность**

Для высоконагруженных сценариев документация и лучшие практики настоятельно рекомендуют использование FastHttpUser. Этот класс использует geventhttpclient — клиент, написанный на C и оптимизированный для работы с gevent. Бенчмарки показывают, что FastHttpUser может обеспечивать производительность в 4-5 раз выше, чем HttpUser, при том же потреблении CPU.5

| Характеристика          | HttpUser (python-requests)                    | FastHttpUser (geventhttpclient)              |
| :---------------------- | :-------------------------------------------- | :------------------------------------------- |
| **Базовая библиотека**  | requests (Python)                             | geventhttpclient (C/C++)                     |
| **Потребление CPU**     | Высокое                                       | Низкое (оптимизировано)                      |
| **Макс. RPS (на ядро)** | \~1k \- 1.5k                                  | \~5k \- 10k+                                 |
| **Управление JSON**     | Требует ручного json.dumps или параметр json= | Автоматический парсинг и заголовки в .rest() |
| **Совместимость**       | Полная поддержка всех фич requests            | Ограниченная поддержка сложных auth-схем     |

Валидация подхода требует тестирования совместимости: FastHttpUser автоматически устанавливает заголовки Content-Type и Accept в application/json при использовании метода rest, что удобно для API, но может вызвать проблемы при тестировании endpoint'ов, ожидающих другие форматы данных. Кроме того, управление cookies в FastHttpUser реализовано иначе и может потребовать переписывания части логики авторизации.6

### **2.3. Структурирование сценариев: TaskSet и вложенность**

Реалистичные сценарии поведения редко бывают линейными. Пользователи выполняют действия с разной вероятностью и в разной последовательности. Механизм TaskSet позволяет моделировать сложные поведенческие паттерны, создавая иерархию задач.  
Например, класс UserBehavior может определять высокоуровневые действия (Логин, Просмотр каталога, Оформление заказа), каждое из которых реализовано как отдельный TaskSet. Использование декоратора @task(weight) позволяет задавать статистические веса: если задаче «Просмотр товара» присвоен вес 10, а задаче «Покупка» — вес 1, то покупка будет совершаться в среднем один раз на десять просмотров.6  
Критически важным аспектом реализации вложенных TaskSet является управление выходом. В отличие от функции, которая возвращает управление после завершения, TaskSet захватывает выполнение виртуального пользователя бесконечно, пока не будет вызван метод self.interrupt(). Отсутствие явного прерывания (например, в задаче stop) — распространенная ошибка, приводящая к тому, что пользователи «застревают» во вложенных меню и перестают выполнять другие действия, искажая профиль нагрузки.7

### **2.4. Валидация данных и Assertions**

Нагрузочное тестирование без проверки корректности ответов бессмысленно. Получение HTTP статуса 200 OK не гарантирует, что сервер вернул правильные данные — это может быть заглушка или пустой ответ.  
В Locust валидация реализуется через контекстный менеджер catch_response=True. Это позволяет перехватить объект ответа и пометить запрос как failure даже при статусе 200, если содержимое не соответствует ожиданиям.  
Пример реализации:

Python

@task  
def get_user_profile(self):  
 with self.client.get("/profile", catch_response=True) as response:  
 if response.status_code \== 200:  
 try:  
 data \= response.json()  
 if "user_id" not in data:  
 response.failure("Missing user_id in response")  
 except JSONDecodeError:  
 response.failure("Response is not valid JSON")  
 else:  
 response.failure(f"Status code: {response.status_code}")

Такой подход гарантирует, что в отчетах о тестировании будут отражены функциональные сбои, возникающие под нагрузкой, а не только сетевые ошибки.5

## ---

**Глава 3\. Контейнеризация: Оптимизация образов и управление зависимостями**

### **3.1. Дилемма выбора базового образа: Alpine vs Slim**

При создании Docker-образа для запуска Locust выбор операционной системы оказывает существенное влияние на процесс сборки (Build Time) и выполнения (Runtime). В сообществе Docker популярен дистрибутив Alpine Linux из\-за его малого размера. Однако для Python-приложений, особенно использующих C-расширения (такие как gevent, greenlet, msgpack, необходимые для Locust), использование Alpine часто является антипаттерном.9

#### **3.1.1. Проблема совместимости libc**

Alpine использует библиотеку musl libc, в то время как стандартный Python и большинство бинарных пакетов (wheels) в PyPI скомпилированы под glibc (стандарт GNU/Linux). Это означает, что при установке зависимостей pip не сможет использовать готовые бинарные файлы и начнет компилировать библиотеки из исходного кода. Это приводит к нескольким негативным последствиям:

1. **Увеличение времени сборки:** Компиляция криптографических библиотек или numpy может занимать десятки минут вместо нескольких секунд.11
2. **Необходимость установки toolchain:** Для компиляции в образ необходимо добавить gcc, make, musl-dev и другие заголовочные файлы, что значительно увеличивает размер итогового образа, нивелируя изначальное преимущество Alpine.11
3. **Риски производительности:** Существуют исследования, показывающие, что Python на musl может работать медленнее из\-за особенностей реализации аллокатора памяти malloc.11

#### **3.1.2. Рекомендация: Debian Slim**

Валидированным подходом для образов Locust является использование тега python:3.x-slim (базируется на Debian). Эти образы поддерживают стандарт manylinux wheels, что позволяет pip загружать и устанавливать прекомпилированные бинарные файлы мгновенно. Итоговый размер образа часто сопоставим с «раздутым» Alpine (после установки компиляторов), но процесс сборки становится на порядок быстрее и надежнее.11

### **3.2. Управление Entrypoint и конфигурацией запуска**

Официальный Docker-образ Locust претерпел изменения в версионировании, перейдя от использования shell-скриптов к прямому вызову исполняемого файла. При запуске контейнера часто возникает необходимость гибко управлять параметрами запуска: режимом (Master/Worker), хостом, количеством пользователей.  
Использование ENTRYPOINT \["locust"\] в Dockerfile делает контейнер исполняемым как бинарный файл. Все аргументы, переданные в CMD или в конце команды docker run, добавляются к locust. Это удобно, но требует понимания различий между ENTRYPOINT и CMD. Если требуется выполнить предварительную настройку среды (например, через скрипт), необходимо переопределить ENTRYPOINT.15  
Для CI/CD сред наиболее надежным способом конфигурации является передача параметров через переменные окружения или монтирование конфигурационного файла locust.conf, что позволяет избежать создания гигантских строк команд запуска в docker-compose.yml.17

## ---

**Глава 4\. Оркестрация среды тестирования с Docker Compose**

### **4.1. Сетевая топология и Service Discovery**

Docker Compose предоставляет удобный механизм для развертывания полного стека тестирования, включая SUT (System Under Test), базы данных, генератор нагрузки и систему мониторинга. Ключевым элементом здесь является встроенный DNS-сервер Docker, который обеспечивает обнаружение сервисов (Service Discovery).  
Контейнеры могут обращаться друг к другу по именам сервисов (например, http://api-service:8000). Однако при нагрузочном тестировании важно учитывать особенности резолвинга DNS. Стандартный механизм возвращает IP-адрес контейнера. Если SUT масштабирован до нескольких реплик (deploy: replicas: 3), Docker DNS может возвращать адреса в режиме Round-Robin, но многие HTTP-клиенты (включая requests и geventhttpclient) могут кэшировать DNS-ответы или устанавливать постоянное соединение (Keep-Alive) с одним IP-адресом. Это приводит к неравномерному распределению нагрузки между репликами приложения.18  
Для решения этой проблемы в высоконагруженных тестах рекомендуется использовать промежуточный балансировщик (например, Nginx или HAProxy) в составе Docker Compose, который будет распределять запросы между репликами SUT, или явно настраивать клиент Locust на отключение Keep-Alive, если целью является тест балансировки.20

### **4.2. Управление зависимостями запуска (Startup Order)**

Частой ошибкой при автоматизации тестов является "гонка" при запуске: контейнер с Locust запускается и начинает отправлять запросы раньше, чем приложение (SUT) успевает инициализировать подключение к базе данных и открыть порт. Это приводит к мгновенному падению теста с ошибками соединения.  
Директива depends_on в Docker Compose v3 контролирует только порядок запуска контейнеров, но не их готовность. Для корректной оркестрации необходимо использовать расширенный синтаксис с условиями здоровья (Healthchecks).  
Пример конфигурации:

YAML

depends_on:  
 sut-service:  
 condition: service_healthy

Для этого в Dockerfile сервиса SUT должна быть определена инструкция HEALTHCHECK, которая проверяет доступность основного эндпоинта. Только такой подход гарантирует, что нагрузка будет подана на готовую систему.21

### **4.3. Ресурсная изоляция и "Co-location Fallacy"**

Критическая уязвимость подхода локального нагрузочного тестирования заключается в размещении генератора нагрузки и тестируемой системы на одном физическом хосте (например, ноутбуке разработчика или одном CI-агенте). Нагрузочное тестирование — это ресурсоемкий процесс, потребляющий значительное количество CPU для генерации запросов, обработки ответов и сериализации данных.23  
Когда Locust и SUT делят одни и те же ресурсы процессора и памяти, возникает конкуренция (resource contention). Если Locust потребляет 60% CPU для генерации нагрузки, у приложения остается меньше ресурсов для обработки запросов. Это приводит к искажению результатов:

1. **Ложная деградация:** Время отклика растет не из\-за проблем в коде приложения, а из\-за того, что приложение "голодает" по CPU.
2. **Bottleneck генератора:** Locust не может создать целевую нагрузку, и тестирование проходит при заниженных показателях RPS.25

Для получения валидных результатов необходимо:

- Строго ограничивать ресурсы контейнеров через deploy.resources.limits в Docker Compose.
- В идеале — запускать генератор нагрузки на отдельной машине или узле кластера, изолированном от SUT по сети.23

## ---

**Глава 5\. Стратегии интеграции в CI/CD (Continuous Performance Testing)**

### **5.1. Автоматизация в Headless режиме**

Интеграция нагрузочного тестирования в конвейеры CI/CD требует полного отказа от взаимодействия с пользователем. Locust поддерживает режим «Headless» (без веб\-интерфейса), который активируется флагом \--headless. В этом режиме параметры теста должны быть переданы явно:

- \-u (users): целевое количество пользователей.
- \-r (spawn-rate): скорость добавления пользователей в секунду.
- \-t (run-time): время выполнения теста.26

Пример команды для CI:  
locust \-f locustfile.py \--headless \-u 100 \-r 5 \-t 10m \--host http://sut:8080

### **5.2. Критерии качества и Exit Codes**

Ключевой задачей CI является автоматическое принятие решения: прошел тест или нет. По умолчанию Locust завершается с кодом 0, даже если были ошибки HTTP 500\. Для изменения этого поведения можно использовать флаг \--exit-code-on-error, но этого часто недостаточно для проверки требований SLA (Service Level Agreement).27  
Более продвинутый подход — использование скриптов-оберток или встроенных возможностей Locust для анализа статистики после теста. Например, можно задать пороговые значения (Thresholds):

- Среднее время отклика \< 200 мс.
- 99-й перцентиль \< 500 мс.
- Процент ошибок \< 0.1%.

Если эти условия нарушены, процесс должен завершиться с ненулевым кодом, что приведет к остановке пайплайна (Fail Fast). Это реализуется через расширение класса Environment и обработку событий quitting.28

### **5.3. Эфемерные окружения (Ephemeral Environments)**

Нагрузочное тестирование на общей staging-среде часто дает нестабильные результаты из\-за "шумных соседей" (других тестов или разработчиков). Валидированным решением является использование эфемерных окружений — временных, изолированных копий инфраструктуры, создаваемых специально для одного прогона тестов.29  
Docker Compose позволяет реализовать это в рамках одного пайплайна:

1. Сборка образов.
2. Поднятие чистого окружения (SUT \+ DB) с помощью docker compose up.
3. Наполнение базы тестовыми данными (seeding).
4. Запуск контейнера с Locust против этого окружения.
5. Сбор отчетов и уничтожение окружения (docker compose down).

Этот подход (Shift-Left Performance Testing) позволяет выявлять регрессии производительности на ранних стадиях, тестируя каждую ветку (Feature Branch) изолированно.29

## ---

**Глава 6\. Observability: Мониторинг и анализ результатов**

### **6.1. Ограничения стандартной отчетности**

Locust генерирует HTML-отчеты и CSV-файлы, которые полезны для пост-анализа, но не дают полной картины динамики. Они показывают агрегированные значения, скрывая кратковременные всплески задержек (micro-bursts) и корреляцию с системными метриками. Для профессионального анализа необходимы временные ряды (Time Series Data).32

### **6.2. Интеграция с Prometheus и Grafana**

Стандартом де\-факто для мониторинга в среде Docker является стек Prometheus \+ Grafana. Поскольку Locust не имеет встроенной поддержки формата Prometheus, используется промежуточный компонент — Locust Exporter (например, containersol/locust_exporter).  
Архитектура мониторинга:

1. **Locust Master** открывает порт 8089 для API статистики.
2. **Locust Exporter** опрашивает API Locust и преобразует JSON в формат метрик Prometheus (locust_requests_current_rps, locust_user_count и т.д.).33
3. **Prometheus** периодически (scrape interval, например, 5с) забирает метрики с Exporter.
4. **Grafana** визуализирует данные, используя готовые дашборды (например, ID 11985, 12081).32

### **6.3. Корреляционный анализ**

Главная ценность такой интеграции — возможность наложить графики производительности приложения (RPS, Latency) на графики потребления ресурсов инфраструктуры (CPU, Memory, Disk I/O контейнеров SUT), собираемые cAdvisor или node_exporter.36  
Это позволяет ответить на вопросы причинно-следственной связи:

- "Вызван ли рост времени отклика исчерпанием пула соединений в базе данных?"
- "Связан ли пик ошибок с троттлингом CPU контейнера приложения?"  
  Без единой системы мониторинга такой анализ требует ручного сопоставления логов и графиков из разных источников, что неэффективно и чревато ошибками.

## ---

**Глава 7\. Распределенное тестирование и масштабирование**

### **7.1. Режим Master-Worker**

Один процесс Locust (даже с FastHttpUser) ограничен ресурсами одного процессорного ядра (из-за Python GIL). Для генерации нагрузок, превышающих возможности одного ядра (обычно \> 5000 RPS), Locust поддерживает распределенный режим.  
Архитектура состоит из:

- **Одного Master-узла:** Не генерирует нагрузку, но управляет тестом, агрегирует статистику и предоставляет веб\-интерфейс.
- **Множества Worker-узлов:** Подключаются к Master-узлу и выполняют генерацию запросов.1

В Docker Compose это реализуется запуском нескольких реплик сервиса worker, которые передают адрес мастера через аргумент \--master-host.

YAML

command: \-f /mnt/locust/locustfile.py \--worker \--master-host locust-master

### **7.2. Проблемы масштабирования**

При масштабировании воркеров важно учитывать пропускную способность сети самого генератора. Тысячи открытых соединений могут исчерпать лимит файловых дескрипторов (ulimits) или диапазон эфемерных портов на хосте. Тюнинг ядра Linux (параметры net.ipv4.tcp_tw_reuse, fs.file-max) является обязательным этапом подготовки инфраструктуры для высоконагруженных тестов.25

## ---

**Заключение**

Валидация подхода «Квест 30.1» демонстрирует, что использование Locust в сочетании с Docker и современными практиками CI/CD является мощным и гибким решением для тестирования производительности API. Однако успешная реализация требует выхода за рамки простого написания скриптов.  
Ключевые факторы успеха включают:

1. **Оптимизацию кода:** Использование FastHttpUser и правильное моделирование поведения пользователей через TaskSet.
2. **Инженерную культуру инфраструктуры:** Выбор правильных базовых образов (Debian Slim), настройка лимитов ресурсов и изоляция тестовой среды.
3. **Автоматизацию:** Внедрение жестких критериев качества (SLA) в CI-пайплайны.
4. **Наблюдаемость:** Построение замкнутого цикла мониторинга с Prometheus и Grafana для глубокого анализа инцидентов.

Реализация описанных в отчете рекомендаций позволяет трансформировать нагрузочное тестирование из рутинной проверки перед релизом в непрерывный процесс обеспечения надежности и масштабируемости программного продукта.

### ---

**Приложение: Рекомендуемая конфигурация docker-compose.yml**

Ниже представлен консолидированный пример конфигурации, учитывающий описанные лучшие практики (разделение сетей, healthcheck, мониторинг).

YAML

version: '3.9'

services:  
 \# \--- System Under Test (SUT) Section \---  
 \# Пример целевого сервиса. В реальности здесь будет ваше API и БД.  
 sut_api:  
 image: hashicorp/http-echo  
 command:  
 ports:  
 \- "5678:5678"  
 deploy:  
 resources:  
 limits:  
 cpus: '0.5'  
 memory: 256M  
 healthcheck:  
 test:  
 interval: 5s  
 timeout: 5s  
 retries: 5  
 networks:  
 \- app-net

\# \--- Load Testing Infrastructure Section \---

locust-master:  
 image: locustio/locust  
 ports:  
 \- "8089:8089" \# Web UI  
 volumes:  
 \-./locust:/mnt/locust  
 command: \-f /mnt/locust/locustfile.py \--master  
 networks:  
 \- app-net  
 \- monitor-net

locust-worker:  
 image: locustio/locust  
 volumes:  
 \-./locust:/mnt/locust  
 command: \-f /mnt/locust/locustfile.py \--worker \--master-host locust-master  
 depends_on:  
 locust-master:  
 condition: service_started  
 sut_api:  
 condition: service_healthy \# Ждем готовности SUT  
 deploy:  
 replicas: 2  
 resources:  
 limits:  
 cpus: '1.0'  
 memory: 512M  
 networks:  
 \- app-net

\# \--- Observability Stack Section \---

locust-exporter:  
 image: containersol/locust_exporter  
 environment:  
 \- LOCUST_EXPORTER_URI=http://locust-master:8089  
 ports:  
 \- "9646:9646" \# Prometheus metrics  
 depends_on:  
 \- locust-master  
 networks:  
 \- monitor-net

prometheus:  
 image: prom/prometheus  
 volumes:  
 \-./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml  
 ports:  
 \- "9090:9090"  
 networks:  
 \- monitor-net

grafana:  
 image: grafana/grafana  
 ports:  
 \- "3000:3000"  
 environment:  
 \- GF_SECURITY_ADMIN_PASSWORD=admin  
 depends_on:  
 \- prometheus  
 networks:  
 \- monitor-net

networks:  
 app-net:  
 driver: bridge  
 monitor-net:  
 driver: bridge

#### **Источники**

1. locustio/locust: Write scalable load tests in plain Python \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/locustio/locust](https://github.com/locustio/locust)
2. Locust Documentation — Locust 2.42.6 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.locust.io/](https://docs.locust.io/)
3. Quick start — Locust 2.0.0b4 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.locust.io/en/2.0.0/quickstart.html](https://docs.locust.io/en/2.0.0/quickstart.html)
4. Writing a locustfile — Locust 2.42.6 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.locust.io/en/stable/writing-a-locustfile.html](https://docs.locust.io/en/stable/writing-a-locustfile.html)
5. Increase performance with a faster HTTP client — Locust 2.42.6 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.locust.io/en/stable/increase-performance.html](https://docs.locust.io/en/stable/increase-performance.html)
6. API — Locust 2.42.6 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.locust.io/en/stable/api.html](https://docs.locust.io/en/stable/api.html)
7. TaskSet class — Locust 2.42.6 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.locust.io/en/stable/tasksets.html](https://docs.locust.io/en/stable/tasksets.html)
8. Load Testing with Locust: A High-Performance, Scalable Tool for Python \- Better Stack, дата последнего обращения: декабря 22, 2025, [https://betterstack.com/community/guides/testing/locust-explained/](https://betterstack.com/community/guides/testing/locust-explained/)
9. Docker Best Practices for Python Developers \- TestDriven.io, дата последнего обращения: декабря 22, 2025, [https://testdriven.io/blog/docker-best-practices/](https://testdriven.io/blog/docker-best-practices/)
10. Differences Between Standard Docker Images and Alpine \\ Slim Versions, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/differences-between-standard-docker-images-and-alpine-slim-versions/134973](https://forums.docker.com/t/differences-between-standard-docker-images-and-alpine-slim-versions/134973)
11. Alpine vs python-slim for deploying python data science stack? : r/docker \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/docker/comments/g5hb93/alpine_vs_pythonslim_for_deploying_python_data/](https://www.reddit.com/r/docker/comments/g5hb93/alpine_vs_pythonslim_for_deploying_python_data/)
12. Using Alpine can make Python Docker builds 50× slower, дата последнего обращения: декабря 22, 2025, [https://pythonspeed.com/articles/alpine-docker-python/](https://pythonspeed.com/articles/alpine-docker-python/)
13. The best Docker base image for your Python application (May 2024), дата последнего обращения: декабря 22, 2025, [https://pythonspeed.com/articles/base-image-python-docker-images/](https://pythonspeed.com/articles/base-image-python-docker-images/)
14. Docker images \- types. Slim vs slim-stretch vs stretch vs alpine, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/54954187/docker-images-types-slim-vs-slim-stretch-vs-stretch-vs-alpine](https://stackoverflow.com/questions/54954187/docker-images-types-slim-vs-slim-stretch-vs-stretch-vs-alpine)
15. Docker run override entrypoint with shell script which accepts arguments \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/41694329/docker-run-override-entrypoint-with-shell-script-which-accepts-arguments](https://stackoverflow.com/questions/41694329/docker-run-override-entrypoint-with-shell-script-which-accepts-arguments)
16. How to properly override the ENTRYPOINT using docker run | by Adrian Oprea | Medium, дата последнего обращения: декабря 22, 2025, [https://oprearocks.medium.com/how-to-properly-override-the-entrypoint-using-docker-run-2e081e5feb9d](https://oprearocks.medium.com/how-to-properly-override-the-entrypoint-using-docker-run-2e081e5feb9d)
17. Docker image should not require TARGET_URL · Issue \#1247 · locustio/locust \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/locustio/locust/issues/1247](https://github.com/locustio/locust/issues/1247)
18. Networking | Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/how-tos/networking/](https://docs.docker.com/compose/how-tos/networking/)
19. Precedence of DNS entry vs. Compose service name \- Docker Community Forums, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/precedence-of-dns-entry-vs-compose-service-name/120967](https://forums.docker.com/t/precedence-of-dns-entry-vs-compose-service-name/120967)
20. How does service discovery work with modern docker/docker-compose? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/37683508/how-does-service-discovery-work-with-modern-docker-docker-compose](https://stackoverflow.com/questions/37683508/how-does-service-discovery-work-with-modern-docker-docker-compose)
21. Control startup and shutdown order with Compose \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/how-tos/startup-order/](https://docs.docker.com/compose/how-tos/startup-order/)
22. Understand The depends_on Field In Docker Compose \- Warp, дата последнего обращения: декабря 22, 2025, [https://www.warp.dev/terminus/docker-compose-depends-on](https://www.warp.dev/terminus/docker-compose-depends-on)
23. Load testing should be done locally or remotely? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/54882301/load-testing-should-be-done-locally-or-remotely](https://stackoverflow.com/questions/54882301/load-testing-should-be-done-locally-or-remotely)
24. Load Testing: An Unorthodox Guide \- Marco Behler, дата последнего обращения: декабря 22, 2025, [https://www.marcobehler.com/guides/load-testing](https://www.marcobehler.com/guides/load-testing)
25. How to to Detect Overloaded Load Generators in Load Testing \- Radview, дата последнего обращения: декабря 22, 2025, [https://www.radview.com/blog/how-to-to-detect-overloaded-load-generators-in-load-testing/](https://www.radview.com/blog/how-to-to-detect-overloaded-load-generators-in-load-testing/)
26. Load Testing AI APIs with Locust: A Practical Guide for Data & MLOps Engineers \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@gema.correa/load-testing-ai-apis-with-locust-a-practical-guide-for-data-mlops-engineers-9b9ac18f690c](https://medium.com/@gema.correa/load-testing-ai-apis-with-locust-a-practical-guide-for-data-mlops-engineers-9b9ac18f690c)
27. Running without the web UI — Locust 2.42.6 documentation, дата последнего обращения: декабря 22, 2025, [https://docs.locust.io/en/stable/running-without-web-ui.html](https://docs.locust.io/en/stable/running-without-web-ui.html)
28. What is the cleanest to completely exit a locust process if test prerequisites fail?, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/70952865/what-is-the-cleanest-to-completely-exit-a-locust-process-if-test-prerequisites-f](https://stackoverflow.com/questions/70952865/what-is-the-cleanest-to-completely-exit-a-locust-process-if-test-prerequisites-f)
29. Ephemeral Environments Explained: Benefits, Tools, and How to Get Started? \- Qovery, дата последнего обращения: декабря 22, 2025, [https://www.qovery.com/blog/ephemeral-environments](https://www.qovery.com/blog/ephemeral-environments)
30. Ephemeral Environments in DevOps: Boost Testing & Efficiency | Atmosly \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/atmosly/ephemeral-environments-in-devops-how-temporary-environments-improve-testing-5b029f02b890](https://medium.com/atmosly/ephemeral-environments-in-devops-how-temporary-environments-improve-testing-5b029f02b890)
31. Do you use ephemeral/preview environments? (Asking as a docker employee) \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/docker/comments/163ugkg/do_you_use_ephemeralpreview_environments_asking/](https://www.reddit.com/r/docker/comments/163ugkg/do_you_use_ephemeralpreview_environments_asking/)
32. How to Move Metrics from Locust.io to Grafana via Prometheus \- Container Solutions, дата последнего обращения: декабря 22, 2025, [https://blog.container-solutions.com/how-to-move-metrics-from-locust.io-to-grafana-via-prometheus](https://blog.container-solutions.com/how-to-move-metrics-from-locust.io-to-grafana-via-prometheus)
33. How to send locust metrics to prometheus using locust exporter? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/60774808/how-to-send-locust-metrics-to-prometheus-using-locust-exporter](https://stackoverflow.com/questions/60774808/how-to-send-locust-metrics-to-prometheus-using-locust-exporter)
34. ContainerSolutions/locust_exporter: A Locust metrics exporter for Prometheus \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/ContainerSolutions/locust_exporter](https://github.com/ContainerSolutions/locust_exporter)
35. Locust Prometheus Monitoring Modern | Grafana Labs, дата последнего обращения: декабря 22, 2025, [https://grafana.com/grafana/dashboards/20462-locust-prometheus-monitoring-modern-2025/](https://grafana.com/grafana/dashboards/20462-locust-prometheus-monitoring-modern-2025/)
36. Monitoring a Linux host with Prometheus, Node Exporter, and Docker Compose \- Grafana, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-prometheus/prometheus-config-examples/docker-compose-linux/](https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-prometheus/prometheus-config-examples/docker-compose-linux/)
37. Feed Prometheus with Locust: performance tests as a metrics' source \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/swlh/feed-prometheus-with-locust-performance-tests-as-a-metrics-source-d8d2bfec918c](https://medium.com/swlh/feed-prometheus-with-locust-performance-tests-as-a-metrics-source-d8d2bfec918c)
