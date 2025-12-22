# **Пробуждение Облачного Голема: Всеобъемлющий аналитический отчет о развертывании контейнеризированных приложений в среде Google Cloud Run**

## **Аннотация**

В современной архитектуре облачных вычислений наблюдается фундаментальный сдвиг от управления инфраструктурой к управлению сервисами. Метафора «Облачного Голема» (Cloud Golem) точно описывает сущность контейнеризированного приложения в бессерверной среде (Serverless): это инертная материя (код и зависимости), которая пробуждается к жизни лишь под воздействием внешнего импульса (HTTP-запроса) и возвращается в небытие (масштабирование до нуля) после выполнения задачи. Настоящий отчет представляет собой исчерпывающее руководство по реализации «Квеста 28.2», целью которого является запуск Docker-контейнера на платформе Google Cloud Run. Документ охватывает весь жизненный цикл артефакта: от написания Dockerfile и оптимизации образов до настройки среды выполнения, интеграции искусственного интеллекта (AI/ML) и финансового планирования (FinOps). Особое внимание уделяется глубинному анализу архитектурных решений, позволяющих минимизировать холодный старт, обеспечить безопасность цепочки поставок ПО и оптимизировать затраты на высокопроизводительные вычисления.

## ---

**Часть I: Генезис Голема — Философия и Архитектура Бессерверных Контейнеров**

### **1.1 Эволюция абстракций: От железа к функции**

Исторический путь к Cloud Run лежит через последовательное повышение уровня абстракции. Если традиционные виртуальные машины (Google Compute Engine) требовали от инженеров управления операционной системой, патчами и сетевыми интерфейсами, то Kubernetes (GKE) абстрагировал оборудование, но оставил необходимость управления кластером. Cloud Run, построенный на базе открытого стандарта Knative, представляет собой вершину этой эволюции, предлагая модель «Контейнер как Услуга» (CaaS).1  
В контексте «Квеста 28.2» задача состоит не просто в запуске кода, а в создании _сервиса_, обладающего свойством эфемерности. В отличие от демонов, работающих на виртуальных машинах постоянно, контейнер Cloud Run существует только тогда, когда он нужен. Это порождает уникальную экономическую и техническую парадигму: разработчик платит только за время обработки запроса, с точностью до 100 миллисекунд.2

### **1.2 Природа Голема: Stateless и Эфемерность**

Ключевым требованием к архитектуре приложения в Cloud Run является отсутствие внутреннего состояния (Statelessness). Поскольку платформа автоматически масштабирует количество экземпляров контейнера от нуля до тысяч в зависимости от трафика, нет никакой гарантии, что последующий запрос от того же пользователя попадет в тот же контейнер. Любые данные, которые должны пережить цикл обработки одного запроса — будь то пользовательские сессии, загруженные файлы или результаты промежуточных вычислений, — должны быть немедленно вынесены во внешние, персистентные хранилища, такие как Cloud SQL, Firestore или Cloud Storage.1  
Этот принцип диктует «анатомию» нашего Голема: он должен быть легким на подъем (быстрый старт) и не иметь привязанностей к локальной файловой системе, которая в Cloud Run реализована как временный диск в оперативной памяти (in-memory tmpfs) и очищается при остановке контейнера.5

## ---

**Часть II: Ковка Сосуда — Принципы Контейнеризации Docker**

### **2.1 Dockerfile как генетический код**

Основой любого развертывания является Dockerfile. Это инструкция, по которой собирается образ контейнера. Для Cloud Run критически важно не просто собрать рабочий образ, но сделать его максимально компактным и безопасным. Исследование показывает, что размер образа напрямую влияет на время холодного старта (cold start latency), особенно если не используется технология стриминга образов.6

#### **2.1.1 Многоэтапная сборка (Multi-stage builds)**

Профессиональным стандартом является использование многоэтапной сборки. Этот метод позволяет отделить среду сборки (содержащую компиляторы, исходный код, инструменты тестирования) от среды выполнения (содержащей только бинарные файлы и минимальные зависимости).  
Рассмотрим пример для Python-приложения (например, FastAPI), часто используемого в AI-микросервисах:

Dockerfile

\# Этап 1: Сборка зависимостей  
FROM python:3.9-slim as builder  
WORKDIR /app  
COPY requirements.txt.  
RUN pip install \--user \--no-cache-dir \-r requirements.txt

\# Этап 2: Финальный образ  
FROM python:3.9-slim  
WORKDIR /app  
\# Копируем только установленные библиотеки из предыдущего этапа  
COPY \--from=builder /root/.local /root/.local  
COPY..  
ENV PATH=/root/.local/bin:$PATH  
\# Контракт Cloud Run: приложение должно слушать порт из переменной окружения PORT  
CMD \["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"\]

Такой подход предотвращает попадание в финальный образ лишних слоев кэша pip и системных утилит, что не только уменьшает размер артефакта, но и сокращает поверхность атаки.7

### **2.2 Проблема прав доступа и исполняемых файлов**

Частой ошибкой при создании контейнеров является некорректная обработка прав доступа к скриптам запуска (entrypoint.sh или start.sh). Если разработчик создает скрипт на Windows, файловая система NTFS не сохраняет бит исполнения (+x), который необходим Linux-контейнеру. В результате при запуске в Cloud Run возникает ошибка Permission denied.  
Исследования предлагают два решения:

1. **Явное изменение прав в Dockerfile:** Добавление инструкции RUN chmod \+x start.sh гарантирует, что файл будет исполняемым внутри образа, независимо от исходной ОС.8
2. **Git-индексация:** Использование команды git update-index \--chmod=+x start.sh позволяет сохранить метаданные о правах доступа непосредственно в репозитории Git, что автоматически применяет их при клонировании репозитория в CI/CD пайплайне.10

### **2.3 Выбор базового образа для AI-нагрузок**

Для задач машинного обучения (ML) выбор базового образа становится критическим. Стандартные образы python:slim могут не содержать необходимых системных библиотек (например, libGL для OpenCV или драйверов CUDA для PyTorch). Google и NVIDIA рекомендуют использовать специализированные Deep Learning Containers (DLC) или образы из NVIDIA Container Registry, которые предварительно оптимизированы и протестированы на совместимость с аппаратными ускорителями Cloud Run.6

## ---

**Часть III: Святилище Артефактов — Управление Реестром и Цепочка Поставок**

### **3.1 Переход от GCR к Artifact Registry**

Важнейшим аспектом современной экосистемы Google Cloud является миграция с устаревшего Container Registry (gcr.io) на Artifact Registry (pkg.dev). Исследовательские данные однозначно указывают: GCR находится в стадии устаревания (deprecated), и новые проекты должны использовать Artifact Registry.11  
Artifact Registry предлагает ряд преимуществ, критичных для промышленной эксплуатации:

- **Гранулярный контроль доступа:** Возможность управлять правами доступа на уровне отдельных репозиториев, а не всего проекта.
- **Региональность:** Хранение образов в том же регионе, где происходит развертывание (например, us-central1), снижает задержки сети и затраты на egress трафик при скачивании образа.13
- **Поддержка различных форматов:** Помимо Docker, поддерживаются Maven, npm, Python пакеты и Helm charts, что позволяет консолидировать все артефакты в одном месте.

### **3.2 Процедура аутентификации и загрузки**

Для успешного выполнения «Квеста» необходимо настроить аутентификацию Docker-клиента. Команда gcloud auth configure-docker обновляет конфигурационный файл Docker, добавляя в него credHelper, который использует учетные данные gcloud для доступа к реестрам.  
Специфика Artifact Registry требует указания регионального домена:

Bash

gcloud auth configure-docker us-central1-docker.pkg.dev

Это действие часто упускается новичками, что приводит к ошибкам unauthorized при попытке docker push. В автоматизированных пайплайнах (CI/CD) вместо персональных учетных записей следует использовать сервисные аккаунты с ролью Artifact Registry Writer.11

### **3.3 Структура наименования образов**

В Artifact Registry имя образа имеет строгую иерархию:  
LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE:TAG  
Например:  
us-central1-docker.pkg.dev/my-project/my-repo/cloud-golem:v1  
Соблюдение этой структуры обязательно. Попытка загрузить образ без предварительного создания репозитория (gcloud artifacts repositories create...) приведет к ошибке, в отличие от старого GCR, который создавал бакеты лениво.15

## ---

**Часть IV: Ритуал Пробуждения — Механика Развертывания**

### **4.1 Команда gcloud run deploy**

Акт развертывания осуществляется через CLI или консоль. Полная команда развертывания инкапсулирует конфигурацию сервиса:

Bash

gcloud run deploy cloud-golem \\  
 \--image us-central1-docker.pkg.dev/my-project/my-repo/cloud-golem:v1 \\  
 \--region us-central1 \\  
 \--platform managed \\  
 \--allow-unauthenticated \\  
 \--memory 4Gi \\  
 \--cpu 2 \\  
 \--timeout 300

Каждый флаг здесь имеет значение. \--allow-unauthenticated делает сервис публичным (что часто требуется для веб\-хуков или публичных API), в то время как удаление этого флага активирует IAM-защиту, требующую от вызывающей стороны наличия токена OIDC.17

### **4.2 Среда выполнения: Gen 1 против Gen 2**

Cloud Run предлагает два поколения среды выполнения.

- **Gen 1 (gVisor):** Обеспечивает высокую изоляцию за счет эмуляции системных вызовов. Идеально подходит для небольших микросервисов, быстро масштабируется, но имеет ограничения по совместимости с некоторыми системными вызовами и производительности файловой системы.
- **Gen 2 (microVM):** Использует полноценную виртуализацию Linux. Это поколение необходимо выбирать для нагрузок, требующих полной совместимости с Linux (например, некоторые инструменты мониторинга), использования сетевых файловых систем (NFS, GCS FUSE) или большого объема памяти (до 32 ГБ). Для AI-моделей и тяжелых вычислений Gen 2 является предпочтительным выбором.19

### **4.3 Управление ресурсами: CPU и Память**

Cloud Run позволяет гибко настраивать ресурсы. Важно понимать зависимость между CPU и памятью. Выделение памяти менее 512 МБ принудительно включает среду Gen 1\. Для приложений на Java или Python (особенно с ML-библиотеками) рекомендуется выделять минимум 1-2 ГБ памяти и 1 vCPU, чтобы избежать проблем с долгим запуском (Garbage Collection overhead).19  
Существует также концепция **CPU allocation**:

1. **CPU is only allocated during request processing:** Процессор доступен только когда сервис обрабатывает активный запрос. Как только ответ отправлен, CPU троттлится почти до нуля. Это экономит деньги, но убивает фоновые процессы.
2. **CPU is always allocated:** Процессор доступен все время жизни инстанса. Это необходимо для асинхронной обработки задач после ответа пользователю.21

## ---

**Часть V: Внедрение Интеллекта — AI и GPU в Serverless**

Наиболее захватывающим аспектом современного «Квеста» является возможность запуска тяжелых AI-моделей в бессерверной среде.

### **5.1 Революция GPU в Cloud Run**

Google Cloud Run внедрил поддержку GPU (NVIDIA L4), что кардинально меняет ландшафт инференса (вывода) моделей. Ранее для запуска LLM (Large Language Models) требовались выделенные VM с GPU, которые простаивали в отсутствие запросов, сжигая бюджет. Теперь Cloud Run позволяет масштабировать GPU-инстансы до нуля.22  
Это открывает возможность использования открытых моделей (Llama 3, Gemma, Mistral) в архитектуре, где оплата взимается только за секунды генерации токенов. Для реализации этого используются специализированные фреймворки, такие как **vLLM** или **TGI (Text Generation Inference)**, упакованные в контейнер.24

### **5.2 Проблема «Гравитации Данных» и Холодный Старт**

Главный вызов при запуске AI-контейнера — это размер модели. Веса современной LLM могут занимать десятки гигабайт. Загрузка такого объема данных при каждом старте контейнера может занимать минуты, что неприемлемо.

#### **Стратегия A: Baking (Антипаттерн)**

Включение весов модели внутрь Docker-образа (COPY./model /app).

- _Недостаток:_ Образ становится огромным. Время docker pull и распаковки увеличивает холодный старт до неприемлемых значений. Обновление модели требует полной пересборки образа.5

#### **Стратегия B: Streaming и GCS FUSE (Рекомендуется)**

Оптимальный подход заключается в хранении весов модели в Google Cloud Storage (GCS). При старте контейнера бакет GCS монтируется как локальная папка через драйвер Cloud Storage FUSE.  
Ключевое преимущество — стриминг. Приложению (например, vLLM) не нужно ждать полной загрузки файла весов. Оно начинает считывать необходимые тензоры по сети по мере инициализации. Это, в сочетании с высокой пропускной способностью внутренней сети Google Cloud, позволяет запускать инференс больших моделей за секунды, а не минуты.4

### **5.3 Container Image Streaming**

Для ситуаций, когда использование GCS FUSE невозможно, Google предлагает технологию **Image Streaming**. Она позволяет Cloud Run запускать контейнер, как только скачаны метаданные образа, а необходимые блоки данных подтягиваются по требованию. Это эффективно для больших базовых образов (например, PyTorch или TensorFlow), сокращая время старта в разы.26

## ---

**Часть VI: Нервная Система Голема — Web-сервер и Конфигурация**

### **6.1 Uvicorn vs. Gunicorn: Битва WSGI и ASGI**

Для Python-приложений (наиболее частый выбор для AI) критически важен выбор веб\-сервера. FastAPI, являясь асинхронным фреймворком, использует стандарт ASGI.

- **Uvicorn:** Высокопроизводительный ASGI-сервер. Отлично справляется с асинхронностью, но в «голом» виде является однопроцессным.
- **Gunicorn:** Проверенный временем WSGI-сервер, который умеет управлять пулом рабочих процессов (workers), перезапускать их при сбоях и распределять нагрузку.

Лучшая практика для Cloud Run:  
Хотя Gunicorn с воркерами Uvicorn (gunicorn \-k uvicorn.workers.UvicornWorker) является золотым стандартом для VM, в Cloud Run единицей масштабирования является сам контейнер. Если контейнеру выделен 1 vCPU, запуск нескольких воркеров Gunicorn лишь увеличит оверхед.  
Рекомендация:

- При 1 vCPU: Использовать чистый uvicorn или gunicorn с 1 воркером.
- При 2+ vCPU: Использовать gunicorn с количеством воркеров, рассчитанным по формуле (2 x vCPU) \+ 1 или vCPU, чтобы утилизировать многоядерность внутри одного инстанса.28

В таблице ниже приведено сравнение конфигураций:

| Характеристика            | Uvicorn (Bare)              | Gunicorn \+ Uvicorn Workers     | Рекомендация для Cloud Run |
| :------------------------ | :-------------------------- | :------------------------------ | :------------------------- |
| **Управление процессами** | Нет (один процесс)          | Да (Master \+ Workers)          | Gunicorn (для надежности)  |
| **Параллелизм**           | Асинхронный (Event Loop)    | Мультипроцессный \+ Асинхронный | Зависит от vCPU            |
| **Перезапуск при сбое**   | Нет (падает весь контейнер) | Да (перезапуск воркера)         | Gunicorn предпочтительнее  |
| **Сложность настройки**   | Низкая                      | Средняя                         | Средняя                    |

### **6.2 Переменные окружения и Секреты**

Приложение должно конфигурироваться через переменные окружения, следуя методологии 12-factor app. Порт приложения _никогда_ не должен быть захардкожен.  
Правильный паттерн инициализации (Python):

Python

import os  
port \= int(os.environ.get("PORT", 8080)) \# Значение по умолчанию 8080

Если Cloud Run передает переменную PORT, приложение обязано слушать именно его. Игнорирование этого требования приводит к ошибке Container failed to start and listen on the port.31  
Для чувствительных данных (ключи API, пароли БД) использование простых переменных окружения (--set-env-vars) небезопасно, так как они видны в консоли Google Cloud. Следует использовать интеграцию с **Secret Manager**, монтируя секреты как файлы или переменные среды непосредственно в момент запуска.33

## ---

**Часть VII: Борьба с Дремотой — Оптимизация Производительности и Холодный Старт**

### **7.1 Феномен Холодного Старта**

Когда сервис масштабируется до нуля, первый пришедший запрос инициирует создание нового инстанса. Это включает: выделение ресурсов, скачивание образа, старт контейнера, инициализацию рантайма (JVM, Python interpreter), загрузку приложения. Суммарно это может занять от 2 до 20 секунд.

### **7.2 Startup CPU Boost**

Технология CPU Boost позволяет временно перераспределить ресурсы процессора. В момент старта контейнеру выделяется значительно больше мощности CPU, чем указано в лимите. Как только приложение начинает слушать порт (проходит health check), буст отключается.  
Это критически важно для Java-приложений (ускорение старта Spring Boot на 50%) и тяжелых ML-моделей, где инициализация требует интенсивных вычислений.34

### **7.3 Min Instances**

Для сервисов, где задержка недопустима, используется параметр \--min-instances. Это заставляет Cloud Run держать указанное количество контейнеров в «горячем» состоянии (idle), готовыми к немедленной обработке запроса. Однако, это меняет модель ценообразования: за простаивающие «минимальные» инстансы взимается плата, пусть и по сниженному тарифу.36

## ---

**Часть VIII: Экономика Магии — FinOps и Ценообразование**

### **8.1 Модели биллинга**

Cloud Run предлагает гибкую систему оплаты, понимание которой необходимо для предотвращения неожиданных счетов.

1. **Request-based Billing (По запросам):**
   - Оплата только когда обрабатывается запрос.
   - CPU выделяется только на время запроса.
   - Идеально для API, веб\-хуков, инструментов с редким использованием.
   - Если сервис простаивает — счет равен $0.
2. **Instance-based Billing (За время жизни):**
   - Оплата за все время, пока контейнер запущен (даже если он idle, при использовании min-instances).
   - Позволяет выполнять фоновые задачи после ответа пользователю.
   - Рекомендуется для высоконагруженных сервисов с постоянным трафиком.21

### **8.2 Сравнительный анализ: Cloud Run vs Compute Engine**

Для задачи инференса AI выбор платформы определяет экономику проекта.  
**Сценарий:** Чат-бот для внутренней документации компании.

- _Нагрузка:_ Активен только в рабочее время (8 часов), простаивает ночью и в выходные. Редкие запросы (интермиттирующая нагрузка).
- _Compute Engine (GPU VM):_ Придется платить за VM 24/7 или настраивать сложные скрипты включения/выключения. Стоимость простоя GPU огромна.
- _Cloud Run:_ Оплата только за секунды генерации ответов. Экономия может достигать 90% по сравнению с постоянно включенной VM.2

**Таблица сравнения затрат (Гипотетический пример):**

| Параметр                    | Cloud Run (Request-based)           | Compute Engine (VM)         | GKE (Kubernetes)                  |
| :-------------------------- | :---------------------------------- | :-------------------------- | :-------------------------------- |
| **Оплата простоя**          | $0 (при min-instances=0)            | 100% стоимости              | 100% стоимости нод                |
| **Минимальная тарификация** | 100 мс                              | 1 минута                    | Зависит от нод                    |
| **Управление**              | Полностью управляемое               | Ручное (OS ops)             | Сложное (Cluster ops)             |
| **Сценарий победы**         | Спорадический трафик, Scale-to-zero | Постоянная, ровная нагрузка | Масштабные микросервисные системы |

### **8.3 Committed Use Discounts (CUD)**

Если проект переходит в стадию зрелости и нагрузка становится предсказуемой, Google предлагает скидки за обязательство использования (Committed Use Discounts). Покупка обязательства (например, на $100 в час) на 1 или 3 года дает скидку от 17% до 35%. Уникальность CUD в Google Cloud в том, что они **Flexible**: скидка распространяется и на Cloud Run, и на Compute Engine, и на GKE, позволяя менять архитектуру без потери скидки.2

## ---

**Часть IX: Диагностика и Устранение Неполадок — Полевое Руководство**

Даже при идеальной подготовке «Ритуал Пробуждения» может дать сбой. Анализ логов — ключ к решению.

### **9.1 Ошибка "Container failed to start"**

Самая распространенная и пугающая ошибка. Она означает, что процесс внутри контейнера либо упал, либо не смог начать отвечать на HTTP-запросы за отведенное время.

- **Диагностика:** Проверить логи stdout и stderr.
- **Причина 1:** Приложение слушает не тот порт. (Например, Flask по умолчанию 5000, а Cloud Run ждет 8080). _Решение:_ Использовать os.environ.get('PORT').
- **Причина 2:** Приложение слушает 127.0.0.1 (localhost). В контейнере это означает, что порт недоступен снаружи. _Решение:_ Слушать 0.0.0.0.31
- **Причина 3:** Тайм-аут. Приложение грузит модель в память 60 секунд, а тайм-аут проверки здоровья — 30 секунд. _Решение:_ Увеличить тайм-аут или использовать Startup CPU Boost.38

### **9.2 Ошибка "Memory Limit Exceeded" (OOM)**

Если контейнер потребляет больше памяти, чем выделено, он будет убит системой (OOM Kill).

- **Симптом:** В логах сообщение Memory limit exceeded или код выхода 137\. График памяти имеет форму пилы.
- **Нюанс:** Запись файлов в /tmp потребляет RAM\! Если ваше приложение скачивает 2 ГБ файл во временную папку, а контейнеру выделено 2 ГБ памяти, он упадет.
- **Решение:** Использовать стриминг данных без сохранения на диск или монтировать GCS бакеты.38

## ---

**Заключение**

Выполнение Квеста 28.2 — это не просто техническое упражнение, а входной билет в мир современной облачной инженерии. Мы рассмотрели, как превратить статичный Docker-образ в динамичный, масштабируемый и интеллектуальный сервис.  
Ключевые выводы для архитектора:

1. **Минимизируйте образы:** Используйте многоэтапные сборки и правильные базовые образы.
2. **Отделяйте данные от кода:** Модели и большие файлы должны жить в GCS и подключаться через FUSE, а не запекаться в образ.
3. **Управляйте состоянием:** Cloud Run — это царство Stateless. Любое состояние — в БД или Redis.
4. **Считайте деньги:** Понимайте разницу между оплатой за запрос и за инстанс, особенно при использовании GPU.

«Облачный Голем» теперь пробужден. Он готов служить, мгновенно реагируя на потребности бизнеса и исчезая, когда работа выполнена, оставляя после себя лишь чистый след в логах и минимальный счет в биллинге.

#### **Источники**

1. Google Cloud Run vs. Google Compute Engine Comparison \- SourceForge, дата последнего обращения: декабря 22, 2025, [https://sourceforge.net/software/compare/Google-Cloud-Run-vs-Google-Compute-Engine/](https://sourceforge.net/software/compare/Google-Cloud-Run-vs-Google-Compute-Engine/)
2. Cloud Run pricing | Google Cloud, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/run/pricing](https://cloud.google.com/run/pricing)
3. Google Cloud Run Pricing in 2025: A Comprehensive Guide \- Cloudchipr, дата последнего обращения: декабря 22, 2025, [https://cloudchipr.com/blog/cloud-run-pricing](https://cloudchipr.com/blog/cloud-run-pricing)
4. Configure Cloud Storage volume mounts for Cloud Run services, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts](https://docs.cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts)
5. Cloud Run / Docker loading large files for ML prediction \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/63847702/cloud-run-docker-loading-large-files-for-ml-prediction](https://stackoverflow.com/questions/63847702/cloud-run-docker-loading-large-files-for-ml-prediction)
6. Best practices: AI inference on Cloud Run services with GPUs, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices](https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices)
7. Building best practices \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/build/building/best-practices/](https://docs.docker.com/build/building/best-practices/)
8. Docker build with start.sh having below content does not execute, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/docker-build-with-start-sh-having-below-content-does-not-execute/136482](https://forums.docker.com/t/docker-build-with-start-sh-having-below-content-does-not-execute/136482)
9. chmod: changing permissions of 'myscript.sh' : Operation not permitted \- Server Fault, дата последнего обращения: декабря 22, 2025, [https://serverfault.com/questions/967580/chmod-changing-permissions-of-myscript-sh-operation-not-permitted](https://serverfault.com/questions/967580/chmod-changing-permissions-of-myscript-sh-operation-not-permitted)
10. chmod in Dockerfile does not permanently change permissions \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/61175552/chmod-in-dockerfile-does-not-permanently-change-permissions](https://stackoverflow.com/questions/61175552/chmod-in-dockerfile-does-not-permanently-change-permissions)
11. Quickstart: Store Docker container images in Artifact Registry | Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/artifact-registry/docs/docker/store-docker-container-images](https://docs.cloud.google.com/artifact-registry/docs/docker/store-docker-container-images)
12. Transition from Container Registry | Artifact Registry \- Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/artifact-registry/docs/transition/transition-from-gcr](https://docs.cloud.google.com/artifact-registry/docs/transition/transition-from-gcr)
13. Create a Google Artifact Docker Registry | by Prayag Sangode | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@prayag-sangode/create-a-docker-gcp-artifactory-registry-c271a467e574](https://medium.com/@prayag-sangode/create-a-docker-gcp-artifactory-registry-c271a467e574)
14. Configure authentication to Artifact Registry for Docker \- Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/artifact-registry/docs/docker/authentication](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication)
15. Create standard repositories | Artifact Registry \- Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/artifact-registry/docs/repositories/create-repos](https://docs.cloud.google.com/artifact-registry/docs/repositories/create-repos)
16. Changes for Docker | Artifact Registry \- Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/artifact-registry/docs/transition/changes-docker](https://docs.cloud.google.com/artifact-registry/docs/transition/changes-docker)
17. gcloud run deploy | Google Cloud SDK, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/sdk/gcloud/reference/run/deploy](https://cloud.google.com/sdk/gcloud/reference/run/deploy)
18. Cloud run: how to mitigate cold starts and how much that would cost? \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/googlecloud/comments/1ita39x/cloud_run_how_to_mitigate_cold_starts_and_how/](https://www.reddit.com/r/googlecloud/comments/1ita39x/cloud_run_how_to_mitigate_cold_starts_and_how/)
19. Configure memory limits for services | Cloud Run \- Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/run/docs/configuring/services/memory-limits](https://docs.cloud.google.com/run/docs/configuring/services/memory-limits)
20. Has gcloud run changed minimum memory limits? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/69085127/has-gcloud-run-changed-minimum-memory-limits](https://stackoverflow.com/questions/69085127/has-gcloud-run-changed-minimum-memory-limits)
21. Billing settings for services | Cloud Run \- Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/run/docs/configuring/billing-settings](https://docs.cloud.google.com/run/docs/configuring/billing-settings)
22. AI/ML orchestration on Cloud Run documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/run/docs/ai](https://docs.cloud.google.com/run/docs/ai)
23. How to deploy Llama 3.2-1B-Instruct model with Google Cloud Run, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/blog/products/ai-machine-learning/how-to-deploy-llama-3-2-1b-instruct-model-with-google-cloud-run](https://cloud.google.com/blog/products/ai-machine-learning/how-to-deploy-llama-3-2-1b-instruct-model-with-google-cloud-run)
24. Scale-to-Zero LLM Inference with vLLM, Cloud Run and Cloud Storage FUSE \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/google-cloud/scale-to-zero-llm-inference-with-vllm-cloud-run-and-cloud-storage-fuse-42c7e62f6ec6](https://medium.com/google-cloud/scale-to-zero-llm-inference-with-vllm-cloud-run-and-cloud-storage-fuse-42c7e62f6ec6)
25. Scalable AI starts with storage: Guide to model artifact strategies | Google Cloud Blog, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/blog/topics/developers-practitioners/scalable-ai-starts-with-storage-guide-to-model-artifact-strategies](https://cloud.google.com/blog/topics/developers-practitioners/scalable-ai-starts-with-storage-guide-to-model-artifact-strategies)
26. Use Image streaming to pull container images | Google Kubernetes Engine (GKE), дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/kubernetes-engine/docs/how-to/image-streaming](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/image-streaming)
27. Improving GKE container image streaming for faster app startup | Google Cloud Blog, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/blog/products/containers-kubernetes/improving-gke-container-image-streaming-for-faster-app-startup](https://cloud.google.com/blog/products/containers-kubernetes/improving-gke-container-image-streaming-for-faster-app-startup)
28. Advanced Performance Tuning for FastAPI on Google Cloud Run \- David Muraya, дата последнего обращения: декабря 22, 2025, [https://davidmuraya.com/blog/fastapi-performance-tuning-on-google-cloud-run/](https://davidmuraya.com/blog/fastapi-performance-tuning-on-google-cloud-run/)
29. FastAPI production deployment best practices \- Render, дата последнего обращения: декабря 22, 2025, [https://render.com/articles/fastapi-production-deployment-best-practices](https://render.com/articles/fastapi-production-deployment-best-practices)
30. Mastering Gunicorn and Uvicorn: The Right Way to Deploy FastAPI Applications \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@iklobato/mastering-gunicorn-and-uvicorn-the-right-way-to-deploy-fastapi-applications-aaa06849841e](https://medium.com/@iklobato/mastering-gunicorn-and-uvicorn-the-right-way-to-deploy-fastapi-applications-aaa06849841e)
31. Cloud Run deployment failing for FastAPI \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/78623517/cloud-run-deployment-failing-for-fastapi](https://stackoverflow.com/questions/78623517/cloud-run-deployment-failing-for-fastapi)
32. Container failed to start. Failed to start and then listen on the port defined by the PORT environment variable \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/55662222/container-failed-to-start-failed-to-start-and-then-listen-on-the-port-defined-b](https://stackoverflow.com/questions/55662222/container-failed-to-start-failed-to-start-and-then-listen-on-the-port-defined-b)
33. Still Packaging AI Models in Containers? Do This Instead on Cloud Run \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/google-cloud/still-packaging-ai-models-in-containers-do-this-instead-on-cloud-run-a0219e625bca](https://medium.com/google-cloud/still-packaging-ai-models-in-containers-do-this-instead-on-cloud-run-a0219e625bca)
34. Faster cold starts with startup CPU Boost | Google Cloud Blog, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/blog/products/serverless/announcing-startup-cpu-boost-for-cloud-run--cloud-functions](https://cloud.google.com/blog/products/serverless/announcing-startup-cpu-boost-for-cloud-run--cloud-functions)
35. Google Cloud Introduces Startup CPU Boost for Cloud Run and Cloud Functions 2nd Gen, дата последнего обращения: декабря 22, 2025, [https://www.infoq.com/news/2022/09/google-startup-cpu-boost/](https://www.infoq.com/news/2022/09/google-startup-cpu-boost/)
36. 3 Ways to optimize Cloud Run response times | Google Cloud Blog, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/blog/topics/developers-practitioners/3-ways-optimize-cloud-run-response-times](https://cloud.google.com/blog/topics/developers-practitioners/3-ways-optimize-cloud-run-response-times)
37. Set minimum instances for services | Cloud Run | Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/run/docs/configuring/min-instances](https://docs.cloud.google.com/run/docs/configuring/min-instances)
38. Troubleshoot Cloud Run issues | Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/run/docs/troubleshooting](https://docs.cloud.google.com/run/docs/troubleshooting)
39. Cloud run in-memory volume limit reached but no error was thrown \- Serverless Applications, дата последнего обращения: декабря 22, 2025, [https://discuss.google.dev/t/cloud-run-in-memory-volume-limit-reached-but-no-error-was-thrown/172660](https://discuss.google.dev/t/cloud-run-in-memory-volume-limit-reached-but-no-error-was-thrown/172660)
