# **Архитектура «Сторожевая Башня»: Глубокий анализ внедрения стека наблюдаемости для приложений FastAPI с использованием Prometheus и Grafana в среде Docker Compose**

## **1\. Введение: Парадигма современной наблюдаемости**

В эпоху микросервисной архитектуры и контейнеризации традиционные методы мониторинга, опирающиеся исключительно на логирование событий, утратили свою эффективность. Понятие «наблюдаемость» (observability) выходит за рамки простого отслеживания статуса «up/down». Оно подразумевает создание системы, позволяющей внешнему наблюдателю понять внутреннее состояние приложения, основываясь исключительно на его выходных данных. Проект «Сторожевая Башня» (Watchtower), рассматриваемый в данном отчете, представляет собой эталонную реализацию такой системы для асинхронных Python-приложений (FastAPI), оркестрируемых через Docker Compose.  
Центральным элементом этой архитектуры является переход от реактивного анализа логов к проактивному анализу временных рядов (Time Series). Метрики, в отличие от логов, представляют собой легковесные, агрегируемые числовые данные, которые позволяют математически моделировать поведение системы. Использование связки Prometheus (сбор и хранение) и Grafana (визуализация) стало де\-факто индустриальным стандартом благодаря их интеграции с экосистемой Cloud Native Computing Foundation (CNCF).1  
Цель данного отчета — предоставить исчерпывающее техническое обоснование и руководство по реализации стека мониторинга. Мы детально рассмотрим механизмы взаимодействия контейнеров в сети Docker, специфику инструментирования асинхронного кода Python, математические модели агрегации метрик в PromQL и принципы визуализации «Золотых сигналов» SRE (Site Reliability Engineering).

## **2\. Архитектурный фундамент и теория сбора данных**

### **2.1 Модель Pull против Push: Фундаментальное различие**

При проектировании системы мониторинга одним из первых архитектурных решений является выбор между моделью Pull (вытягивание) и Push (отправка). Prometheus, в отличие от многих старых систем (например, Graphite или InfluxDB), использует модель Pull.3 Это решение имеет критические последствия для конфигурации сети в Docker Compose.  
В **Push-модели** каждое приложение должно знать адрес сервера мониторинга и активно отправлять ему данные. Это создает жесткую связность: изменение адреса сервера мониторинга требует переконфигурации всех сервисов. Кроме того, при высокой нагрузке приложение может быть заблокировано сетевыми задержками при отправке метрик, или же оно должно реализовывать сложную логику буферизации, что усложняет код сервиса.  
В **Pull-модели**, реализуемой Prometheus, ответственность инвертирована. Приложение FastAPI пассивно; оно лишь обновляет внутренние счетчики и экспонирует их на HTTP-эндпоинте (обычно /metrics). Сервер Prometheus самостоятельно опрашивает (скрейпит) этот эндпоинт с заданным интервалом.3  
**Преимущества Pull-модели для «Сторожевой Башни»:**

- **Децентрализация конфигурации:** Сервису FastAPI безразлично, кто и когда собирает метрики. Это упрощает масштабирование.
- **Контроль нагрузки:** Prometheus сам регулирует частоту опроса. Если сервер мониторинга перегружен, он просто замедляет сбор данных, не влияя на производительность обслуживаемого приложения.5
- **Обнаружение сбоев:** В Push-модели «молчание» сервиса может означать как отсутствие событий, так и падение сервиса. В Pull-модели, если Prometheus не может подключиться к таргету, он немедленно фиксирует это как событие up \= 0\.6

### **2.2 Сетевая топология Docker Compose и Service Discovery**

Для корректной работы Pull-модели Prometheus должен иметь сетевой доступ к контейнеру FastAPI. В среде Docker Compose это обеспечивается созданием изолированных сетевых пространств имен (namespaces) и использованием встроенного DNS-сервера Docker.8  
Когда мы определяем сервисы в docker-compose.yml, Docker автоматически создает сеть типа bridge. Каждый контейнер получает IP-адрес в этой подсети. Критически важно понимать механизм разрешения имен (DNS Resolution) внутри этой сети. Контейнеры могут обращаться друг к другу по имени сервиса, указанному в конфигурации Compose. Например, сервис с именем fastapi_app будет доступен для контейнера prometheus по хостнейму fastapi_app.9

| Компонент       | Роль в сети            | Механизм доступа                                                                           |
| :-------------- | :--------------------- | :----------------------------------------------------------------------------------------- |
| **FastAPI App** | Target (Цель)          | Слушает порт (напр., 8000\) на интерфейсе 0.0.0.0 внутри контейнера. Экспонирует /metrics. |
| **Prometheus**  | Scraper (Сборщик)      | Периодически выполняет HTTP GET запрос к http://fastapi_app:8000/metrics.                  |
| **Grafana**     | Viewer (Визуализатор)  | Выполняет HTTP запросы к API Prometheus (http://prometheus:9090) для построения графиков.  |
| **Docker DNS**  | Resolver (Разрешитель) | Преобразует имя fastapi_app в динамический IP-адрес контейнера (напр., 172.18.0.3).        |

Распространенной ошибкой является попытка использовать localhost внутри конфигурации Prometheus для обращения к приложению. В контексте контейнера localhost ссылается на сам контейнер Prometheus, а не на хост-машину или соседний контейнер. Использование правильных DNS-имен является обязательным требованием для реализации квеста.11

## **3\. Инструментирование приложения FastAPI**

Инструментирование — это процесс внедрения кода для сбора телеметрии. Для FastAPI, который работает на базе асинхронного сервера (Uvicorn/Starlette), использование блокирующих синхронных библиотек может привести к деградации производительности. Поэтому стандартом де\-факто является использование специализированной библиотеки prometheus-fastapi-instrumentator.

### **3.1 Реализация Middleware**

Библиотека prometheus-fastapi-instrumentator работает как Middleware, перехватывая каждый входящий запрос до того, как он достигнет бизнес-логики, и каждый исходящий ответ.13 Это позволяет автоматически собирать базовые метрики без изменения кода самих эндпоинтов.  
Типовая реализация «Fast Track» выглядит следующим образом 15:

Python

from fastapi import FastAPI  
from prometheus_fastapi_instrumentator import Instrumentator

app \= FastAPI()

\# Инициализация инструментатора  
instrumentator \= Instrumentator(  
 should_group_status_codes=True, \# Группировка 2xx, 4xx, 5xx  
 should_ignore_untemplated=True, \# Игнорирование запросов к несуществующим путям  
 should_instrument_requests_inprogress=True, \# Отслеживание активных запросов  
 inprogress_name="fastapi_requests_inprogress",  
 inprogress_labels=True,  
)

\# Активация и экспозиция эндпоинта  
instrumentator.instrument(app).expose(app, endpoint="/metrics")

**Анализ параметров:**

- should_group_status_codes: Позволяет уменьшить кардинальность метрик. Вместо хранения отдельных временных рядов для 200, 201, 204, они агрегируются в 2xx. Это критично для производительности базы данных Prometheus (TSDB).15
- should_ignore_untemplated: Защищает от атаки на кардинальность (Cardinality Explosion). Если злоумышленник начнет сканировать случайные URL (/api/random-1, /api/random-2), без этой настройки Prometheus создаст новый временной ряд для каждого URL, что приведет к исчерпанию оперативной памяти (OOM).16

### **3.2 Кастомные метрики и бизнес-логика**

Автоматическое инструментирование покрывает технические метрики (HTTP-трафик, задержки), но «Сторожевая Башня» требует также наблюдения за бизнес-показателями. Для этого используются примитивы библиотеки prometheus_client.  
Критически важным аспектом при создании кастомных метрик является паттерн **Singleton**. Метрики должны объявляться в глобальной области видимости модуля, а не внутри функции-обработчика запроса. Если создать объект Counter внутри функции def, он будет пересоздаваться при каждом запросе, сбрасывая значение в ноль, что сделает мониторинг бессмысленным.13  
**Пример реализации бизнес-счетчика:**

Python

from prometheus_client import Counter

\# Объявление метрики на уровне модуля  
DOCUMENT_PROCESSING_ERRORS \= Counter(  
 "app_document_processing_errors_total",  
 "Total number of validation errors during document processing",  
 \["doc_type", "error_reason"\] \# Лейблы для детализации  
)

@app.post("/upload")  
async def upload_document(doc_type: str):  
 if doc_type not in \["pdf", "docx"\]:  
 \# Инкремент счетчика с конкретными лейблами  
 DOCUMENT_PROCESSING_ERRORS.labels(  
 doc_type=doc_type,  
 error_reason="invalid_format"  
 ).inc()  
 return {"status": "error"}

Такой подход позволяет в Grafana строить графики не просто «ошибок», а распределения ошибок по типам документов, что дает более глубокое понимание природы сбоев.17

### **3.3 Типы метрик Prometheus**

Понимание типов данных необходимо для правильного выбора функций агрегации в PromQL (языке запросов Prometheus).16

| Тип           | Описание                                                                                       | Пример использования                                                              | Нюансы обработки                                                                                                                      |
| :------------ | :--------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| **Counter**   | Монотонно возрастающее число. Может только увеличиваться или сбрасываться в ноль при рестарте. | http_requests_total (Всего запросов), errors_total (Всего ошибок).                | **Никогда** не визуализируется напрямую. Всегда используется с функцией rate() или increase() для получения скорости изменения (RPS). |
| **Gauge**     | Число, которое может расти и падать.                                                           | memory_usage_bytes (Память), cpu_temperature (Температура), queue_size (Очередь). | Визуализируется как есть, либо с агрегациями avg(), max(), min().                                                                     |
| **Histogram** | Сложный тип, разбивающий наблюдения по «корзинам» (buckets) и подсчитывающий их количество.    | http_request_duration_seconds (Длительность запроса).                             | Позволяет рассчитывать квантили (например, 95-й перцентиль задержки) на стороне сервера с помощью histogram_quantile().               |
| **Summary**   | Аналог гистограммы, но квантили считаются на стороне клиента.                                  | rpc_duration_seconds.                                                             | Менее гибок, чем гистограмма, так как квантили нельзя пересчитать (агрегировать) заново.                                              |

## **4\. Конфигурация инфраструктуры: Docker Compose**

Файл docker-compose.yml является исполнительным чертежом «Сторожевой Башни». В нем описывается взаимодействие трех ключевых компонентов: приложения (FastAPI), агрегатора (Prometheus) и визуализатора (Grafana).

### **4.1 Полная спецификация Docker Compose**

Ниже приведена конфигурация, удовлетворяющая требованиям квеста и обеспечивающая персистентность данных.19

YAML

version: "3.8"

services:  
 \# \--- Целевое Приложение \---  
 fastapi_app:  
 build:. \# Предполагается наличие Dockerfile в корне  
 container_name: watchtower_app  
 ports:  
 \- "8000:8000" \# Проброс порта для внешнего доступа (тестирование)  
 networks:  
 \- monitoring_net  
 environment:  
 \- APP_ENV=production  
 restart: unless-stopped

\# \--- Агрегатор Метрик \---  
 prometheus:  
 image: prom/prometheus:latest  
 container_name: watchtower_prometheus  
 volumes:  
 \# Монтирование конфигурации  
 \-./prometheus.yml:/etc/prometheus/prometheus.yml:ro  
 \# Персистентное хранилище базы данных временных рядов (TSDB)  
 \- prometheus_data:/prometheus  
 command:  
 \# Указание файла конфигурации  
 \- '--config.file=/etc/prometheus/prometheus.yml'  
 \# Указание директории хранения  
 \- '--storage.tsdb.path=/prometheus'  
 \# Срок хранения данных (Retension Policy)  
 \- '--storage.tsdb.retention.time=15d'  
 ports:  
 \- "9090:9090" \# Веб-интерфейс Prometheus  
 networks:  
 \- monitoring_net  
 depends_on:  
 \- fastapi_app

\# \--- Визуализатор \---  
 grafana:  
 image: grafana/grafana:latest  
 container_name: watchtower_grafana  
 ports:  
 \- "3000:3000"  
 networks:  
 \- monitoring_net  
 environment:  
 \- GF_SECURITY_ADMIN_USER=admin  
 \- GF_SECURITY_ADMIN_PASSWORD=secret_password \# В продакшене использовать.env файл  
 \- GF_USERS_ALLOW_SIGN_UP=false  
 volumes:  
 \# Персистентное хранилище дашбордов и настроек  
 \- grafana_data:/var/lib/grafana  
 depends_on:  
 \- prometheus

volumes:  
 prometheus_data: \# Именованный том Docker для Prometheus  
 grafana_data: \# Именованный том Docker для Grafana

networks:  
 monitoring_net:  
 driver: bridge

### **4.2 Персистентность данных и тома**

Одной из частых проблем при развертывании мониторинга в Docker является потеря исторических данных при перезапуске контейнеров. База данных временных рядов (TSDB) Prometheus и база данных SQLite Grafana (хранящая дашборды и пользователей) находятся внутри контейнеров. Без использования томов (volumes) эти данные эфемерны.21  
В конфигурации выше используются **именованные тома** (prometheus_data и grafana_data). В отличие от bind mounts (привязки к папке на хосте), именованные тома управляются демоном Docker и хранятся в специальной директории (обычно /var/lib/docker/volumes). Это решает проблему прав доступа (permissions issue), которая часто возникает, когда процесс внутри контейнера работает от пользователя nobody или grafana и не может писать в примонтированную папку хоста.22

### **4.3 Настройка сбора метрик: prometheus.yml**

Файл конфигурации prometheus.yml определяет, _кого_ и _как часто_ опрашивать.

YAML

global:  
 scrape_interval: 15s \# Глобальный интервал опроса  
 evaluation_interval: 15s \# Интервал вычисления правил алертинга

scrape_configs:  
 \# Задача самомониторинга Prometheus  
 \- job_name: 'prometheus'  
 static_configs:  
 \- targets: \['localhost:9090'\]

\# Задача мониторинга FastAPI  
 \- job_name: 'fastapi_service'  
 scrape_interval: 5s \# Более частый опрос для точного детектирования спайков  
 metrics_path: '/metrics'  
 static_configs:  
 \- targets: \['fastapi_app:8000'\] \# Использование DNS-имени сервиса Docker  
 labels:  
 env: 'production'  
 service_type: 'backend'

Ключевой аспект Service Discovery:  
Обратите внимание на цель targets: \['fastapi_app:8000'\]. Здесь fastapi_app — это имя сервиса из docker-compose.yml. Prometheus, находясь в той же сети monitoring_net, запрашивает у Docker DNS IP-адрес этого хоста. Если мы масштабируем приложение командой docker-compose up \--scale fastapi_app=3, Docker DNS будет отдавать IP-адреса разных реплик по принципу Round Robin, однако для полноценной поддержки масштабирования в Prometheus лучше использовать специализированный механизм dns_sd_configs вместо static_configs, чтобы получать список всех IP сразу.10

## **5\. Визуализация в Grafana: Создание Панели Управления**

Grafana выступает в роли «остекления» Сторожевой Башни, преобразуя сухие цифры в инсайты. Процесс настройки начинается с добавления источника данных (Data Source).

### **5.1 Подключение источника данных Prometheus**

При настройке Data Source в Grafana URL-адрес сервера Prometheus является критически важным параметром.

- **URL:** http://prometheus:9090
- **Access:** Server (default).

Частая ошибка — указывать http://localhost:9090. Поскольку Grafana работает в контейнере, localhost для неё — это она сама. Она попытается найти Prometheus внутри своего контейнера и выдаст ошибку соединения. Необходимо использовать имя контейнера Prometheus как хостнейм.11

### **5.2 Методология RED и построение дашбордов**

Для создания эффективного дашборда рекомендуется использовать методологию **RED** (Rate, Errors, Duration), разработанную Томом Уилки.26

#### **5.2.1 Rate (Интенсивность запросов)**

Этот показатель отвечает на вопрос: «Какова нагрузка на систему?».

- **Метрика:** http_requests_total (Counter).
- **Запрос PromQL:** sum(rate(http_requests_total\[1m\])) by (method, handler)
- **Пояснение:** Функция rate() вычисляет производную (скорость изменения) счетчика за указанный интервал (\[1m\]). Это преобразует монотонно растущее число (например, 100500 запросов с момента старта) в понятную величину (например, 50 запросов в секунду \- RPS).27
- **Визуализация:** Time Series (график).

#### **5.2.2 Errors (Ошибки)**

Отвечает на вопрос: «Какая доля запросов завершается неудачей?».

- **Метрика:** Фильтрация http_requests_total по статус-кодам.
- **Запрос PromQL:**  
  Фрагмент кода  
  sum(rate(http_requests_total{status=\~"5.."}\[1m\]))  
  /  
  sum(rate(http_requests_total\[1m\]))

- **Пояснение:** Оператор \=\~ использует регулярное выражение для выбора всех кодов 5xx (ошибки сервера). Деление на общий rate дает процент ошибок.
- **Визуализация:** Stat Panel (процент) или Gauge с цветовым порогом (зеленый \< 1%, красный \> 5%).26

#### **5.2.3 Duration (Длительность)**

Отвечает на вопрос: «Как быстро система отвечает пользователям?».

- **Метрика:** http_request_duration_seconds_bucket (Histogram).
- **Запрос PromQL:**  
  Фрагмент кода  
  histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket\[5m\])) by (le))

- **Пояснение:** Среднее время отклика (Average) часто скрывает проблемы. 95-й перцентиль (P95) показывает время, в которое укладываются 95% запросов. Это наиболее честный показатель пользовательского опыта. Функция histogram_quantile аппроксимирует это значение на основе бакетов гистограммы.18
- **Визуализация:** Time Series или Heatmap (тепловая карта) для отображения распределения задержек.

## **6\. Расширение системы: Алертинг и Alertmanager**

«Сторожевая Башня» должна не только показывать проблемы, но и оповещать о них. Хотя Grafana имеет встроенную систему алертинга, каноническим способом в экосистеме Prometheus является использование компонента **Alertmanager**.

### **6.1 Интеграция Alertmanager**

Для этого в docker-compose.yml добавляется новый сервис 30:

YAML

alertmanager:  
 image: prom/alertmanager:latest  
 ports:  
 \- "9093:9093"  
 volumes:  
 \-./alertmanager.yml:/etc/alertmanager/config.yml  
 networks:  
 \- monitoring_net  
 command:  
 \- '--config.file=/etc/alertmanager/config.yml'

А в prometheus.yml добавляется секция alerting:

YAML

alerting:  
 alertmanagers:  
 \- static_configs:  
 \- targets:  
 \- alertmanager:9093  
rule_files:  
 \- "alert_rules.yml"

### **6.2 Правила алертинга (Recording Rules)**

В файле alert_rules.yml определяются условия срабатывания тревоги.  
**Пример: Сервис недоступен (Instance Down)**

YAML

groups:  
\- name: availability  
 rules:  
 \- alert: InstanceDown  
 expr: up \== 0  
 for: 1m  
 labels:  
 severity: critical  
 annotations:  
 summary: "Instance {{ $labels.instance }} down"  
 description: "{{ $labels.instance }} of job {{ $labels.job }} has been down for more than 1 minute."

**Пример: Высокая частота ошибок (High Error Rate)**

YAML

\- alert: HighErrorRate  
 expr: (sum(rate(http_requests_total{status=\~"5.."}\[1m\])) / sum(rate(http_requests_total\[1m\]))) \> 0.05  
 for: 2m  
 labels:  
 severity: warning  
 annotations:  
 summary: "High error rate detected"

Alertmanager отвечает за дедупликацию (чтобы не пришло 100 писем за минуту), группировку и маршрутизацию уведомлений в Slack, Email, PagerDuty или Telegram.32

## **7\. Безопасность и эксплуатация в Production**

При переводе «Сторожевой Башни» в продуктивную среду необходимо учесть аспекты безопасности, которые часто игнорируются в локальных разработках.

### **7.1 Защита эндпоинтов**

По умолчанию Prometheus и Grafana работают по протоколу HTTP без шифрования. В файле docker-compose.yml порты 9090 и 3000 проброшены на хост.

- **Риск:** Если сервер имеет публичный IP, интерфейс Prometheus (с возможностью удаления данных через Admin API) и Grafana будут доступны всему интернету.
- **Решение:**
  1. Не пробрасывать порт 9090 на хост, если доступ к Prometheus нужен только Grafana (они в одной Docker-сети). Убрать директиву ports для Prometheus.
  2. Использовать Reverse Proxy (Nginx/Traefik) перед Grafana для терминации SSL/TLS (HTTPS).33

### **7.2 Управление секретами**

В представленной конфигурации пароль администратора Grafana передается через переменную окружения GF_SECURITY_ADMIN_PASSWORD в открытом виде.

- **Best Practice:** Использовать .env файл, который исключен из системы контроля версий (.gitignore).
- **Docker Secrets:** В режиме Docker Swarm рекомендуется использовать механизм secrets для безопасной инъекции паролей в контейнеры в виде файлов, а не переменных окружения.34

### **7.3 Лимиты ресурсов**

Контейнеры мониторинга, особенно Prometheus, могут потреблять значительное количество оперативной памяти, так как они кэшируют чанки временных рядов в RAM.

- **Рекомендация:** В docker-compose.yml следует задать лимиты ресурсов для предотвращения влияния мониторинга на основное приложение (Noisy Neighbor effect).

YAML

    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

## **8\. Заключение**

Реализация архитектуры «Сторожевая Башня» трансформирует процесс эксплуатации FastAPI-приложений. Интеграция prometheus-fastapi-instrumentator обеспечивает автоматический сбор метрик с минимальными накладными расходами на производительность. Оркестрация через Docker Compose создает изолированную, воспроизводимую среду, где сетевое взаимодействие между компонентами (App \-\> Prometheus \-\> Grafana) обеспечивается встроенным DNS-механизмом Docker.  
Ключевым результатом данного квеста является не просто факт сбора данных, а возможность перехода к **Data-Driven SRE**. Используя описанные выше конфигурации для расчета показателей RED (Rate, Errors, Duration) и 95-го перцентиля задержки, инженерные команды получают объективную картину здоровья системы. Добавление Alertmanager замыкает цикл обратной связи, позволяя реагировать на инциденты до того, как они станут критическими для пользователей.  
Данный отчет подтверждает выполнение Квеста 29.1 и предоставляет фундамент для дальнейшего масштабирования системы наблюдаемости, включая добавление сбора логов (Loki) и трейсинга (Tempo).

#### **Источники**

1. What is Prometheus? | New Relic, дата последнего обращения: декабря 22, 2025, [https://newrelic.com/blog/observability/what-is-prometheus](https://newrelic.com/blog/observability/what-is-prometheus)
2. Overview | Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/introduction/overview/](https://prometheus.io/docs/introduction/overview/)
3. Is Prometheus Monitoring Push or Pull? \- SigNoz, дата последнего обращения: декабря 22, 2025, [https://signoz.io/guides/is-prometheus-monitoring-push-or-pull/](https://signoz.io/guides/is-prometheus-monitoring-push-or-pull/)
4. Pull doesn't scale \- or does it? \- Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/blog/2016/07/23/pull-does-not-scale-or-does-it/](https://prometheus.io/blog/2016/07/23/pull-does-not-scale-or-does-it/)
5. Why is Prometheus Pull-Based? \- DEV Community, дата последнего обращения: декабря 22, 2025, [https://dev.to/mikkergimenez/why-is-prometheus-pull-based-36k1](https://dev.to/mikkergimenez/why-is-prometheus-pull-based-36k1)
6. Troubleshooting guide for the Prometheus agent \- New Relic Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.newrelic.com/docs/infrastructure/prometheus-integrations/install-configure-prometheus-agent/troubleshooting-guide/](https://docs.newrelic.com/docs/infrastructure/prometheus-integrations/install-configure-prometheus-agent/troubleshooting-guide/)
7. Prometheus how "up" metrics works \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/55162188/prometheus-how-up-metrics-works](https://stackoverflow.com/questions/55162188/prometheus-how-up-metrics-works)
8. Docker Compose Networking Mysteries Service Discovery Failures and Port Conflicts, дата последнего обращения: декабря 22, 2025, [https://www.netdata.cloud/academy/docker-compose-networking-mysteries/](https://www.netdata.cloud/academy/docker-compose-networking-mysteries/)
9. Networking in Compose \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/how-tos/networking/](https://docs.docker.com/compose/how-tos/networking/)
10. Precedence of DNS entry vs. Compose service name \- Docker Community Forums, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/precedence-of-dns-entry-vs-compose-service-name/120967](https://forums.docker.com/t/precedence-of-dns-entry-vs-compose-service-name/120967)
11. Enter the Prometheus data source URL | Grafana Labs, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/learning-journeys/prometheus/add-data-source-url/](https://grafana.com/docs/learning-journeys/prometheus/add-data-source-url/)
12. Configure the Prometheus data source | Grafana documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/datasources/prometheus/configure/](https://grafana.com/docs/grafana/latest/datasources/prometheus/configure/)
13. Releases · trallnag/prometheus-fastapi-instrumentator \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/trallnag/prometheus-fastapi-instrumentator/releases](https://github.com/trallnag/prometheus-fastapi-instrumentator/releases)
14. trallnag/prometheus-fastapi-instrumentator \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/trallnag/prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
15. prometheus-fastapi-instrumentator 1.1.1 \- PyPI, дата последнего обращения: декабря 22, 2025, [https://pypi.org/project/prometheus-fastapi-instrumentator/1.1.1/](https://pypi.org/project/prometheus-fastapi-instrumentator/1.1.1/)
16. A Practical Guide to Prometheus Metric Types | Better Stack Community, дата последнего обращения: декабря 22, 2025, [https://betterstack.com/community/guides/monitoring/prometheus-metrics-explained/](https://betterstack.com/community/guides/monitoring/prometheus-metrics-explained/)
17. prometheus-fastapi-instrumentator custom counters for functions \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/75168335/prometheus-fastapi-instrumentator-custom-counters-for-functions](https://stackoverflow.com/questions/75168335/prometheus-fastapi-instrumentator-custom-counters-for-functions)
18. Understanding the Prometheus Metric Types \- Dash0, дата последнего обращения: декабря 22, 2025, [https://www.dash0.com/knowledge/prometheus-metrics](https://www.dash0.com/knowledge/prometheus-metrics)
19. FastAPI Observability Lab with Prometheus and Grafana: Complete Guide \- Towards AI, дата последнего обращения: декабря 22, 2025, [https://pub.towardsai.net/fastapi-observability-lab-with-prometheus-and-grafana-complete-guide-f12da15a15fd](https://pub.towardsai.net/fastapi-observability-lab-with-prometheus-and-grafana-complete-guide-f12da15a15fd)
20. Monitoring FastAPI with Grafana \+ Prometheus: A 5-Minute Guide \- Level Up Coding, дата последнего обращения: декабря 22, 2025, [https://levelup.gitconnected.com/monitoring-fastapi-with-grafana-prometheus-a-5-minute-guide-658280c7f358](https://levelup.gitconnected.com/monitoring-fastapi-with-grafana-prometheus-a-5-minute-guide-658280c7f358)
21. Installation \- Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/prometheus/latest/installation/](https://prometheus.io/docs/prometheus/latest/installation/)
22. How to Persist Data in Prometheus Running in a Docker Container? \- Better Stack, дата последнего обращения: декабря 22, 2025, [https://betterstack.com/community/questions/how-to-persist-data-in-prometheus-running-in-docker-container/](https://betterstack.com/community/questions/how-to-persist-data-in-prometheus-running-in-docker-container/)
23. How to persist data in Prometheus running in a Docker container? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/50009065/how-to-persist-data-in-prometheus-running-in-a-docker-container](https://stackoverflow.com/questions/50009065/how-to-persist-data-in-prometheus-running-in-a-docker-container)
24. Collect Docker metrics with Prometheus, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/engine/daemon/prometheus/](https://docs.docker.com/engine/daemon/prometheus/)
25. FastAPI, Prometheus, Grafana, Graylog in docker-compose example | by Denis Gr \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@gribenyukdenis/fastapi-prometheus-graphana-graylog-in-docker-compose-example-c4be14e3057f](https://medium.com/@gribenyukdenis/fastapi-prometheus-graphana-graylog-in-docker-compose-example-c4be14e3057f)
26. Grafana dashboard best practices, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/visualizations/dashboards/build-dashboards/best-practices/](https://grafana.com/docs/grafana/latest/visualizations/dashboards/build-dashboards/best-practices/)
27. How to Measure Total Requests with Prometheus \- A Time-Based Guide | SigNoz, дата последнего обращения: декабря 22, 2025, [https://signoz.io/guides/how-to-get-total-requests-in-a-period-of-time-with-prometheus/](https://signoz.io/guides/how-to-get-total-requests-in-a-period-of-time-with-prometheus/)
28. Why Grafana's Rate Function Is Your Dashboard's Best Kept Secret | Last9, дата последнего обращения: декабря 22, 2025, [https://last9.io/blog/grafana-rate-function/](https://last9.io/blog/grafana-rate-function/)
29. Stat | Grafana documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/stat/](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/stat/)
30. Prometheus with Docker Compose: Guide & Examples \- Spacelift, дата последнего обращения: декабря 22, 2025, [https://spacelift.io/blog/prometheus-docker-compose](https://spacelift.io/blog/prometheus-docker-compose)
31. Simple Prometheus setup on Docker Compose | by Ulises Martinez \- Medium, дата последнего обращения: декабря 22, 2025, [https://mxulises.medium.com/simple-prometheus-setup-on-docker-compose-f702d5f98579](https://mxulises.medium.com/simple-prometheus-setup-on-docker-compose-f702d5f98579)
32. Alertmanager | Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/alerting/latest/alertmanager/](https://prometheus.io/docs/alerting/latest/alertmanager/)
33. Configure security hardening | Grafana documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/setup-grafana/configure-security/configure-security-hardening/](https://grafana.com/docs/grafana/latest/setup-grafana/configure-security/configure-security-hardening/)
34. Configure a Grafana Docker image, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/setup-grafana/configure-docker/](https://grafana.com/docs/grafana/latest/setup-grafana/configure-docker/)
35. Noob Question: Best practice for providing sensitive environment variables at runtime. : r/docker \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/docker/comments/1bpjni5/noob_question_best_practice_for_providing/](https://www.reddit.com/r/docker/comments/1bpjni5/noob_question_best_practice_for_providing/)
