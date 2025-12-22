# **Архитектура и Оркестрация Мульти-Контейнерных Сред: Глубокий Анализ Docker Compose**

## **1\. Введение: Абстракция Инфраструктуры и Парадигма Микросервисов**

Современная разработка программного обеспечения претерпела фундаментальный сдвиг от монолитных архитектур к распределенным микросервисным системам. В этом контексте задача управления инфраструктурой трансформировалась из ручной настройки серверов в декларативное описание состояний. "Магия Мульти-Контейнеров", упомянутая в задании Квеста 25.1, на самом деле представляет собой строгую инженерную дисциплину, в которой сложность сетевого взаимодействия, управления хранилищами и жизненным циклом процессов абстрагируется через инструменты оркестрации. Docker Compose выступает в роли критически важного транслятора, преобразующего высокоуровневые определения сервисов в низкоуровневые системные вызовы ядра Linux, необходимые для создания изолированных пользовательских пространств (контейнеров), виртуальных сетевых мостов и томов для хранения данных.  
Данный отчет представляет собой исчерпывающее исследование механизмов, лежащих в основе Docker Compose, с целью выполнения требований по созданию файла docker-compose.yaml. Мы деконструируем кажущуюся "магию" оркестрации, раскрывая принципы service discovery, управления зависимостями и персистентности данных. Анализ базируется на предоставленных исследовательских материалах и охватывает весь спектр технологий: от спецификации Compose до интеграции с Python (FastAPI), Redis и PostgreSQL. Цель документа — предоставить не просто инструкцию, а глубокое архитектурное понимание того, как создать надежную, масштабируемую и безопасную среду для разработки и эксплуатации.

## **2\. Эволюция Спецификации Docker Compose: От Разделения к Унификации**

Понимание синтаксиса и возможностей файла docker-compose.yaml невозможно без глубокого погружения в историю его версионирования. Экосистема Docker прошла через несколько фаз развития, что породило значительную путаницу в документации и подходах к конфигурации.

### **2.1 Исторический Раскол: Версии 2.x против 3.x**

Исторически формат файла Compose был разделен на две основные ветви, каждая из которых преследовала свои цели, создавая дихотомию в инструментарии разработчиков.1

- **Версия 2.x:** Изначально эта ветка была спроектирована для локальных сред разработки (single-node development). Она предоставляла зеркальное отражение возможностей команды docker run, позволяя разработчикам тонко настраивать ограничения ресурсов (CPU, память) непосредственно в конфигурации сервиса. Это делало версию 2.x идеальным выбором для локального тестирования, где требовалось точное воспроизведение ограничений среды.2
- **Версия 3.x:** С появлением Docker Swarm, встроенного оркестратора для кластеризации, была представлена третья версия спецификации. В ней акцент сместился на развертывание в распределенных системах. Критическим изменением стал перенос параметров управления ресурсами под ключ deploy. Проблема заключалась в том, что стандартный бинарный файл docker-compose игнорировал секцию deploy при запуске вне режима Swarm. Это приводило к тому, что ограничения ресурсов, критичные для предотвращения "шумных соседей" (noisy neighbor problem), просто не применялись при локальной разработке.3

Этот раскол заставлял архитекторов выбирать между схемой, оптимизированной для локальной работы (v2), и схемой для продакшн-кластеров (v3). Многие разработчики по инерции использовали "последнюю" версию (например, 3.8 или 3.9), не осознавая, что теряют контроль над гранулярным управлением ресурсами на локальных машинах.5

### **2.2 Современная Эпоха: Compose Specification**

Современный ландшафт контейнеризации разрешил этот конфликт через введение **Compose Specification**. Этот открытый стандарт объединил возможности версий 2.x и 3.x в единый, агностический формат. Спецификация теперь управляется сообществом и реализована в современных инструментах Docker.1  
Ключевые аспекты современной спецификации:

- **Опциональность версии:** Поле version в корне YAML-файла теперь является информативным, но не обязательным для новых версий CLI. Инструменты автоматически определяют возможности на основе схемы.
- **Контекстная интерпретация:** Функции, ранее жестко разделенные (например, конфигурации deploy), теперь интерпретируются более гибко. Если параметр не поддерживается текущим бэкендом (например, локальным Docker Engine), он может быть проигнорирован с предупреждением, но не вызывает фатальной ошибки, если это не критично.
- **Переход на Go (Docker Compose v2):** Важнейшим изменением стал переход от утилиты docker-compose (написанной на Python, известной как v1) к плагину docker compose (написанному на Go, v2).3

### **2.3 Сравнительный Анализ Инструментария: v1 (Python) vs v2 (Go)**

Понимание разницы между командами docker-compose (с дефисом) и docker compose (с пробелом) критически важно для операционной совместимости и производительности.

| Характеристика        | Docker Compose v1 (Legacy)                             | Docker Compose v2 (Modern)                 |
| :-------------------- | :----------------------------------------------------- | :----------------------------------------- |
| **Язык реализации**   | Python                                                 | Go (Golang)                                |
| **Команда вызова**    | docker-compose                                         | docker compose                             |
| **Спецификация**      | Строгая зависимость от version: '2' или '3'            | Полная поддержка Compose Specification     |
| **Имена контейнеров** | Использовали нижнее подчеркивание (\_) как разделитель | Используют дефис (-) по умолчанию 6        |
| **Статус поддержки**  | Deprecated, поддержка прекращена (июнь 2023\) 3        | Активная разработка, стандарт по умолчанию |
| **Интеграция**        | Отдельный бинарный файл                                | Интегрированный плагин Docker CLI          |

**Глубокий инсайт:** Хотя v2 является обратно совместимым, изменение разделителя в именах контейнеров с \_ на \- может сломать скрипты автоматизации, которые полагаются на парсинг имен контейнеров (например, project_web_1 vs project-web-1). Для сохранения старого поведения необходимо использовать флаг \--compatibility.6 Однако, для новых проектов ("Квест 25.1") настоятельно рекомендуется использовать нативный синтаксис v2 и отказаться от устаревшего поля version в YAML.

## **3\. Стратегии Управления Образами: Alpine против Slim**

При определении сервисов в секции services, выбор базового образа для Python-приложений (например, FastAPI) является одним из самых обсуждаемых и критичных архитектурных решений. Этот выбор напрямую влияет на размер итогового артефакта, скорость сборки CI/CD пайплайнов и стабильность работы в продакшене.

### **3.1 Дилемма Alpine Linux**

Alpine Linux традиционно привлекает разработчиков своим минимализмом. Базовый образ может занимать всего около 23.5 МБ, что кажется идеальным для микросервисов.8 Однако, этот минимализм достигается за счет использования библиотеки musl libc вместо стандартной glibc, используемой в большинстве дистрибутивов Linux (Debian, Ubuntu, CentOS).  
Проблематика Python на Alpine:  
Python-сообщество распространяет предварительно скомпилированные бинарные пакеты (wheels) в формате manylinux, которые динамически слинкованы с glibc. Поскольку Alpine использует musl, стандартные wheels (например, для numpy, pandas, grpcio или драйверов баз данных типа psycopg2) несовместимы с ним.  
Это приводит к следующим последствиям 9:

1. **Компиляция из исходников:** При установке зависимостей pip вынужден скачивать исходный код пакетов и компилировать их непосредственно во время сборки образа.
2. **Раздувание образа:** Для компиляции необходимо устанавливать тяжеловесные инструменты сборки (gcc, g++, make, заголовочные файлы ядра). Хотя их можно удалить в многоэтапной сборке (multi-stage build), сам процесс сборки становится значительно сложнее.
3. **Замедление CI/CD:** Время сборки увеличивается в десятки раз (иногда до 50 раз медленнее), так как компиляция C-расширений — ресурсоемкая операция.9
4. **Скрытые баги:** Существуют тонкие различия в поведении musl и glibc (например, в обработке DNS или потоков), которые могут привести к трудноуловимым ошибкам в продакшене.

### **3.2 Преимущество Debian Slim**

Образы на базе Debian Slim (например, python:3.10-slim) представляют собой урезанную версию Debian. Из них удалены документация (man-pages) и вспомогательные утилиты, но сохранена библиотека glibc.

- **Размер:** Хотя они больше Alpine (около 140 МБ против 23 МБ в распакованном виде), разница нивелируется при установке тяжелых зависимостей, так как не требуются компиляторы.10
- **Совместимость:** Полная поддержка стандартных manylinux wheels. Установка пакетов происходит мгновенно, так как скачиваются готовые бинарные файлы.

**Рекомендация для Квеста:** Для сервисов, использующих Python и требующих взаимодействия с базами данных (драйверы PostgreSQL, Redis) или сложных вычислений, использование python:3.10-slim является **best practice**. Это обеспечивает баланс между размером, скоростью сборки и надежностью.11 Использование Alpine оправдано только для статически скомпилированных языков (например, Go) или крайне простых Python-скриптов без C-зависимостей.

### **3.3 Контекст Сборки и Многоэтапность**

Docker Compose позволяет гибко управлять сборкой через секцию build.

YAML

services:  
 web:  
 build:  
 context:.  
 dockerfile: Dockerfile  
 args:  
 \- BUILDKIT_INLINE_CACHE=1

Интеграция аргументов сборки (args) позволяет передавать параметры в Dockerfile (инструкция ARG), что дает возможность использовать один и тот же Dockerfile для создания разных вариаций образа (например, dev-версия с отладчиками и prod-версия без них). Спецификация Compose также поддерживает использование secrets на этапе сборки, что критично для безопасного внедрения приватных SSH-ключей или токенов доступа без их сохранения в слоях образа.13

## **4\. Сетевая Магия: Service Discovery и Внутренний DNS**

Одной из самых "магических" функций Docker Compose является автоматическая настройка сети, позволяющая контейнерам общаться друг с другом по именам сервисов, игнорируя IP-адреса.

### **4.1 Механизм Внутреннего DNS**

Когда вы запускаете docker compose up, Docker создает изолированную сеть типа bridge (по умолчанию именуемую \<project_name\>\_default).15

1. **Регистрация:** Каждый контейнер при запуске получает IP-адрес внутри этой подсети. Docker Daemon автоматически регистрирует имя сервиса (как указано в YAML) и имя контейнера в своем встроенном DNS-сервере.
2. **DNS Resolver (127.0.0.11):** Внутри каждого контейнера файл /etc/resolv.conf настроен на использование специального DNS-резолвера по адресу 127.0.0.11 (это внутренний прокси Docker).17
3. **Разрешение имен:** Когда код приложения (например, FastAPI) пытается подключиться к хосту db, запрос перехватывается этим резолвером. Docker возвращает внутренний IP-адрес контейнера базы данных.

**Нюанс Round-Robin:** Если сервис масштабирован (например, запущено 3 реплики веб\-сервера), внутренний DNS Docker будет возвращать список IP-адресов всех реплик, циклически меняя их порядок (Round-Robin DNS). Это обеспечивает базовую балансировку нагрузки на уровне 4 (транспортный уровень) без необходимости во внешнем балансировщике.18

### **4.2 Проблема Привязки Портов: 0.0.0.0 против 127.0.0.1**

Самая распространенная ошибка при контейнеризации веб\-приложений (особенно на Python/Uvicorn/Gunicorn) связана с пониманием сетевых интерфейсов.19  
В традиционной разработке запуск сервера на localhost (127.0.0.1) является стандартом безопасности. Однако в контексте контейнеров:

- **Loopback Интерфейс (lo):** Внутри контейнера 127.0.0.1 указывает только на _сам_ контейнер. Процесс, слушающий этот адрес, недоступен ни для хоста, ни для других контейнеров.
- **Docker Bridge Интерфейс (eth0):** Чтобы сервис был доступен извне (через проброс портов \-p) или из других контейнеров сети, он должен слушать на адресе 0.0.0.0. Это мета-адрес, означающий "все доступные интерфейсы".20

**Решение:** В файле docker-compose.yaml команда запуска должна явно указывать хост:

YAML

command: uvicorn main:app \--host 0.0.0.0 \--port 8000

Без этого флага (--host 0.0.0.0) сервер запустится на 127.0.0.1 по умолчанию, и запросы от Docker Proxy, перенаправляющего трафик с хоста, будут отвергнуты ("Connection Refused"), так как они приходят на интерфейс eth0, который никто не слушает.19

### **4.3 Именование Сетей и Проектов**

По умолчанию имя сети формируется на основе имени директории проекта. Это может привести к коллизиям, если у вас есть две папки с одинаковым именем app в разных местах.  
Для избежания этого и явного контроля рекомендуется:

1. Задавать имя проекта через переменную окружения COMPOSE_PROJECT_NAME или флаг \-p.23
2. Явно именовать сети в YAML файле, если требуется интеграция с внешними ресурсами.25

YAML

networks:  
 default:  
 name: magic_network_custom

## **5\. Управление Состоянием: Тома и Персистентность Данных**

Контейнеры по своей природе эфемерны. Любые данные, записанные в файловую систему контейнера (layer R/W), исчезают при его удалении. Для баз данных (PostgreSQL, Redis) это недопустимо. Docker Compose решает эту проблему через механизм томов (Volumes).

### **5.1 Типология Томов: Bind Mounts vs Named Volumes**

| Тип Тома         | Синтаксис     | Назначение               | Особенности                                                                                                                                                                                                                             |
| :--------------- | :------------ | :----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Bind Mount**   | ./src:/app    | Разработка (Development) | Монтирует директорию хоста в контейнер. Изменения файлов на хосте мгновенно видны в контейнере. Критично для "Hot Reloading".26 Зависит от файловой системы хоста.                                                                      |
| **Named Volume** | db_data:/data | Данные (Persistence)     | Управляется Docker. Хранится в /var/lib/docker/volumes. Полностью отделен от структуры проекта. Идеален для БД, так как обеспечивает лучшую производительность файловой системы по сравнению с bind mounts (особенно на Windows/Mac).27 |

### **5.2 Жизненный Цикл Данных: Stop vs Down**

Понимание разницы между командами остановки критично для предотвращения случайной потери данных или, наоборот, для очистки среды.29

- **docker compose stop:** Останавливает запущенные контейнеры (отправляет SIGTERM, затем SIGKILL). Контейнеры не удаляются. Их состояние (включая данные во временном слое записи) сохраняется. Сети не удаляются.
- **docker compose down:** Останавливает и **удаляет** контейнеры и сети. Однако, **именованные тома по умолчанию не удаляются**. Это защитный механизм. Если вы перезапустите docker compose up после down, база данных подтянет старый том и данные сохранятся.26
- **docker compose down \-v:** Это "ядерная опция". Флаг \-v (или \--volumes) принуждает Docker удалить также и именованные тома. Это необходимо для полной очистки среды (например, для сброса базы данных к начальному состоянию).28

**Инсайт:** Часто возникающая проблема "почему моя база данных не инициализируется скриптом /docker-entrypoint-initdb.d?" связана именно с этим. Скрипты инициализации PostgreSQL запускаются _только_ если директория данных пуста. Если том сохранился после предыдущего запуска (даже после down), скрипты будут проигнорированы. Для повторной инициализации необходим docker compose down \-v.

## **6\. Конфигурация и Безопасность: Иерархия Переменных Окружения**

В соответствии с методологией 12-Factor App, конфигурация должна храниться в окружении. Docker Compose предоставляет сложную систему приоритетов для внедрения переменных, что часто становится источником ошибок конфигурации.

### **6.1 Матрица Приоритетов (Precedence Matrix)**

Значение переменной определяется в следующем порядке (от высшего к низшему) 31:

1. **CLI (docker compose run \-e):** Явное указание при запуске.
2. **Shell Environment:** Переменные, экспортированные в оболочке терминала (export VAR=val).
3. **Environment File (.env):** Значения из файла .env в корне проекта (загружаются автоматически).
4. **YAML environment attribute:** Значения, жестко прописанные в docker-compose.yaml.
5. **Dockerfile ENV:** Значения по умолчанию, зашитые в образ при сборке.

**Критический конфликт:** Переменные оболочки имеют приоритет над .env файлом. Это означает, что если у вас в терминале случайно установлена переменная POSTGRES_USER=admin, а в .env файле указано POSTGRES_USER=dev, Docker Compose использует значение admin из вашего терминала. Это часто приводит к ошибкам подключения ("FATAL: role does not exist"), которые трудно диагностировать, так как конфигурация в файлах выглядит корректной.33

### **6.2 Безопасность и .env файлы**

Хранение секретов (пароли БД, API ключи) в docker-compose.yaml является антипаттерном, так как этот файл обычно попадает в систему контроля версий (Git). Рекомендуемый подход — использование интерполяции переменных.32

YAML

services:  
 db:  
 environment:  
 POSTGRES_PASSWORD: ${DB_PASSWORD}

Файл .env, содержащий реальные значения (DB_PASSWORD=secret), должен быть добавлен в .gitignore. Для удобства команды следует создавать файл .env.example с безопасными значениями по умолчанию или пустыми шаблонами.34 В более сложных сценариях (особенно в Swarm) следует использовать механизм secrets, который монтирует секреты как файлы в /run/secrets, что безопаснее переменных окружения, но для локальной разработки через Compose подход с .env остается стандартом де\-факто.

## **7\. Оркестрация Запуска: Race Conditions и Healthchecks**

В мульти-контейнерной среде сервисы имеют зависимости. API не может работать без БД. Однако директива depends_on в своей базовой форме решает только часть проблемы.

### **7.1 Проблема "Запущен, но не готов"**

Классическая запись:

YAML

depends_on:  
 \- db

Она гарантирует только порядок запуска процессов. Docker запустит контейнер db, и как только процесс (PID 1\) стартует, он немедленно запустит web. Однако базе данных PostgreSQL может потребоваться 5-10 секунд для инициализации файловой системы и открытия сокета. В этот момент приложение попытается подключиться и упадет с ошибкой. Традиционным решением были скрипты-обертки типа wait-for-it.sh, но это "костыль".35

### **7.2 Нативные Healthchecks: service_healthy**

Современный стандарт — использование встроенных проверок здоровья (Healthchecks).

1. Определение проверки в сервисе-зависимости:  
   Сервис базы данных сам сообщает о своем состоянии.  
   YAML  
   healthcheck:  
    test:  
    interval: 5s  
    timeout: 5s  
    retries: 5  
    start_period: 10s

   Параметр start_period критически важен: он дает контейнеру время на бутстраппинг, в течение которого проваленные проверки не засчитываются в лимит retries.36

2. Условная зависимость:  
   Зависимый сервис ждет не просто старта контейнера, а его "здоровья".  
   YAML  
   depends_on:  
    db:  
    condition: service_healthy

   Это полностью устраняет состояние гонки (race condition) при запуске стека.36

**Влияние на производительность:** Следует учитывать, что слишком частые проверки (interval) создают лишнюю нагрузку, а слишком редкие — замедляют старт зависимых сервисов. Настройка этих параметров — это баланс между скоростью запуска среды и нагрузкой на CPU.37

## **8\. Интеграция Прикладного Уровня: FastAPI и Redis**

Специфика работы Python в контейнерах, особенно при использовании асинхронных фреймворков вроде FastAPI, требует особого внимания при интеграции с Redis.

### **8.1 Синхронный vs Асинхронный Клиент Redis**

FastAPI основан на asyncio. Использование блокирующих (синхронных) операций ввода-вывода внутри async def функций губительно для производительности. Если использовать стандартный синхронный клиент redis внутри асинхронного эндпоинта, весь цикл событий (event loop) будет заблокирован на время ожидания ответа от Redis. В контейнере с ограниченными ресурсами CPU это приведет к тому, что приложение перестанет отвечать на healthcheck-запросы и может быть перезагружено оркестратором.39  
**Решение:** Использовать асинхронный клиент (aioredis или redis.asyncio в новых версиях библиотеки redis-py).

Python

\# Правильный паттерн использования в FastAPI  
from redis.asyncio import Redis

@app.on_event("startup")  
async def startup_event():  
 app.state.redis \= Redis.from_url("redis://redis_cache:6379", decode_responses=True)

@app.get("/items/{item_id}")  
async def read_item(item_id: str):  
 value \= await app.state.redis.get(item_id) \# Неблокирующий вызов  
 return value

Это позволяет FastAPI обрабатывать сотни других запросов, пока один запрос ожидает данные от кэша.41

## **9\. Операционная Эксплуатация и Практическая Реализация**

Обобщая вышесказанное, мы можем сформировать эталонный файл docker-compose.yaml для выполнения Квеста 25.1. Этот файл учитывает все аспекты: от выбора образов до стратегии запуска.

### **9.1 Итоговый Артефакт: docker-compose.yaml**

YAML

name: quest_magic_stack \# Явное имя проекта для изоляции \[42\]

services:  
 \# \--- Application Service (FastAPI) \---  
 api:  
 build:  
 context:.  
 dockerfile: Dockerfile  
 \# Оптимизация кэша сборки  
 args:  
 \- ENV_TYPE=dev  
 image: myapp:custom-v1  
 container_name: magic_api  
 \# Привязка к 0.0.0.0 обязательна для доступа извне контейнера  
 command: uvicorn main:app \--host 0.0.0.0 \--port 8000 \--reload  
 ports:  
 \- "8000:8000" \# Проброс порта Host:Container  
 environment:  
 \# Использование имен сервисов как хостов для подключения \[16\]  
 \- DATABASE_URL=postgresql://user:password@db:5432/magic_db  
 \- REDIS_URL=redis://redis_cache:6379/0  
 depends_on:  
 \# Ожидание реальной готовности зависимостей \[36\]  
 db:  
 condition: service_healthy  
 redis_cache:  
 condition: service_healthy  
 volumes:  
 \-./src:/app/src \# Bind mount для горячей перезагрузки кода  
 networks:  
 \- magic_net

\# \--- Database Service (PostgreSQL) \---  
 db:  
 image: postgres:15-alpine  
 container_name: magic_db  
 environment:  
 POSTGRES_USER: user  
 POSTGRES_PASSWORD: password  
 POSTGRES_DB: magic_db  
 volumes:  
 \- postgres_data:/var/lib/postgresql/data \# Named volume для персистентности  
 networks:  
 \- magic_net  
 \# Нативный Healthcheck для PostgreSQL  
 healthcheck:  
 test:  
 interval: 5s  
 timeout: 5s  
 retries: 5  
 start_period: 10s \# Даем время на инициализацию БД  
 restart: unless-stopped

\# \--- Cache Service (Redis) \---  
 redis_cache:  
 image: redis:7-alpine  
 container_name: magic_redis  
 networks:  
 \- magic_net  
 \# Нативный Healthcheck для Redis  
 healthcheck:  
 test:  
 interval: 5s  
 timeout: 3s  
 retries: 3  
 restart: always

\# \--- Volumes Declaration \---  
volumes:  
 postgres_data: \# Создается Docker-ом, переживает 'docker compose down' (без \-v)

\# \--- Networks Declaration \---  
networks:  
 magic_net:  
 driver: bridge  
 name: magic_network_internal \# Явное имя для избежания коллизий

### **9.2 Анализ Архитектурных Решений в Коде**

1. **name: quest_magic_stack**: Это нововведение спецификации позволяет жестко задать имя проекта, отвязав его от имени папки, в которой лежит файл. Это предотвращает создание сети folder_default и делает деплой детерминированным.24
2. **Healthchecks**: Мы не используем скрипты ожидания в контейнере приложения. Вместо этого оркестратор (Compose) сам мониторит состояние БД и Redis, блокируя запуск API до "зеленого" статуса.
3. **Сетевая изоляция**: Все сервисы находятся в кастомной сети magic_net. Это хорошая практика, изолирующая стек от других контейнеров на хосте.
4. **Alpine для БД и Redis**: В отличие от Python-приложения (где мы бы рекомендовали slim), для Redis и Postgres официальные образы на базе Alpine отлично оптимизированы и безопасны, так как они содержат уже скомпилированные бинарные файлы, протестированные мейнтейнерами.

## **10\. Заключение**

"Магия" мульти-контейнеров, о которой говорится в задании, на самом деле является результатом строгой абстракции. Docker Compose позволяет разработчику мыслить понятиями сервисов, зависимостей и потоков данных, скрывая сложность настройки iptables, создания сетевых мостов и управления пространствами имен ядра (namespaces).  
Успешное выполнение Квеста 25.1 требует не просто копирования YAML-кода, но и понимания трех столпов надежной контейнеризации:

1. **Детерминизм окружения:** Использование конкретных тегов образов и файлов блокировки (lock files).
2. **Управление жизненным циклом:** Правильное использование healthcheck и depends_on для синхронизации запуска.
3. **Изоляция данных:** Четкое разделение кода (bind mounts) и данных (named volumes).

Внедрение этих практик переводит использование Docker Compose из разряда "утилиты для запуска" в разряд мощного инструмента управления инфраструктурой, способного обеспечить идентичность поведения системы на ноутбуке разработчика и в CI/CD пайплайне.43

#### **Источники**

1. Compose file reference | Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/reference/compose-file/](https://docs.docker.com/reference/compose-file/)
2. docker.docker.github.io-1/compose/compose-file/compose-versioning.md at master · docker-archive-public/docker.docker.github.io-1 · GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/docker/docker.github.io-1/blob/master/compose/compose-file/compose-versioning.md](https://github.com/docker/docker.github.io-1/blob/master/compose/compose-file/compose-versioning.md)
3. What does the first line in the "docker-compose.yml" file, that specifies the version mean?, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/76156527/what-does-the-first-line-in-the-docker-compose-yml-file-that-specifies-the-ve](https://stackoverflow.com/questions/76156527/what-does-the-first-line-in-the-docker-compose-yml-file-that-specifies-the-ve)
4. Docker Compose version 3.8 or 3.9 for latest?, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/docker-compose-version-3-8-or-3-9-for-latest/102439](https://forums.docker.com/t/docker-compose-version-3-8-or-3-9-for-latest/102439)
5. Compose schema version for depends_on condition to wait for successful service completion \- Docker Community Forums, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/compose-schema-version-for-depends-on-condition-to-wait-for-successful-service-completion/118493](https://forums.docker.com/t/compose-schema-version-for-depends-on-condition-to-wait-for-successful-service-completion/118493)
6. Migrate to Compose v2 \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/releases/migrate/](https://docs.docker.com/compose/releases/migrate/)
7. Announcing Compose V2 General Availability \- Docker, дата последнего обращения: декабря 22, 2025, [https://www.docker.com/blog/announcing-compose-v2-general-availability/](https://www.docker.com/blog/announcing-compose-v2-general-availability/)
8. Differences Between Standard Docker Images and Alpine \\ Slim Versions, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/differences-between-standard-docker-images-and-alpine-slim-versions/134973](https://forums.docker.com/t/differences-between-standard-docker-images-and-alpine-slim-versions/134973)
9. Alpine vs python-slim for deploying python data science stack? : r/docker \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/docker/comments/g5hb93/alpine_vs_pythonslim_for_deploying_python_data/](https://www.reddit.com/r/docker/comments/g5hb93/alpine_vs_pythonslim_for_deploying_python_data/)
10. The best Docker base image for your Python application (May 2024), дата последнего обращения: декабря 22, 2025, [https://pythonspeed.com/articles/base-image-python-docker-images/](https://pythonspeed.com/articles/base-image-python-docker-images/)
11. Why is the python docker image so big (\~750 MB)? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/31060871/why-is-the-python-docker-image-so-big-750-mb](https://stackoverflow.com/questions/31060871/why-is-the-python-docker-image-so-big-750-mb)
12. Slimming Down Your Docker Images: A Guide to Single-Stage vs. Multi-Stage Python Builds, дата последнего обращения: декабря 22, 2025, [https://dev.to/aakashkhanna/slimming-down-your-docker-images-a-guide-to-single-stage-vs-multi-stage-python-builds-3m77](https://dev.to/aakashkhanna/slimming-down-your-docker-images-a-guide-to-single-stage-vs-multi-stage-python-builds-3m77)
13. Services | Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/reference/compose-file/services/](https://docs.docker.com/reference/compose-file/services/)
14. Docker Compose: What's New, What's Changing, What's Next, дата последнего обращения: декабря 22, 2025, [https://www.docker.com/blog/new-docker-compose-v2-and-v1-deprecation/](https://www.docker.com/blog/new-docker-compose-v2-and-v1-deprecation/)
15. Networking in Compose \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/how-tos/networking/](https://docs.docker.com/compose/how-tos/networking/)
16. Docker Compose Networking Mysteries Service Discovery Failures and Port Conflicts, дата последнего обращения: декабря 22, 2025, [https://www.netdata.cloud/academy/docker-compose-networking-mysteries/](https://www.netdata.cloud/academy/docker-compose-networking-mysteries/)
17. How Docker Desktop Networking Works Under the Hood, дата последнего обращения: декабря 22, 2025, [https://www.docker.com/blog/how-docker-desktop-networking-works-under-the-hood/](https://www.docker.com/blog/how-docker-desktop-networking-works-under-the-hood/)
18. How does service discovery work with modern docker/docker-compose? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/37683508/how-does-service-discovery-work-with-modern-docker-docker-compose](https://stackoverflow.com/questions/37683508/how-does-service-discovery-work-with-modern-docker-docker-compose)
19. Escaping the Port Binding Loop: A Docker Developer's Rite of Passage \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@raufa0742/escaping-the-port-binding-loop-a-docker-developers-rite-of-passage-2f5c221ae2c2](https://medium.com/@raufa0742/escaping-the-port-binding-loop-a-docker-developers-rite-of-passage-2f5c221ae2c2)
20. 0.0.0.0 instead of localhost for port : r/docker \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/docker/comments/rk1kfp/0000_instead_of_localhost_for_port/](https://www.reddit.com/r/docker/comments/rk1kfp/0000_instead_of_localhost_for_port/)
21. Cannot connect to fast api server at localhost:8000 from my application which is running under a docker container \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/68205757/cannot-connect-to-fast-api-server-at-localhost8000-from-my-application-which-is](https://stackoverflow.com/questions/68205757/cannot-connect-to-fast-api-server-at-localhost8000-from-my-application-which-is)
22. Bug? Unable to access webserver via docker because it binds to localhost \- Prodigy Support, дата последнего обращения: декабря 22, 2025, [https://support.prodi.gy/t/bug-unable-to-access-webserver-via-docker-because-it-binds-to-localhost/7083](https://support.prodi.gy/t/bug-unable-to-access-webserver-via-docker-because-it-binds-to-localhost/7083)
23. Network different than defined name after docker-compose? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/60358862/network-different-than-defined-name-after-docker-compose](https://stackoverflow.com/questions/60358862/network-different-than-defined-name-after-docker-compose)
24. Specify a project name \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/how-tos/project-name/](https://docs.docker.com/compose/how-tos/project-name/)
25. Set default network name for compose \- Docker Community Forums, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/set-default-network-name-for-compose/36779](https://forums.docker.com/t/set-default-network-name-for-compose/36779)
26. I am new to Docker and for some reason my data are persisting even if i dont use volume in my files. \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/docker/comments/18vazu6/i_am_new_to_docker_and_for_some_reason_my_data/](https://www.reddit.com/r/docker/comments/18vazu6/i_am_new_to_docker_and_for_some_reason_my_data/)
27. Building Scalable APIs with FastAPI, Docker, and Docker Compose | by Devendra \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@devendra631995/building-scalable-apis-with-fastapi-docker-and-docker-compose-ce9bc5ca55d2](https://medium.com/@devendra631995/building-scalable-apis-with-fastapi-docker-and-docker-compose-ce9bc5ca55d2)
28. Why docker-compose down deletes my volume? how to define volume as external?, дата последнего обращения: декабря 22, 2025, [https://forums.docker.com/t/why-docker-compose-down-deletes-my-volume-how-to-define-volume-as-external/67433](https://forums.docker.com/t/why-docker-compose-down-deletes-my-volume-how-to-define-volume-as-external/67433)
29. Docker compose stop VS down. Discover the differences between… | by Laurap | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@laurap_85411/docker-compose-stop-vs-down-e4e8d6515a85](https://medium.com/@laurap_85411/docker-compose-stop-vs-down-e4e8d6515a85)
30. What is the difference between \`docker-compose\` commands \`down\`, \`kill\` and \`stop\`?, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/63740108/what-is-the-difference-between-docker-compose-commands-down-kill-and-sto](https://stackoverflow.com/questions/63740108/what-is-the-difference-between-docker-compose-commands-down-kill-and-sto)
31. Environment variables precedence in Docker Compose, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/how-tos/environment-variables/envvars-precedence/](https://docs.docker.com/compose/how-tos/environment-variables/envvars-precedence/)
32. Set, use, and manage variables in a Compose file with interpolation \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/](https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/)
33. How to make docker-compose ".env" file take precedence over shell env vars?, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/72832926/how-to-make-docker-compose-env-file-take-precedence-over-shell-env-vars](https://stackoverflow.com/questions/72832926/how-to-make-docker-compose-env-file-take-precedence-over-shell-env-vars)
34. FastAPI Docker Best Practices | Better Stack Community, дата последнего обращения: декабря 22, 2025, [https://betterstack.com/community/guides/scaling-python/fastapi-docker-best-practices/](https://betterstack.com/community/guides/scaling-python/fastapi-docker-best-practices/)
35. Forget wait-for-it, use docker-compose healthcheck and depends_on instead \- Denat Hoxha, дата последнего обращения: декабря 22, 2025, [https://www.denhox.com/posts/forget-wait-for-it-use-docker-compose-healthcheck-and-depends-on-instead/](https://www.denhox.com/posts/forget-wait-for-it-use-docker-compose-healthcheck-and-depends-on-instead/)
36. Docker Compose Health Checks: An Easy-to-follow Guide \- Last9, дата последнего обращения: декабря 22, 2025, [https://last9.io/blog/docker-compose-health-checks/](https://last9.io/blog/docker-compose-health-checks/)
37. docker-compose healthcheck and depends_on service_healthy is super slow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/69228847/docker-compose-healthcheck-and-depends-on-service-healthy-is-super-slow](https://stackoverflow.com/questions/69228847/docker-compose-healthcheck-and-depends-on-service-healthy-is-super-slow)
38. Control startup and shutdown order with Compose \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/compose/how-tos/startup-order/](https://docs.docker.com/compose/how-tos/startup-order/)
39. Asynchronous vs. Synchronous Functions in FastAPI When to Pick Which | Leapcell, дата последнего обращения: декабря 22, 2025, [https://leapcell.io/blog/asynchronous-vs-synchronous-functions-in-fastapi-when-to-pick-which](https://leapcell.io/blog/asynchronous-vs-synchronous-functions-in-fastapi-when-to-pick-which)
40. FastAPI is blocked when an endpoint takes longer \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/FastAPI/comments/1euhq69/fastapi_is_blocked_when_an_endpoint_takes_longer/](https://www.reddit.com/r/FastAPI/comments/1euhq69/fastapi_is_blocked_when_an_endpoint_takes_longer/)
41. Using Redis with FastAPI, дата последнего обращения: декабря 22, 2025, [https://redis.io/tutorials/develop/python/fastapi/](https://redis.io/tutorials/develop/python/fastapi/)
42. docker compose \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/reference/cli/docker/compose/](https://docs.docker.com/reference/cli/docker/compose/)
43. Docker Compose: Powering the Full App Lifecycle, дата последнего обращения: декабря 22, 2025, [https://www.docker.com/blog/docker-compose-powering-the-full-app-lifecycle/](https://www.docker.com/blog/docker-compose-powering-the-full-app-lifecycle/)
44. Using Docker Compose for Local Development | Ruan Martinelli, дата последнего обращения: декабря 22, 2025, [https://ruanmartinelli.com/blog/docker-compose-local-dev-environment/](https://ruanmartinelli.com/blog/docker-compose-local-dev-environment/)
