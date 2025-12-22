# **Отчет по Квесту 32.1: Архитектура и Реализация «Автоматизированной Кузницы» (Automated Forge) для ML-сервисов**

## **Аннотация**

В настоящем отчете представлен всесторонний, глубинным образом детализированный анализ технической реализации автоматизированного конвейера доставки (CI/CD) для сервисов машинного обучения (ML) в среде Google Cloud Platform (GCP). Исследование, метафорически обозначенное как «Квест 32.1: Воплощение Автоматизированной Кузницы», охватывает полный жизненный цикл создания инфраструктуры: от фундаментальных принципов управления идентификацией без использования долгоживущих ключей (Keyless Authentication) до нюансов контейнеризации тяжеловесных нейросетевых моделей и настройки систем наблюдаемости. Особое внимание уделяется преодолению ограничений среды исполнения GitHub Actions, архитектуре безопасности на базе Workload Identity Federation (WIF) и стратегии использования Google Artifact Registry (GAR). Документ предназначен для архитекторов облачных решений, DevOps-инженеров и MLOps-специалистов, стремящихся к построению безопасных, воспроизводимых и масштабируемых систем.

## ---

**Глава 1\. Эволюция парадигмы доступа: От статических ключей к Федерации Идентичностей**

### **1.1. Кризис управления секретами в эпоху облачных вычислений**

Исторически сложившаяся практика автоматизации взаимодействия между внешними CI/CD системами (такими как GitHub Actions, GitLab CI, Jenkins) и облачными провайдерами опиралась на использование сервисных аккаунтов. Стандартным методом аутентификации являлась генерация JSON-файла с закрытым ключом сервисного аккаунта (Service Account Key). Этот подход, будучи простым в реализации, породил системный кризис безопасности, известный как «проблема управления секретами».  
Статический ключ сервисного аккаунта в Google Cloud Platform (GCP) представляет собой криптографический артефакт, который, будучи выпущенным, обладает валидностью вплоть до 10 лет, если не будет отозван вручную. Файл ключа предоставляет предъявителю права доступа, закрепленные за сервисным аккаунтом, независимо от того, кто именно использует этот ключ — легитимный сборочный сервер или злоумышленник, получивший доступ к репозиторию.1  
Анализ инцидентов безопасности показывает, что утечка сервисных ключей через коммиты в публичные репозитории или небезопасные каналы связи является одним из наиболее распространенных векторов атак на облачную инфраструктуру. Фундаментальная проблема заключается в том, что ключ привязан к _владению_ («то, что у меня есть»), а не к _сущности_ («то, кто я есть»). Кроме того, ротация таких ключей требует сложной координации между администраторами IAM и разработчиками пайплайнов, что часто приводит к простоям систем (downtime) или отказу от ротации вовсе, что лишь усугубляет риски.2

### **1.2. Архитектура Workload Identity Federation (WIF)**

Ответом индустрии на эти вызовы стала технология Workload Identity Federation (WIF), которая реализует переход к модели «Zero Trust». В рамках Квеста 32.1 внедрение WIF является обязательным требованием, исключающим использование статических JSON-ключей.  
Суть WIF заключается в создании доверительных отношений между поставщиком облачных ресурсов (Google Cloud) и внешним поставщиком идентификации (Identity Provider — IdP), в роли которого выступает GitHub. Вместо хранения секрета, GitHub Actions в момент запуска workflow генерирует краткосрочный токен OpenID Connect (OIDC). Этот токен подписывается сертификатом GitHub и содержит набор утверждений (claims), описывающих контекст запуска: репозиторий, ветку git, имя workflow и инициатора запуска.4  
Процесс аутентификации трансформируется в последовательность шагов обмена токенами, где отсутствуют долгоживущие секреты:

1. **Генерация OIDC-токена:** Воркер GitHub Actions, имея разрешение id-token: write, запрашивает у провайдера GitHub токен JWT (JSON Web Token).
2. **Предъявление токена:** Экшен google-github-actions/auth передает этот токен в службу Google Security Token Service (STS).
3. **Валидация и Маппинг:** Google STS проверяет цифровую подпись токена, убеждаясь, что он действительно выпущен GitHub. Затем происходит сопоставление атрибутов токена с правилами доступа, настроенными в пуле WIF.6
4. **Имперсонация:** При успешной проверке STS выдает краткосрочный федеративный токен, который затем обменивается на токен доступа (access token) сервисного аккаунта GCP.
5. **Доступ:** Пайплайн выполняет операции от имени сервисного аккаунта, используя временный токен, который истекает автоматически (обычно через 1 час).3

### **1.3. Гранулярное управление доступом через Attribute Mapping**

Критическим преимуществом WIF перед ключами является возможность сверхточной настройки доступа. В то время как JSON-ключ дает доступ ко всему сервисному аккаунту, конфигурация WIF позволяет ограничить доступ на основе атрибутов репозитория.  
Механизм **Attribute Mapping** (сопоставление атрибутов) позволяет транслировать данные из OIDC-токена GitHub в атрибуты Google Cloud, которые затем используются в IAM-политиках. Язык Common Expression Language (CEL) позволяет создавать сложные логические конструкции для фильтрации доступа.4  
В таблице ниже приведено сравнение атрибутов, доступных для маппинга, и их влияние на безопасность пайплайна.  
**Таблица 1\. Атрибуты OIDC и стратегии ограничения доступа**

| Атрибут OIDC (GitHub) | Атрибут Google (Target)       | Описание и Стратегия безопасности                                                                                                                                                                                                                                               |
| :-------------------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| sub (Subject)         | google.subject                | Уникальный идентификатор, объединяющий репозиторий и ветку (например, repo:org/repo:ref:refs/heads/main). Маппинг по умолчанию. Позволяет привязать доступ к конкретной ветке конкретного репозитория.                                                                          |
| repository            | attribute.repository          | Имя репозитория (org/repo). Позволяет разрешить доступ для всех веток внутри одного проекта, но запретить доступ из форков.                                                                                                                                                     |
| repository_owner_id   | attribute.repository_owner_id | ID организации владельца. Критически важен для предотвращения атак типа "Confused Deputy", если злоумышленник создаст репозиторий с таким же именем в своей организации (хотя GitHub гарантирует уникальность путей, ID является более надежным неизменяемым идентификатором).7 |
| actor                 | attribute.actor               | Имя пользователя GitHub, запустившего workflow. Позволяет ограничить запуск деплоя только определенными инженерами, даже если у других есть права на запись в репозиторий.                                                                                                      |
| job_workflow_ref      | attribute.workflow_ref        | Ссылка на конкретный файл workflow. Гарантирует, что используется только утвержденный, прошедший ревью файл пайплайна, а не модифицированный злоумышленником скрипт.8                                                                                                           |

Использование условия assertion.repository_owner_id является важной практикой для защиты от подмены контекста, гарантируя, что токен пришел именно из авторизованной организации, а не из личного аккаунта разработчика с похожим названием репозитория.8

## ---

**Глава 2\. Инфраструктурный фундамент: Google Artifact Registry и IAM**

### **2.1. Миграция с Container Registry на Artifact Registry**

Вторым столпом «Автоматизированной Кузницы» является хранилище артефактов. Google Cloud активно выводит из эксплуатации устаревший Container Registry (GCR) в пользу Artifact Registry (GAR). Для Квеста 32.1 использование GAR является обязательным архитектурным решением.  
GAR предоставляет ряд критических преимуществ:

- **Региональность:** В отличие от мультирегионального GCR, GAR позволяет создавать репозитории в конкретных регионах (например, europe-west3). Это снижает задержки (latency) при скачивании образов сервисами, расположенными в том же регионе (Cloud Run, GKE), и уменьшает затраты на исходящий трафик (egress traffic).9
- **Управление доступом:** GAR поддерживает IAM-политики на уровне отдельных репозиториев, тогда как GCR управлялся на уровне бакетов Cloud Storage, что было менее прозрачно и гибко.
- **Мультиформатность:** Поддержка Docker, Maven, npm, Python и других форматов в едином интерфейсе управления.11

### **2.2. Создание репозитория и управление иммутабельностью**

Процесс создания репозитория через gcloud требует явного указания формата и политики иммутабельности тегов. Иммутабельность (Immutable Tags) — это концепция, запрещающая перезапись уже существующего тега образа. В контексте CI/CD это предотвращает ситуации, когда под тегом v1.0.0 внезапно оказывается другой код из\-за ошибки в пайплайне или злого умысла.

Bash

gcloud artifacts repositories create ml-forge-repo \\  
 \--project=my-project-id \\  
 \--location=europe-west3 \\  
 \--repository-format=docker \\  
 \--description="Docker repository for ML Service" \\  
 \--immutable-tags \\  
 \--async

Включение флага \--immutable-tags требует изменения стратегии тегирования в пайплайне: нельзя использовать тег latest для релизных артефактов, так как он по определению должен перезаписываться. Вместо этого следует использовать уникальные идентификаторы, такие как SHA-хеш коммита (github.sha).9

### **2.3. Принцип наименьших привилегий в IAM**

Для реализации безопасного пайплайна необходимо создать выделенный сервисный аккаунт (Service Account \- SA), который будет использоваться исключительно GitHub Actions. Этому SA не следует давать роль Owner или Editor.  
В соответствии с принципом наименьших привилегий, требуются следующие, строго ограниченные роли:

1. **roles/iam.workloadIdentityUser**: Назначается на сам сервисный аккаунт. Эта роль разрешает принципалу из пула WIF (то есть GitHub Action) "превращаться" (impersonate) в данный сервисный аккаунт.3
2. **roles/artifactregistry.writer**: Назначается на уровне конкретного репозитория GAR (или проекта, если требуется доступ ко всем репозиториям). Дает право загружать (push) образы. Важно не давать роль admin, чтобы пайплайн не мог удалять репозитории или менять политики.13
3. **roles/serviceusage.serviceUsageConsumer**: Необходима для проверки статуса API и корректного биллинга запросов, инициируемых сервисным аккаунтом.

## ---

**Глава 3\. Разработка Наблюдаемого ML-сервиса**

Ядром системы является микросервис, реализующий инференс модели машинного обучения. В рамках современной экосистемы Python стандартом де\-факто стал фреймворк **FastAPI**. Его асинхронная архитектура (ASGI) и нативная интеграция с Pydantic для валидации данных обеспечивают высокую производительность, критичную для ML-нагрузок.

### **3.1. Инструментация и метрики Prometheus**

В концепции «Автоматизированной Кузницы» сервис не может считаться готовым к эксплуатации ("production ready"), если он не обладает наблюдаемостью (observability). Простого логирования недостаточно; необходим сбор метрик в реальном времени.  
Для интеграции с системой мониторинга Prometheus используется библиотека prometheus-fastapi-instrumentator. Она позволяет автоматически экспортировать ключевые метрики производительности API по стандартному эндпоинту /metrics.14  
**Ключевые метрики для ML-сервиса:**

1. **http_requests_total (Counter):**
   - _Описание:_ Общее количество запросов к сервису.
   - _Лейблы:_ method (POST, GET), handler (/predict, /health), status (200, 422, 500).
   - _Инсайт:_ Резкий рост 4xx ошибок может указывать на изменение формата входных данных клиентами. Рост 500-х — на сбой внутри модели (например, OOM \- Out Of Memory).
2. **http_request_duration_seconds (Histogram):**
   - _Описание:_ Распределение времени обработки запросов.
   - _Важность для ML:_ Время инференса модели часто варьируется в зависимости от длины входного текста или размера изображения. Гистограмма позволяет рассчитывать перцентили (P95, P99), которые являются реальными показателями качества обслуживания (SLA).15
3. **Кастомные метрики (Business Logic Metrics):**
   - Помимо стандартных HTTP-метрик, ML-сервис должен экспортировать специфические показатели, например, длину входного промпта (в токенах) или уверенность модели (confidence score). Это позволяет отслеживать дрейф данных (Data Drift) еще до того, как он скажется на бизнес-показателях.

Пример реализации middleware для перехвата запросов и обновления метрик демонстрирует простоту интеграции, при которой Instrumentator оборачивает приложение FastAPI, добавляя необходимые хуки без изменения бизнес-логики эндпоинтов.16

### **3.2. Паттерны A/B тестирования и Canary Releases**

Наличие метрик открывает возможности для продвинутых стратегий развертывания. Использование метки версии модели (например, model_version="v2.1") в метриках Prometheus позволяет реализовать A/B тестирование на уровне инфраструктуры.  
В сценарии, где одновременно запущены две версии сервиса (Stable и Canary), балансировщик нагрузки распределяет трафик. Система мониторинга (Grafana/Prometheus) агрегирует метрики http_request_duration_seconds отдельно для каждой версии. Если новая версия показывает деградацию производительности (latency увеличилась на 20%), система CD может автоматически остановить раскатку. Этот подход превращает метрики из средства диагностики в управляющий сигнал для пайплайна.18

## ---

**Глава 4\. Искусство Контейнеризации Тяжелых Нагрузок**

Контейнеризация ML-приложений кардинально отличается от упаковки обычных веб\-сервисов. Основная причина — огромный размер артефактов. Библиотеки глубокого обучения (PyTorch, TensorFlow) и веса моделей (Hugging Face Transformers) могут занимать десятки гигабайт.

### **4.1. Стратегия управления весами: Baking vs. Runtime**

Существует дилемма: включать веса модели внутрь Docker-образа («запекание» \- baking) или скачивать их при старте контейнера (runtime download).

- **Runtime Download:** Образ остается легким (только код и библиотеки). При старте скрипт выкачивает модель с S3 или Hugging Face Hub.
  - _Риски:_ Контейнер может не запуститься, если внешний сервис недоступен или скорость сети упала. Время холодного старта (cold start) становится непредсказуемым. Отсутствует гарантия воспроизводимости (веса во внешнем хранилище могли измениться).19
- **Baking (Рекомендуемый подход для Кузницы):** Веса скачиваются на этапе сборки (docker build) и становятся частью файловой системы образа.
  - _Преимущества:_ Образ самодостаточен, неизменяем и может работать в полностью изолированном контуре (air-gapped). Старт мгновенный (данные уже на диске).
  - _Сложности:_ Огромный размер образа усложняет пуш/пул и требует оптимизации слоев.20

Для реализации стратегии «запекания» эффективно использовать скрипты-прелоадеры на Python. Вместо того чтобы просто копировать файлы, можно запустить скрипт с использованием snapshot_download из библиотеки huggingface_hub внутри RUN инструкции Dockerfile.  
Критически важный нюанс кэширования слоев:  
Инструкции в Dockerfile должны быть упорядочены от наименее изменяемых к наиболее часто изменяемым.

1. Установка системных зависимостей (apt-get).
2. Установка тяжелых библиотек Python (PyTorch, Transformers).
3. Скачивание весов модели (меняется редко).
4. Копирование кода приложения (COPY. /app — меняется часто).

Если поменять местами п.3 и п.4, то любое изменение в коде (app.py) приведет к инвалидации кэша и повторному скачиванию гигабайтов весов модели, что катастрофически замедлит CI/CD пайплайн.19

### **4.2. Кросс-платформенные конфликты: Проблема CRLF и Архитектур**

Разработка ML-сервисов часто ведется на машинах с Windows, а деплой происходит в Linux-контейнеры. Это порождает две классические проблемы, способные остановить пайплайн.  
1\. Ошибка формата исполняемого файла (exec format error) из\-за окончаний строк.  
Скрипты shell (.sh), созданные в Windows, используют символы возврата каретки CRLF (\\r\\n). Linux ожидает LF (\\n). При попытке запустить такой скрипт в контейнере (например, entrypoint.sh), интерпретатор Bash видит символ \\r как часть имени команды или пути, что приводит к загадочным ошибкам no such file or directory или exec format error.21

- _Решение:_ Использование файла .gitattributes в репозитории для принудительной нормализации окончаний строк (\* text=auto eol=lf) или запуск утилиты dos2unix внутри Dockerfile перед исполнением скриптов.

2\. Архитектурный конфликт (amd64 vs arm64).  
Разработчики, использующие Apple Silicon (M1/M2/M3), по умолчанию собирают Docker-образы архитектуры linux/arm64. Облачные среды (Cloud Run) и стандартные раннеры GitHub работают на архитектуре linux/amd64. Попытка запустить ARM-образ на AMD-сервере приведет к сбою exec user process caused: exec format error.21

- _Решение:_ Использование Docker Buildx с явным указанием целевой платформы \--platform linux/amd64 в пайплайне сборки. Это гарантирует совместимость независимо от того, где была запущена сборка локально.13

## ---

**Глава 5\. Конструирование CI/CD Пайплайна в GitHub Actions**

Пайплайн GitHub Actions — это конвейерная лента «Автоматизированной Кузницы». Он объединяет все вышеописанные компоненты в единый процесс.

### **5.1. Управление ресурсами раннера и проблема дискового пространства**

Стандартные hosted-раннеры GitHub (ubuntu-latest) предоставляют ограниченные ресурсы: 2 ядра CPU и около 14 ГБ доступного дискового пространства (из общих 84 ГБ, большая часть занята предустановленным ПО).  
Для ML-задач это критическое "бутылочное горлышко". Сборка Docker-образа с PyTorch и весами модели требует места для:

1. Базовых слоев (NVIDIA CUDA runtime).
2. Контекста сборки.
3. Промежуточных слоев с установленными пакетами.
4. Финального образа.

Суммарно это часто превышает 20 ГБ, что приводит к ошибке No space left on device в середине процесса сборки.23  
Стратегия освобождения пространства:  
Необходимо добавить в пайплайн шаг предварительной очистки ("Deep Clean"). Существуют готовые решения, но ручной скрипт дает больше контроля. Следует удалить массивные, неиспользуемые директории:

- Android SDK (/usr/local/lib/android)
- .NET Framework (/usr/share/dotnet)
- Компиляторы Haskell и CodeQL.  
  Это позволяет высвободить до 20 ГБ дополнительного пространства, что является решающим фактором для успеха сборки ML-образа без перехода на платные (Larger) раннеры.25

### **5.2. Анатомия Workflow-файла**

Ниже приведен детальный разбор ключевых секций YAML-конфигурации, реализующей безопасный и эффективный пайплайн.  
**Секция прав доступа (Permissions):**

YAML

permissions:  
 contents: read \# Чтение кода репозитория  
 id-token: write \# КРИТИЧНО: Разрешает запрос OIDC-токена для WIF

Без id-token: write шаг аутентификации в Google Cloud немедленно завершится ошибкой, так как GitHub не выдаст JWT токен.27  
Аутентификация (Google Auth Action):  
Используется экшен google-github-actions/auth@v2.

YAML

\- name: Authenticate to Google Cloud  
 id: auth  
 uses: google-github-actions/auth@v2  
 with:  
 workload_identity_provider: 'projects/123456789/locations/global/workloadIdentityPools/my-pool/providers/my-provider'  
 service_account: 'my-service-account@my-project.iam.gserviceaccount.com'  
 token_format: 'access_token' \# Необходимо для последующего логина в Docker

Параметр token_format: 'access_token' инструктирует экшен выдать не просто файл credentials, а raw OAuth2 токен, который можно использовать как пароль.13  
Логин в Docker Registry:  
В отличие от старых методов, здесь не используется gcloud auth configure-docker. Используется стандартный docker/login-action.

YAML

\- name: Login to Artifact Registry  
 uses: docker/login-action@v3  
 with:  
 registry: europe-west3-docker.pkg.dev  
 username: oauth2accesstoken  
 password: ${{ steps.auth.outputs.access\_token }}

Имя пользователя всегда фиксировано: oauth2accesstoken. Паролем служит токен, полученный на предыдущем шаге из outputs.13

### **5.3. Продвинутое кэширование Docker-слоев**

Для ускорения сборки необходимо использовать возможности кэширования GitHub Actions API. Экшен docker/build-push-action поддерживает специальный бэкенд gha.

YAML

\- name: Build and Push  
 uses: docker/build-push-action@v6  
 with:  
 context:.  
 push: true  
 tags: europe-west3-docker.pkg.dev/my-project/my-repo/image:${{ github.sha }}  
 cache-from: type=gha  
 cache-to: type=gha,mode=max

Режим mode=max заставляет кэшировать не только финальные слои, но и все промежуточные. Это критично для ML-сборок: если изменился только код приложения, слой с установленным PyTorch и скачанной моделью будет взят из кэша GitHub, сокращая время сборки с 15 минут до 30 секунд.13

## ---

**Глава 6\. Операционная Надежность и Траблшутинг**

### **6.1. Диагностика ошибок WIF**

Одной из самых частых проблем при настройке WIF является ошибка 403 Forbidden или Subject token validation failed. Часто это связано с несовпадением значений в Attribute Mapping и IAM binding.

- _Симптом:_ Пайплайн падает на шаге google-github-actions/auth.
- _Причина:_ В IAM политике указан subject repo:org/repo:ref:refs/heads/main, а workflow запущен из ветки dev.
- _Диагностика:_ Необходимо проверить логи шага аутентификации. Google Cloud часто скрывает детали ошибки безопасности, поэтому проверку лучше начинать с валидации значений sub в самом токене GitHub (можно вывести payload токена в лог в debug режиме, но осторожно) и сверки с конфигурацией провайдера через gcloud iam workload-identity-pools providers describe.28

### **6.2. Безопасность цепочки поставок**

Использование WIF решает проблему аутентификации, но не валидации кода. Для повышения безопасности «Автоматизированной Кузницы» рекомендуется внедрить подпись образов (Image Signing) с использованием Sigstore/Cosign. Это позволяет гарантировать, что образ в GAR действительно был собран в доверенном пайплайне GitHub Actions, а не загружен злоумышленником, укравшим токен. Интеграция Cosign в пайплайн использует тот же OIDC токен GitHub для генерации эфемерных ключей подписи (Keyless Signing), идеологически продолжая концепцию WIF.

## ---

**Заключение**

Реализация Квеста 32.1 трансформирует процесс доставки ML-решений из кустарного ремесла в индустриальный стандарт. Отказ от сервисных ключей в пользу Workload Identity Federation устраняет корневую причину множества инцидентов безопасности. Переход на Artifact Registry и использование региональных репозиториев оптимизирует логистику данных. Глубокая инструментация FastAPI-сервисов через Prometheus делает их прозрачными для эксплуатации.  
Совокупность этих практик формирует «Автоматизированную Кузницу» — среду, где инфраструктура не требует ручного вмешательства, безопасность встроена в дизайн (Security by Design), а разработчики могут сосредоточиться на улучшении качества моделей, доверив их доставку надежному, наблюдаемому и защищенному конвейеру. Техническая сложность начальной настройки WIF и Docker-оптимизаций многократно окупается стабильностью и скоростью итераций в долгосрочной перспективе.

#### **Источники**

1. The importance of Workload Identity Federation Over Service Account Keys \- Onix, дата последнего обращения: декабря 22, 2025, [https://www.onixnet.com/blog/the-importance-of-workload-identity-federation-over-service-account-keys/?utm_campaign=11737946-Blog%20Share\&utm_content=342637652\&utm_medium=social\&utm_source=facebook\&hss_channel=fbp-157737364257661](https://www.onixnet.com/blog/the-importance-of-workload-identity-federation-over-service-account-keys/?utm_campaign=11737946-Blog+Share&utm_content=342637652&utm_medium=social&utm_source=facebook&hss_channel=fbp-157737364257661)
2. Enabling keyless authentication from GitHub Actions | Google Cloud Blog, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/blog/products/identity-security/enabling-keyless-authentication-from-github-actions](https://cloud.google.com/blog/products/identity-security/enabling-keyless-authentication-from-github-actions)
3. Goodbye, Service Account Keys\! Secure Your GCP access & go keyless with Workload Identity Federation | by Saloni Patidar | Google Cloud \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/google-cloud/goodbye-service-account-keys-e009f3b8ffef](https://medium.com/google-cloud/goodbye-service-account-keys-e009f3b8ffef)
4. Notes on Workload Identity Federation from GitHub Actions to Google Cloud Platform, дата последнего обращения: декабря 22, 2025, [https://medium.com/@bbeesley/notes-on-workload-identity-federation-from-github-actions-to-google-cloud-platform-7a818da2c33e](https://medium.com/@bbeesley/notes-on-workload-identity-federation-from-github-actions-to-google-cloud-platform-7a818da2c33e)
5. OpenID Connect \- GitHub Docs, дата последнего обращения: декабря 22, 2025, [https://docs.github.com/en/actions/concepts/security/openid-connect](https://docs.github.com/en/actions/concepts/security/openid-connect)
6. Workload Identity Federation | Identity and Access Management (IAM) | Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/iam/docs/workload-identity-federation](https://docs.cloud.google.com/iam/docs/workload-identity-federation)
7. OpenID Connect reference \- GitHub Docs, дата последнего обращения: декабря 22, 2025, [https://docs.github.com/actions/reference/openid-connect-reference](https://docs.github.com/actions/reference/openid-connect-reference)
8. GitHub Actions: OpenID Connect token now supports more claims for configuring granular cloud access, дата последнего обращения: декабря 22, 2025, [https://github.blog/changelog/2023-01-10-github-actions-openid-connect-token-now-supports-more-claims-for-configuring-granular-cloud-access/](https://github.blog/changelog/2023-01-10-github-actions-openid-connect-token-now-supports-more-claims-for-configuring-granular-cloud-access/)
9. Create standard repositories | Artifact Registry \- Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/artifact-registry/docs/repositories/create-repos](https://docs.cloud.google.com/artifact-registry/docs/repositories/create-repos)
10. Help with gcloud artifacts docker upgrade migrate: Repository Not Found and No Images Copied \- Google Developer forums, дата последнего обращения: декабря 22, 2025, [https://discuss.google.dev/t/help-with-gcloud-artifacts-docker-upgrade-migrate-repository-not-found-and-no-images-copied/179861](https://discuss.google.dev/t/help-with-gcloud-artifacts-docker-upgrade-migrate-repository-not-found-and-no-images-copied/179861)
11. Create remote repositories | Artifact Registry \- Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/artifact-registry/docs/repositories/remote-repo](https://docs.cloud.google.com/artifact-registry/docs/repositories/remote-repo)
12. Push code with GitHub Actions to Google Cloud's Artifact Registry \- Roger Martinez, дата последнего обращения: декабря 22, 2025, [https://roger-that-dev.medium.com/push-code-with-github-actions-to-google-clouds-artifact-registry-60d256f8072f](https://roger-that-dev.medium.com/push-code-with-github-actions-to-google-clouds-artifact-registry-60d256f8072f)
13. Push Images To Artifact Registry Using GitHub Actions & Workload ..., дата последнего обращения: декабря 22, 2025, [https://dev.to/filip-lindqvist/google-cloud-artifact-registry-with-github-actions-using-workload-identity-2h8c](https://dev.to/filip-lindqvist/google-cloud-artifact-registry-with-github-actions-using-workload-identity-2h8c)
14. trallnag/prometheus-fastapi-instrumentator \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/trallnag/prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
15. View pypi: prometheus-fastapi-instrumentator | OpenText Core SCA \- Debricked, дата последнего обращения: декабря 22, 2025, [https://debricked.com/select/package/pypi-prometheus-fastapi-instrumentator](https://debricked.com/select/package/pypi-prometheus-fastapi-instrumentator)
16. Getting Started: Monitoring a FastAPI App with Grafana and Prometheus \- A Step-by-Step Guide \- DEV Community, дата последнего обращения: декабря 22, 2025, [https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-grafana-and-prometheus-a-step-by-step-guide-3fbn](https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-grafana-and-prometheus-a-step-by-step-guide-3fbn)
17. Monitoring FastAPI with Grafana \+ Prometheus: A 5-Minute Guide \- Level Up Coding, дата последнего обращения: декабря 22, 2025, [https://levelup.gitconnected.com/monitoring-fastapi-with-grafana-prometheus-a-5-minute-guide-658280c7f358](https://levelup.gitconnected.com/monitoring-fastapi-with-grafana-prometheus-a-5-minute-guide-658280c7f358)
18. PROMETHEUS Metrics for your Python FastAPI App \- YouTube, дата последнего обращения: декабря 22, 2025, [https://www.youtube.com/watch?v=WWzl53ObYvo](https://www.youtube.com/watch?v=WWzl53ObYvo)
19. \[D\] Best practices to dockerize hugginface hub models : r/MachineLearning \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/MachineLearning/comments/13jud83/d_best_practices_to_dockerize_hugginface_hub/](https://www.reddit.com/r/MachineLearning/comments/13jud83/d_best_practices_to_dockerize_hugginface_hub/)
20. Manually Downloading Models in docker build with snapshot_download \- Transformers, дата последнего обращения: декабря 22, 2025, [https://discuss.huggingface.co/t/manually-downloading-models-in-docker-build-with-snapshot-download/19637](https://discuss.huggingface.co/t/manually-downloading-models-in-docker-build-with-snapshot-download/19637)
21. Docker exec format error \- How do we fix it? \- Bobcares, дата последнего обращения: декабря 22, 2025, [https://bobcares.com/blog/docker-exec-format-error/](https://bobcares.com/blog/docker-exec-format-error/)
22. How to Fix "exec user process caused: exec format error" in Linux | Beebom, дата последнего обращения: декабря 22, 2025, [https://beebom.com/how-fix-exec-user-process-caused-exec-format-error-linux/](https://beebom.com/how-fix-exec-user-process-caused-exec-format-error-linux/)
23. Github Actions Server's total space is very low for AI ML Docker building \! : r/devops \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/devops/comments/18q10co/github_actions_servers_total_space_is_very_low/](https://www.reddit.com/r/devops/comments/18q10co/github_actions_servers_total_space_is_very_low/)
24. \[error\] No space left on device · community · Discussion \#25678 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/orgs/community/discussions/25678](https://github.com/orgs/community/discussions/25678)
25. Increasing GitHub Actions Disk Space \- Carlos Becker, дата последнего обращения: декабря 22, 2025, [https://carlosbecker.com/posts/github-actions-disk-space/](https://carlosbecker.com/posts/github-actions-disk-space/)
26. GitHub Actions Docker Service Container \>25GB Cannot Be Loaded \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/76294509/github-actions-docker-service-container-25gb-cannot-be-loaded](https://stackoverflow.com/questions/76294509/github-actions-docker-service-container-25gb-cannot-be-loaded)
27. google-github-actions/auth: A GitHub Action for ... \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/google-github-actions/auth](https://github.com/google-github-actions/auth)
28. Github Actions OIDC \- what are all the possible values of the "sub" claim on the token (for creating trust policy) \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/76784270/github-actions-oidc-what-are-all-the-possible-values-of-the-sub-claim-on-the](https://stackoverflow.com/questions/76784270/github-actions-oidc-what-are-all-the-possible-values-of-the-sub-claim-on-the)
