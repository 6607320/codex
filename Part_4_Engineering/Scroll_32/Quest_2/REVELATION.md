# **Отчет по архитектурной реализации системной телеметрии, управлению артефактами и обеспечению безопасности в среде непрерывной интеграции «Кузница»**

## **Аннотация**

В настоящем документе представлен исчерпывающий технический анализ и архитектурное руководство по реализации задачи «Квест 32.2: Установка "Телеметрии" в Кузницу». Под термином «Кузница» (Forge) в рамках данной инженерной инициативы понимается высокопроизводительный конвейер непрерывной интеграции и доставки (CI/CD), обеспечивающий сборку, тестирование и развертывание программного обеспечения. Отчет детализирует переход от базовой автоматизации к наблюдаемой, безопасной и оптимизированной экосистеме.  
Ключевыми направлениями исследования являются: внедрение метрик покрытия кода (Code Coverage) с жесткими критериями качества (Quality Gates), архитектурная миграция механизмов управления артефактами на версию v4 в GitHub Actions, обеспечение безопасности через федеративную идентификацию (Workload Identity Federation) для взаимодействия с Google Cloud Platform (GCP), а также оптимизация ресурсов среды выполнения и прикладная инструментация сервисов посредством Prometheus. Анализ базируется на современных практиках DevOps, требованиях безопасности Supply Chain и принципах SRE (Site Reliability Engineering).

## ---

**Глава 1\. Концептуальные основы наблюдаемости в производственном конвейере**

### **1.1. Эволюция парадигмы: от логгирования к телеметрии**

В традиционных моделях разработки программного обеспечения понятие «телеметрия» часто ограничивалось сбором логов приложений в продуктивной среде. Однако в контексте современной инженерии («Кузницы»), телеметрия охватывает весь жизненный цикл поставки кода. Задача 32.2 подразумевает внедрение инструментов, которые превращают CI/CD пайплайн из «черного ящика» в прозрачную систему, предоставляющую количественные данные о своем состоянии, эффективности и надежности.1  
Наблюдаемость (Observability) в пайплайне не является пассивным процессом. Это активный механизм, влияющий на принятие решений: остановить ли сборку, если покрытие тестами упало на 0.5%? Как долго хранить артефакты неудачных сборок для эффективной отладки? Каким образом обеспечить аутентификацию во внешних облачных системах без компрометации долгоживущих секретов? Ответы на эти вопросы лежат в плоскости интеграции специализированных инструментов, таких как pytest-cov, actions/upload-artifact и Google Workload Identity Federation.

### **1.2. Стратегическое значение разделения данных и артефактов**

Фундаментальным принципом построения надежной системы является разделение потоков данных:

1. **Логи выполнения (Execution Logs):** Текстовый вывод консоли, необходимый для оперативного анализа причин падения. Они эфемерны и часто трудночитаемы при больших объемах.3
2. **Артефакты (Artifacts):** Структурированные файлы (отчеты XML/HTML, бинарные файлы, дампы), которые сохраняются для глубокого анализа, аудита и последующих этапов развертывания.
3. **Метрики (Metrics):** Числовые показатели (процент покрытия, время сборки, потребление диска), позволяющие отслеживать тренды во времени.

Эффективная реализация «Телеметрии» требует настройки каждого из этих потоков. Как показывает практика, отсутствие структурированных артефактов делает отладку распределенных систем практически невозможной, заставляя инженеров полагаться на интуицию, а не на факты.2

## ---

**Глава 2\. Архитектура тестирования и анализа покрытия кода**

Центральным элементом обеспечения качества в Python-экосистеме является связка фреймворка pytest и плагина pytest-cov. Однако их интеграция в CI/CD требует учета множества нюансов, связанных с конфигурацией, форматами отчетов и пороговыми значениями.

### **2.1. Глубокая конфигурация pytest-cov**

Инструмент pytest-cov предоставляет интерфейс командной строки для движка coverage.py, позволяя запускать тесты с одновременным анализом исполняемых строк кода. Для серверов непрерывной интеграции критически важно генерировать отчеты в машиночитаемых форматах.  
Стандартный паттерн запуска выглядит следующим образом:  
pytest \--cov-report=xml:coverage.xml \--cov=src tests/.5  
Здесь флаг \--cov-report=xml инструктирует систему создать файл coverage.xml, который является стандартом де\-факто для интеграции с внешними системами анализа (SonarQube, Codecov) и инструментами визуализации в GitHub Actions.5 Флаг \--cov=src явно указывает директорию исходного кода, подлежащую анализу. Это важно, так как без явного указания coverage.py может включить в отчет сторонние библиотеки или тестовые файлы, искажая метрики.

#### **Иерархия конфигурации и управление зависимостями**

Одной из частых проблем при настройке является ошибка unrecognized arguments: \--cov. Это свидетельствует о том, что плагин pytest-cov не установлен в окружении, где запускается pytest. В рамках CI пайплайна управление зависимостями должно быть строгим. Рекомендуется использовать файлы манифестов (pyproject.toml или requirements.txt) с фиксацией версий:

- pytest\>=8.0.0
- pytest-cov\>=5.0.0.7

Существует сложная иерархия приоритетов конфигурации. Плагин pytest-cov может считывать настройки из .coveragerc, setup.cfg или pyproject.toml. Однако, как отмечается в документации, аргументы командной строки имеют приоритет. Например, если в конфигурационном файле задана опция source, но в командной строке передан аргумент \--cov=package_name, значение из файла будет переопределено. Это может привести к неочевидному поведению, когда исключения (omissions), настроенные в файле, игнорируются при запуске с флагами.8

### **2.2. Механизм Quality Gates: \--cov-fail-under**

Автоматическое обеспечение качества кода реализуется через механизм «Quality Gate» (Ворота качества). Плагин pytest-cov предоставляет аргумент \--cov-fail-under=MIN, который заставляет процесс завершиться с ненулевым кодом возврата, если общее покрытие ниже заданного порога.9  
При внедрении этого механизма инженеры сталкиваются с рядом технических особенностей:  
Проблема округления чисел с плавающей точкой:  
Существуют задокументированные случаи, когда пайплайн успешно проходит проверку, даже если фактическое покрытие ниже заявленного порога. Например, при требовании 67% и фактическом покрытии 66.666%, система может округлить значение до 67% и пропустить сборку, или же, наоборот, отображать сообщение о неудаче, но возвращать код выхода 0\. Исследования показывают, что это связано с внутренней логикой округления coverage.py. Для решения этой проблемы рекомендуется использовать конфигурационные файлы для задания точности (precision) или анализировать XML-отчет внешними скриптами для принятия решения о статусе сборки.11  
Проблема модульного импорта:  
Попытка передать список модулей через запятую в аргументе \--cov (например, \--cov=module1,module2) часто приводит к ошибке CoverageWarning: Module module1,module2 was never imported. Это происходит потому, что интерпретатор воспринимает строку как имя одного модуля. Корректный подход заключается в многократном использовании флага \--cov или настройке путей в конфигурационном файле.11

### **2.3. Визуализация и обратная связь (Feedback Loops)**

«Телеметрия» эффективна только тогда, когда ее данные видны разработчику. Использование «сырых» логов консоли для проверки покрытия неэффективно. Современный подход предполагает интеграцию отчетов непосредственно в Pull Request (PR).  
Действие MishaKav/pytest-coverage-comment позволяет парсить сгенерированные файлы pytest-coverage.txt (вывод терминала) и pytest.xml (JUnit XML) для создания форматированного комментария в PR.7

| Параметр                 | Описание                          | Значение по умолчанию |
| :----------------------- | :-------------------------------- | :-------------------- |
| pytest-coverage-path     | Путь к текстовому выводу покрытия | ./pytest-coverage.txt |
| junitxml-path            | Путь к XML отчету о тестах        | Нет (Опционально)     |
| pytest-xml-coverage-path | Путь к XML отчету о покрытии      | Нет (для бейджей)     |

**Рабочий процесс (Workflow):**

1. **Запуск тестов:** Команда должна использовать утилиту tee для одновременного вывода в консоль и записи в файл: pytest \--cov=src tests/ | tee pytest-coverage.txt.
2. **Генерация комментария:** Action считывает этот файл и публикует таблицу с дифференциалом покрытия (насколько изменился процент по сравнению с основной веткой).7

Этот подход реализует принцип «Shift Left», предоставляя разработчику информацию о качестве кода до момента слияния ветки.

## ---

**Глава 3\. Управление артефактами: Миграция на v4 и стратегии сохранения**

Сборка логов и артефактов в «Кузнице» осуществляется посредством механизма GitHub Actions Artifacts. В 2024 году произошел значительный технологический сдвиг с выходом версии v4 действий upload-artifact и download-artifact.

### **3.1. Архитектурные изменения в upload-artifact@v4**

Версия v4 привнесла критические изменения, несовместимые с предыдущими версиями (Breaking Changes). Главным из них является **неизменяемость артефактов (Immutability)**. В версиях v1-v3 допускалась дозапись данных в существующий артефакт или загрузка нескольких файлов из разных джоб (jobs) в артефакт с одним и тем же именем. В v4 это вызывает ошибку. Теперь каждый артефакт должен иметь уникальное имя.13  
Последствия для матричных сборок:  
Если пайплайн использует стратегию matrix для параллельного запуска тестов (например, на разных версиях Python), каждая джоба должна генерировать уникальное имя артефакта.

- _Неправильно:_ name: test-results
- _Правильно:_ name: test-results-${{ matrix.python-version }}.14

Кроме того, v4 значительно повысила производительность загрузки и внесла улучшения в безопасность, включая автоматическую генерацию и валидацию SHA256 дайджестов при скачивании, что гарантирует целостность данных.15

### **3.2. Стратегия безусловного сохранения (if: always())**

Одной из самых распространенных ошибок в конфигурации CI/CD является потеря телеметрии при сбое тестов. Стандартное поведение GitHub Actions — прерывание выполнения джобы при ненулевом коде возврата любого шага. Это означает, что если шаг pytest обнаружит ошибку, шаг upload-artifact, идущий следом, не выполнится, и разработчик останется без отчетов для анализа.  
Для решения этой проблемы необходимо использовать условное выражение if: always().  
**Сценарий использования:**

YAML

\- name: Run Tests  
 run: pytest \--junitxml=results.xml

\- name: Upload Artifacts  
 if: always() \# Выполняется даже при падении тестов  
 uses: actions/upload-artifact@v4  
 with:  
 name: test-report  
 path: results.xml  
 retention-days: 5

Конструкция always() гарантирует выполнение шага независимо от статуса предыдущих шагов (success, failure, cancelled). Однако здесь есть нюанс: если рабочий процесс был отменен пользователем вручную (cancelled), шаг с always() все равно попытается выполниться. В некоторых случаях это нежелательно (например, зачем загружать артефакты, если я отменил сборку, потому что заметил опечатку?). Для более тонкой настройки можно использовать комбинацию \`if: success() |  
| failure(), которая исключает статус cancelled\`.17

### **3.3. Локализация и контекст файловой системы**

При работе с артефактами важно понимать, как GitHub Actions управляет рабочими директориями. Каждый шаг run сбрасывает рабочую директорию в GITHUB_WORKSPACE, если не указано иное. Однако действие upload-artifact работает в контексте корня репозитория.  
Частая ошибка — попытка загрузить файл, созданный в подпапке, без указания полного пути или неправильное использование working-directory с ключевым словом uses. В отличие от шагов run, uses не поддерживает working-directory. Пути к файлам в upload-artifact всегда должны быть относительны корню репозитория или абсолютными.21

## ---

**Глава 4\. Федеративная идентификация и безопасность (Workload Identity Federation)**

Внедрение «Телеметрии» часто требует взаимодействия с внешними облачными провайдерами, такими как Google Cloud Platform (GCP), для хранения долгосрочных логов, образов контейнеров или аналитики. Традиционный метод аутентификации с использованием JSON-ключей сервисных аккаунтов (Service Account Keys) представляет собой значительный риск безопасности.

### **4.1. Проблема долгоживущих секретов**

Ключи сервисных аккаунтов не имеют срока действия по умолчанию. Если такой ключ случайно попадет в репозиторий кода (commit leak) или будет перехвачен в логах, злоумышленник получит постоянный доступ к облачным ресурсам до тех пор, пока ключ не будет отозван вручную. Управление ротацией этих ключей создает дополнительную операционную нагрузку.22

### **4.2. Workload Identity Federation (WIF): Бесключевой доступ**

Решением является **Workload Identity Federation (WIF)**. Этот механизм позволяет GitHub Actions аутентифицироваться в GCP, используя временные токены OpenID Connect (OIDC), подписанные GitHub.  
**Механизм работы:**

1. **Trust:** В GCP создается Пул идентификации рабочей нагрузки (Workload Identity Pool) и Провайдер (Provider), настроенный на доверие токенам, выпущенным https://token.actions.githubusercontent.com.
2. **Exchange:** Во время выполнения workflow, GitHub Actions запрашивает OIDC токен у GitHub.
3. **Access:** Действие google-github-actions/auth отправляет этот токен в GCP Security Token Service (STS), который обменивает его на короткоживущий токен доступа GCP (федеративный токен или OAuth 2.0 access token).23

### **4.3. Практическая реализация и Attribute Mapping**

Настройка WIF требует тщательного маппинга атрибутов (Attribute Mapping). Это процесс сопоставления полей OIDC токена GitHub (например, имя репозитория, ветка, актор) с атрибутами Google Cloud, которые можно использовать для ограничения доступа.  
**Пример конфигурации через gcloud CLI:**

Bash

\# Создание пула  
gcloud iam workload-identity-pools create "github-pool" \\  
 \--project="${PROJECT_ID}" \--location="global" \--display-name="GitHub Actions Pool"

\# Создание провайдера с маппингом  
gcloud iam workload-identity-pools providers create-oidc "github-provider" \\  
 \--project="${PROJECT_ID}" \--location="global" \\  
 \--workload-identity-pool="github-pool" \\  
 \--issuer-uri="https://token.actions.githubusercontent.com" \\  
 \--attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \\  
 \--attribute-condition="assertion.repository_owner \== 'MyOrgName'"

Критически важен параметр \--attribute-condition. Без него любой репозиторий на GitHub теоретически мог бы попытаться аутентифицироваться, если знает ID вашего пула (хотя ему все равно потребуется привязка к Service Account). Условие assertion.repository_owner ограничивает доступ только репозиториями вашей организации.23  
В GitHub Workflow интеграция выглядит следующим образом:

YAML

permissions:  
 id-token: write \# Необходимо для запроса OIDC токена  
 contents: read

steps:  
\- uses: google-github-actions/auth@v2  
 with:  
 workload_identity_provider: 'projects/123456789/locations/global/workloadIdentityPools/github-pool/providers/github-provider'  
 service_account: 'my-service-account@my-project.iam.gserviceaccount.com'

После успешной аутентификации, все последующие шаги (использующие gcloud или клиентские библиотеки Google) будут автоматически использовать полученные учетные данные.26

## ---

**Глава 5\. Оптимизация среды выполнения: Управление дисковым пространством**

При масштабной сборке, особенно с использованием Docker и больших наборов данных для тестов, стандартные раннеры ubuntu-latest часто исчерпывают доступное дисковое пространство. По умолчанию они предоставляют около 14 ГБ свободного места, так как большая часть диска (всего около 72 ГБ) занята предустановленным инструментарием.28

### **5.1. Анализ состава образа Runner**

Образ ubuntu-latest — это «швейцарский нож», содержащий инструменты для большинства популярных языков: Java (несколько JDK),.NET, Android SDK, Haskell, CodeQL, Docker images (alpine, node и др.). Для Python-проекта, реализующего задачу 32.2, 80% этих инструментов являются «мертвым грузом», занимающим десятки гигабайт.28

### **5.2. Стратегии очистки: rm против rmz и автоматизация**

Существует два подхода к освобождению места:

1. **Агрессивная ручная очистка:** Использование shell-скриптов для удаления директорий.
   - /usr/local/lib/android (\~10 ГБ)
   - /usr/share/dotnet (\~3 ГБ)
   - /opt/ghc (Haskell, \~6 ГБ).29
2. Использование специализированных Actions:  
   Действия, такие как jlumbroso/free-disk-space или insightsengineering/disk-space-reclaimer, автоматизируют этот процесс.

Сравнительный анализ производительности:  
Очистка не является бесплатной операцией — она занимает время сборки. Удаление файлов через стандартный rm может быть медленным из\-за огромного количества мелких файлов в SDK.

- Удаление Android SDK через rm: \~54 секунды.
- Удаление Android SDK через rmz (оптимизированный инструмент удаления): \~13 секунд.
- Полная очистка всех инструментов может занять до 4-5 минут, но освобождает более 30 ГБ.29

**Рекомендация:** Для оптимизации CI/CD следует применять выборочную очистку, удаляя только самые тяжелые и неиспользуемые компоненты (Android, Haskell, CodeQL), избегая полной зачистки (docker-images), если это не критично, чтобы не увеличивать время инициализации пайплайна.31

## ---

**Глава 6\. Прикладная телеметрия: Инструментация сервисов (FastAPI \+ Prometheus)**

Установка «Телеметрии» в «Кузницу» была бы неполной без внедрения метрик внутрь самого приложения. В экосистеме Python стандартом для микросервисов является FastAPI, а для сбора метрик — Prometheus.

### **6.1. Паттерны инструментации**

Библиотека prometheus-fastapi-instrumentator предоставляет высокоуровневую абстракцию для автоматического сбора метрик HTTP (задержка, количество запросов, коды статусов).  
**Базовая реализация:**

Python

from prometheus_fastapi_instrumentator import Instrumentator

instrumentator \= Instrumentator().instrument(app)  
@app.on_event("startup")  
async def \_startup():  
 instrumentator.expose(app)

Этот код автоматически создает эндпоинт /metrics, который будет опрашиваться (scrape) сервером Prometheus.33

### **6.2. Типология метрик и кастомные коллекторы**

Для глубокой аналитики недостаточно стандартных метрик. Необходимо внедрять кастомные метрики бизнес-логики.

- **Counter (Счетчик):** Используется для событий, которые только накапливаются (количество заказов, ошибок). Важно помнить, что Counter никогда не уменьшается. Для получения скорости событий (RPS) к нему применяется функция rate().35
- **Gauge (Измеритель):** Используется для значений, которые могут колебаться (потребление памяти, количество активных соединений, температура CPU).35

**Проблема конкурентности:** При использовании глобальных переменных для метрик в асинхронном приложении (FastAPI) необходимо учитывать потокобезопасность клиента Prometheus (который в Python обычно потокобезопасен, но требует правильного использования меток). Часто используется паттерн с декораторами или контекстными менеджерами для замера времени выполнения конкретных блоков кода:

Python

with PROCESS_TIME.labels("handler_name").time():  
 await process_data()

Это позволяет избежать дублирования кода и ошибок в измерении длительности.36

### **6.3. Аналитика и A/B тестирование с PromQL**

Собранные данные визуализируются в Grafana с помощью языка запросов PromQL. Особую ценность представляет возможность сравнения метрик между версиями приложения (например, при Canary-развертывании).  
Для сравнения эффективности двух версий (A/B тест) используются запросы с агрегацией по лейблам.

- _Задача:_ Сравнить частоту ошибок (Error Rate) для версии v1 и v2.
- Запрос: sum by (version) (rate(http_requests_total{status=\~"5.."}\[5m\]))  
  Этот запрос сгруппирует данные по версии, позволив наглядно увидеть, какая из версий генерирует больше ошибок.

Сложные кейсы, такие как вычисление разницы между двумя счетчиками с разными лейблами (например, «отправлено со склада 1» минус «получено на складе 2»), требуют использования операторов векторного сопоставления (on, ignoring):  
(warehouse_sent_total \- on(product_id) warehouse_received_total).37

## ---

**Глава 7\. Заключение**

Реализация задачи «Квест 32.2» по установке телеметрии в «Кузницу» представляет собой комплексную инженерную работу, затрагивающую все уровни CI/CD. Мы перешли от простого запуска тестов к системе с жесткими Quality Gates на базе pytest-cov, обеспечили надежное сохранение артефактов с помощью if: always() и upload-artifact@v4, внедрили безопасную аутентификацию через Workload Identity Federation и оптимизировали среду выполнения, освободив критически важные ресурсы. Инструментация приложения метриками Prometheus замкнула цикл обратной связи, предоставив разработчикам данные для принятия обоснованных решений. Данная архитектура является фундаментом для дальнейшего масштабирования и автоматизации процессов разработки.

#### **Источники**

1. Best practices for CI/CD monitoring \- Datadog, дата последнего обращения: декабря 22, 2025, [https://www.datadoghq.com/blog/best-practices-for-ci-cd-monitoring/](https://www.datadoghq.com/blog/best-practices-for-ci-cd-monitoring/)
2. Observability in CI/CD: Logs, Metrics, and Tracing Explained \- Devtron, дата последнего обращения: декабря 22, 2025, [https://devtron.ai/blog/ci-cd-observability-with-devtron/](https://devtron.ai/blog/ci-cd-observability-with-devtron/)
3. Best Practices for Successful CI/CD | TeamCity CI/CD Guide \- JetBrains, дата последнего обращения: декабря 22, 2025, [https://www.jetbrains.com/teamcity/ci-cd-guide/ci-cd-best-practices/](https://www.jetbrains.com/teamcity/ci-cd-guide/ci-cd-best-practices/)
4. How to Debug CI/CD Pipelines: A Handbook on Troubleshooting with Observability Tools, дата последнего обращения: декабря 22, 2025, [https://www.freecodecamp.org/news/how-to-debug-cicd-pipelines-handbook/](https://www.freecodecamp.org/news/how-to-debug-cicd-pipelines-handbook/)
5. Reporting \- pytest-cov 7.0.0 documentation, дата последнего обращения: декабря 22, 2025, [https://pytest-cov.readthedocs.io/en/latest/reporting.html](https://pytest-cov.readthedocs.io/en/latest/reporting.html)
6. Python Coverage · Actions · GitHub Marketplace, дата последнего обращения: декабря 22, 2025, [https://github.com/marketplace/actions/python-coverage](https://github.com/marketplace/actions/python-coverage)
7. Pytest Coverage Comment · Actions · GitHub Marketplace, дата последнего обращения: декабря 22, 2025, [https://github.com/marketplace/actions/pytest-coverage-comment](https://github.com/marketplace/actions/pytest-coverage-comment)
8. Configuration \- pytest-cov 7.0.0 documentation, дата последнего обращения: декабря 22, 2025, [https://pytest-cov.readthedocs.io/en/latest/config.html](https://pytest-cov.readthedocs.io/en/latest/config.html)
9. Building Robust CI/CD Pipelines: Best Practices and Automation \- Wolk, дата последнего обращения: декабря 22, 2025, [https://www.wolk.work/blog/posts/building-robust-ci-cd-pipelines-best-practices-and-automation](https://www.wolk.work/blog/posts/building-robust-ci-cd-pipelines-best-practices-and-automation)
10. Is there a standard way to fail pytest if test coverage falls under x% \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/59420123/is-there-a-standard-way-to-fail-pytest-if-test-coverage-falls-under-x](https://stackoverflow.com/questions/59420123/is-there-a-standard-way-to-fail-pytest-if-test-coverage-falls-under-x)
11. Why doesn't pytest-cov fail with a non-zero exit code when code coverage threshold isn't met, when running on multiple directories? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/79815717/why-doesnt-pytest-cov-fail-with-a-non-zero-exit-code-when-code-coverage-thresho](https://stackoverflow.com/questions/79815717/why-doesnt-pytest-cov-fail-with-a-non-zero-exit-code-when-code-coverage-thresho)
12. cov-fail-under should round before comparing · Issue \#601 · pytest-dev/pytest-cov \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/pytest-dev/pytest-cov/issues/601](https://github.com/pytest-dev/pytest-cov/issues/601)
13. \[bug\] v4 uniquely named artifacts fail with 409 · Issue \#481 · actions/upload-artifact \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/actions/upload-artifact/issues/481](https://github.com/actions/upload-artifact/issues/481)
14. Pytest coverage not showing up in sonarcloud \- Sonar Community, дата последнего обращения: декабря 22, 2025, [https://community.sonarsource.com/t/pytest-coverage-not-showing-up-in-sonarcloud/136230](https://community.sonarsource.com/t/pytest-coverage-not-showing-up-in-sonarcloud/136230)
15. Store and share data with workflow artifacts \- GitHub Docs, дата последнего обращения: декабря 22, 2025, [https://docs.github.com/en/actions/tutorials/store-and-share-data](https://docs.github.com/en/actions/tutorials/store-and-share-data)
16. actions/upload-artifact \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/actions/upload-artifact](https://github.com/actions/upload-artifact)
17. Workflow cancellation reference \- GitHub Docs, дата последнего обращения: декабря 22, 2025, [https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation)
18. Cancel github actions wokflow that uses if:always() · community · Discussion \#26303, дата последнего обращения: декабря 22, 2025, [https://github.com/orgs/community/discussions/26303](https://github.com/orgs/community/discussions/26303)
19. How to run a github-actions step, even if the previous step fails, while still failing the job, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/58858429/how-to-run-a-github-actions-step-even-if-the-previous-step-fails-while-still-f](https://stackoverflow.com/questions/58858429/how-to-run-a-github-actions-step-even-if-the-previous-step-fails-while-still-f)
20. Github Actions if condition requires "always()" to run but that makes it not cancellable · community · Discussion \#25789, дата последнего обращения: декабря 22, 2025, [https://github.com/orgs/community/discussions/25789](https://github.com/orgs/community/discussions/25789)
21. Github action not uploading artifact \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/58226636/github-action-not-uploading-artifact](https://stackoverflow.com/questions/58226636/github-action-not-uploading-artifact)
22. Enabling keyless authentication from GitHub Actions | Google Cloud Blog, дата последнего обращения: декабря 22, 2025, [https://cloud.google.com/blog/products/identity-security/enabling-keyless-authentication-from-github-actions](https://cloud.google.com/blog/products/identity-security/enabling-keyless-authentication-from-github-actions)
23. Configure Workload Identity Federation with deployment pipelines | Identity and Access Management (IAM) | Google Cloud Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines](https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines)
24. A GitHub Action for authenticating to Google Cloud., дата последнего обращения: декабря 22, 2025, [https://github.com/google-github-actions/auth](https://github.com/google-github-actions/auth)
25. seandavi/ghactions-gcp-example: Example repo for authenticating to GCP from ghactions \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/seandavi/ghactions-gcp-example](https://github.com/seandavi/ghactions-gcp-example)
26. auth/docs/EXAMPLES.md at main · google-github-actions/auth, дата последнего обращения: декабря 22, 2025, [https://github.com/google-github-actions/auth/blob/main/docs/EXAMPLES.md](https://github.com/google-github-actions/auth/blob/main/docs/EXAMPLES.md)
27. Using GitHub Actions to authenticate to Google Workload Identity Federation for credentials to use in a Python script \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/78317643/using-github-actions-to-authenticate-to-google-workload-identity-federation-for](https://stackoverflow.com/questions/78317643/using-github-actions-to-authenticate-to-google-workload-identity-federation-for)
28. Squeezing Disk Space from GitHub Actions Runners: An Engineer's Guide \- Matej Lednicky, дата последнего обращения: декабря 22, 2025, [https://mathio28.medium.com/squeezing-disk-space-from-github-actions-runners-an-engineers-guide-d5fbe4443692](https://mathio28.medium.com/squeezing-disk-space-from-github-actions-runners-an-engineers-guide-d5fbe4443692)
29. Free Disk Space \- Ubuntu Runners · Actions · GitHub Marketplace, дата последнего обращения: декабря 22, 2025, [https://github.com/marketplace/actions/free-disk-space-ubuntu-runners](https://github.com/marketplace/actions/free-disk-space-ubuntu-runners)
30. Free Disk Space Action Inspired by jlumbroso/free-disk-space \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/endersonmenezes/free-disk-space](https://github.com/endersonmenezes/free-disk-space)
31. Free Disk Space (Ubuntu) · Actions · GitHub Marketplace, дата последнего обращения: декабря 22, 2025, [https://github.com/marketplace/actions/free-disk-space-ubuntu](https://github.com/marketplace/actions/free-disk-space-ubuntu)
32. Disk Space Reclaimer · Actions · GitHub Marketplace, дата последнего обращения: декабря 22, 2025, [https://github.com/marketplace/actions/disk-space-reclaimer](https://github.com/marketplace/actions/disk-space-reclaimer)
33. trallnag/prometheus-fastapi-instrumentator \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/trallnag/prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
34. PROMETHEUS Metrics for your Python FastAPI App \- YouTube, дата последнего обращения: декабря 22, 2025, [https://www.youtube.com/watch?v=WWzl53ObYvo](https://www.youtube.com/watch?v=WWzl53ObYvo)
35. Sending Custom Metrics from Python to Prometheus | by K Shekar \- Medium, дата последнего обращения: декабря 22, 2025, [https://observabilityfeed.medium.com/sending-custom-metrics-from-python-app-to-prometheus-722211bedfe9](https://observabilityfeed.medium.com/sending-custom-metrics-from-python-app-to-prometheus-722211bedfe9)
36. Dual annotations/decorators of Prometheus Client & FastAPI on same function is not working as expected \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/78636212/dual-annotations-decorators-of-prometheus-client-fastapi-on-same-function-is-n](https://stackoverflow.com/questions/78636212/dual-annotations-decorators-of-prometheus-client-fastapi-on-same-function-is-n)
37. Metric queries | Grafana Loki documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/loki/latest/query/metric_queries/](https://grafana.com/docs/loki/latest/query/metric_queries/)
38. Time Series of counter rates from Prometheus: organize as (a, b) with an additional label z, дата последнего обращения: декабря 22, 2025, [https://community.grafana.com/t/time-series-of-counter-rates-from-prometheus-organize-as-a-b-with-an-additional-label-z/137108](https://community.grafana.com/t/time-series-of-counter-rates-from-prometheus-organize-as-a-b-with-an-additional-label-z/137108)
39. Prometheus \- How to “join” two metrics and calculate the difference?, дата последнего обращения: декабря 22, 2025, [https://community.grafana.com/t/prometheus-how-to-join-two-metrics-and-calculate-the-difference/30055](https://community.grafana.com/t/prometheus-how-to-join-two-metrics-and-calculate-the-difference/30055)
