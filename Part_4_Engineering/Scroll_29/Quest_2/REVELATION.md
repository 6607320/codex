# **Архитектурная парадигма «Тревожного Сигнала»: Глубокий анализ систем автоматизированного мониторинга точности ML-моделей**

## **Аннотация**

В современной экосистеме машинного обучения (MLOps) переход от лабораторных экспериментов к промышленной эксплуатации сопряжен с фундаментальным сдвигом в парадигме контроля качества. Если на этапе разработки (Model Development) фокус сосредоточен на максимизации метрик точности на статических валидационных выборках, то в фазе эксплуатации (Model Operation) критическую роль играет стабильность этих метрик во времени. Данный отчет представляет собой исчерпывающее техническое исследование методов настройки автоматизированных систем оповещения о деградации качества моделей, метафорически обозначенных как ритуал «Тревожного Сигнала». В документе детально рассматривается интеграция фреймворка FastAPI, системы сбора метрик Prometheus, инструмента активного зондирования Blackbox Exporter и платформы визуализации Grafana. Особое внимание уделяется архитектурным паттернам обработки асинхронных вызовов, сложностям валидации JSON-пейлоудов через регулярные выражения, инверсии логики пороговых значений (thresholds) для метрик качества и предотвращению блокировок событийного цикла (Event Loop) при расчете метрик. Анализ опирается на широкую базу технических источников, синтезируя разрозненные практики в единую стратегию обеспечения надежности интеллектуальных систем.

## ---

**1\. Эпистемология сбоев в машинном обучении: От технического мониторинга к контролю качества**

### **1.1. Дихотомия мониторинга в MLOps**

Традиционные практики DevOps, сформировавшиеся за последние десятилетия, достигли высокой зрелости в вопросах обеспечения надежности программного обеспечения. Инженеры располагают обширным инструментарием для отслеживания здоровья сервисов: метрики использования CPU, потребления оперативной памяти, задержки сети (latency) и коды ответов HTTP являются стандартными индикаторами жизнеспособности системы.1 Однако в контексте машинного обучения эти метрики, будучи необходимыми, оказываются недостаточными. Фундаментальное отличие ML-сервисов заключается в том, что модель может функционировать технически безупречно — отвечать за миллисекунды, не потреблять лишних ресурсов и возвращать статус 200 OK — но при этом генерировать абсолютно некорректные предсказания, разрушающие бизнес-логику приложения.2  
Этот феномен, известный как «молчаливый сбой» (silent failure), требует внедрения специализированного уровня мониторинга — мониторинга качества модели (Model Quality Monitoring). В иерархии потребностей MLOps, предложенной ведущими исследователями, мониторинг качества занимает вершину пирамиды, опираясь на мониторинг здоровья системы и качества данных.2 Реализация ритуала «Тревожного Сигнала» — это процесс настройки автоматической системы, способной детектировать именно семантические сбои, когда модель теряет связь с реальностью, продолжая при этом работать технически исправно.

### **1.2. Природа деградации: Дрейф данных и концепций**

Для корректной настройки порогов срабатывания «Тревожного Сигнала» необходимо глубокое понимание причин падения точности. В отличие от программного кода, который детерминирован, поведение ML-модели является стохастическим и зависит от входных данных. Существует два основных типа дрейфа, приводящих к деградации:

1. **Дрейф данных (Data Drift):** Это изменение статистического распределения входных признаков $P(X)$ по сравнению с обучающей выборкой. Например, если модель компьютерного зрения обучалась на качественных изображениях, а в продакшене на вход поступают зашумленные снимки с камер видеонаблюдения, точность неизбежно упадет, даже если сама логика принятия решений (функция $P(Y|X)$) осталась прежней.1
2. **Дрейф концепции (Concept Drift):** Это изменение самой зависимости между входными данными и целевой переменной $P(Y|X)$. Классическим примером является модель анализа тональности или спам-фильтр. С течением времени язык, сленг и методы спамеров меняются. Словосочетания, которые вчера имели положительную коннотацию, сегодня могут стать негативными или ироничными. В этом случае модель «устаревает» концептуально.3

Понимание этих процессов критично для выбора стратегии мониторинга. Если деградация происходит резко (dramatic shift), это часто указывает на технический сбой в пайплайне данных (например, изменение формата входных данных или единицах измерения). Если же деградация происходит медленно (slow-leak regression), это, как правило, признак естественного дрейфа концепции, требующего переобучения модели.1

### **1.3. Проблема Ground Truth и методы оценки**

Центральной проблемой мониторинга качества в реальном времени является задержка получения истинных меток (Ground Truth). В задачах, где правильный ответ становится известен лишь спустя значительное время (например, выдача кредита и факт его возврата через год), прямое вычисление точности (Accuracy) или F1-score в момент инференса невозможно.4  
В связи с этим выделяют два подхода к генерации «Тревожного Сигнала»:

1. **Отложенная оценка (Backtesting):** Сравнение предсказаний с метками, поступившими с задержкой. Этот метод точен, но не позволяет реагировать оперативно.4
2. **Активное зондирование (Active Probing) на «Золотом наборе» (Golden Set):** Именно этот метод лежит в основе описываемого в данном отчете решения. Сервис мониторинга периодически отправляет модели специально подготовленные запросы, правильные ответы на которые известны заранее. Если модель ошибается на этих эталонных данных, система немедленно генерирует алерт. Этот подход позволяет оценить состояние модели «здесь и сейчас», имитируя поведение реального пользователя, но с контролируемым результатом.5

## ---

**2\. Архитектура системы мониторинга: Компоненты и Взаимодействие**

Для реализации надежной системы оповещения о падении точности ниже заданного порога предлагается использовать микросервисную архитектуру, оркестрируемую посредством Docker Compose. Данный подход обеспечивает изоляцию компонентов, воспроизводимость среды и легкость масштабирования.

### **2.1. Обзор технологического стека**

Архитектура включает четыре ключевых компонента, каждый из которых выполняет строго определенную роль в контуре обратной связи:

| Компонент                | Роль в системе                                                    | Обоснование выбора                                                                                       |
| :----------------------- | :---------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------- |
| **ML Service (FastAPI)** | Хост для ML-модели, обработка запросов, экспорт внутренних метрик | Высокая производительность, асинхронность, встроенная поддержка валидации данных.7                       |
| **Prometheus**           | Сбор (scraping) и хранение метрик во временном ряду (TSDB)        | Стандарт де\-факто в cloud-native мониторинге, мощный язык запросов PromQL, модель pull-based.7          |
| **Blackbox Exporter**    | Активное зондирование API внешними запросами                      | Возможность имитации клиентских запросов (POST, GET), проверка кодов ответа и содержимого body.5         |
| **Grafana**              | Визуализация метрик и управление алертами                         | Гибкая настройка дашбордов, поддержка сложных пороговых значений (thresholds) и визуальных индикаторов.7 |

### **2.2. Топология сети и Service Discovery**

В среде Docker Compose критически важно обеспечить корректное сетевое взаимодействие между контейнерами. Prometheus должен иметь доступ к сетевым интерфейсам как ML-сервиса, так и Blackbox Exporter для сбора метрик. Blackbox Exporter, в свою очередь, должен иметь возможность отправлять запросы к ML-сервису для проверки его работоспособности.11  
Конфигурация docker-compose.yml определяет пользовательскую сеть (bridge network), в которую включены все сервисы. Это позволяет использовать имена сервисов (service names) в качестве хостнеймов. Например, если ML-сервис назван model_api, а Blackbox Exporter — blackbox, то Prometheus может обращаться к ним по адресам http://model_api:8000/metrics и http://blackbox:9115 соответственно.12

### **2.3. Жизненный цикл метрики**

Поток данных в системе «Тревожного Сигнала» выглядит следующим образом:

1. **Генерация:** Blackbox Exporter периодически (согласно scrape_interval в Prometheus) отправляет тестовый запрос к model_api.
2. **Валидация:** Blackbox Exporter анализирует ответ модели на соответствие ожидаемому паттерну (например, наличие правильного класса в JSON).
3. **Экспорт:** Результат проверки (1 — успех, 0 — провал) и временные характеристики запроса экспонируются на эндпоинте /probe.
4. **Сбор:** Prometheus забирает эти данные и сохраняет их в своей базе.
5. **Визуализация/Алерт:** Grafana запрашивает данные из Prometheus, отображает их на панели Gauge и, если значение падает ниже порога (например, 0.9), меняет цвет индикатора на красный, сигнализируя о проблеме.7

## ---

**3\. Реализация ML-сервиса на FastAPI: Инструментация и Кастомные Метрики**

FastAPI зарекомендовал себя как оптимальный выбор для развертывания моделей благодаря нативной поддержке асинхронности и Pydantic. Однако интеграция мониторинга требует внимательного подхода к управлению конкурентностью и выбору типов метрик.

### **3.1. Интеграция Prometheus-инструментатора**

Для базового мониторинга HTTP-трафика (RPS, latency, error rate) наиболее эффективным решением является использование библиотеки prometheus-fastapi-instrumentator. Она работает как middleware, перехватывая каждый запрос и автоматически обновляя соответствующие метрики.9

Python

from fastapi import FastAPI  
from prometheus_fastapi_instrumentator import Instrumentator

app \= FastAPI()

\# Инициализация и запуск экспортера метрик  
Instrumentator().instrument(app).expose(app)

Этот код создает эндпоинт /metrics, который Prometheus будет опрашивать. Однако стандартных метрик недостаточно для мониторинга _качества_ модели. Необходимо внедрение кастомных метрик.15

### **3.2. Типология метрик точности: Почему Gauge?**

В системе Prometheus существует четыре основных типа метрик: Counter, Gauge, Histogram и Summary. Выбор правильного типа критичен для семантики «Тревожного Сигнала».

- **Counter (Счетчик):** Монотонно возрастающая величина. Подходит для подсчета общего количества запросов или ошибок, но не для точности, так как точность может колебаться.16
- **Gauge (Шкала):** Величина, которая может произвольно изменяться вверх и вниз. **Именно этот тип является единственно верным выбором для отображения текущей точности модели (accuracy), доли уверенности (confidence score) или потерь (loss)**.17

Пример реализации метрики точности:

Python

from prometheus_client import Gauge

\# Определение метрики. Важно использовать labels для детализации  
MODEL_ACCURACY \= Gauge(  
 "model_accuracy_current",  
 "Real-time accuracy of the model evaluated on golden set",  
 \["version", "environment"\]  
)

\# Функция обновления метрики (вызывается при прохождении теста)  
def update_accuracy(value: float):  
 MODEL_ACCURACY.labels(version="v1.0.2", environment="prod").set(value)

Использование меток (labels) позволяет отслеживать точность различных версий модели одновременно, что критично при A/B тестировании или канареечных релизах.1

### **3.3. Проблема блокировки Event Loop при вычислении метрик**

Одной из самых коварных ловушек при использовании FastAPI является блокировка событийного цикла (Event Loop) синхронными операциями. Если вычисление точности модели или валидация входных данных требует значительных вычислений (CPU-bound) и выполняется внутри асинхронного эндпоинта (async def), это может парализовать работу всего сервиса, включая эндпоинт /metrics.19  
В Python, даже при использовании async, существует глобальная блокировка интерпретатора (GIL). Если «тяжелая» функция инференса модели (например, прогон через нейросеть) вызывается напрямую в async функции, она захватывает управление и не отдает его до завершения, блокируя обработку других запросов (heartbeats, metrics scraping).21  
**Рекомендация:** Для CPU-емких задач следует использовать def (синхронные) эндпоинты, которые FastAPI автоматически запускает в отдельном пуле потоков (thread pool), либо выносить вычисления в отдельные процессы (Celery, background tasks).21 Это гарантирует, что система мониторинга будет получать метрики даже под высокой нагрузкой.

## ---

**4\. Методология активного зондирования: Конфигурация Blackbox Exporter**

Для реализации проверки того, что «точность модели упала ниже порога», недостаточно просто знать, что сервис жив. Необходимо проверить, что он возвращает _правильные_ ответы. Для этого используется паттерн активного зондирования (Active Probing) с помощью Blackbox Exporter.

### **4.1. Специфика POST-запросов и JSON-пейлоудов**

Большинство ML-моделей принимают входные данные через POST-запросы с телом в формате JSON. Стандартные примеры Blackbox Exporter часто ограничиваются GET-запросами и проверкой статуса 200\. Для нашей задачи требуется более сложная конфигурация.14  
В файле конфигурации blackbox.yml необходимо определить модуль, который будет отправлять специфический JSON.  
Пример конфигурации для проверки модели классификации:

YAML

modules:  
 model_inference_check:  
 prober: http  
 timeout: 5s \# Таймаут на выполнение проверки  
 http:  
 method: POST  
 headers:  
 Content-Type: application/json  
 \# Тело запроса с тестовым вектором признаков ("Golden Set")  
 body: '{"feature1": 0.5, "feature2": 10, "text": "test sentence"}'  
 valid_status_codes:

Здесь мы имитируем реальный запрос к модели. Однако получения ответа 200 OK недостаточно — модель может вернуть 200 OK и абсолютно неверное предсказание.6

### **4.2. Валидация содержимого: Regexp vs JSON Parsing**

Blackbox Exporter не имеет встроенного парсера JSON, который позволял бы выполнять логические сравнения (например, if prediction_score \< 0.8). Единственным инструментом валидации содержимого являются регулярные выражения.24  
Для реализации «Тревожного Сигнала» используется параметр fail_if_body_not_matches_regexp. Это мощный механизм, который помечает пробу как неуспешную (Failed), если в теле ответа не найдена ожидаемая строка.  
Пример для задачи классификации (ожидаем класс "fraud"):

YAML

      fail\_if\_body\_not\_matches\_regexp:
        \- '.\*"class":\\s\*"fraud".\*'

В данном случае, если модель вернет класс "legit" вместо ожидаемого "fraud", регулярное выражение не совпадет, метрика probe_success станет равна 0, и Prometheus зафиксирует сбой. Это и есть триггер для нашего сигнала.6

### **4.3. Управление таймаутами и ложные срабатывания**

Распространенной ошибкой при настройке является несогласованность таймаутов. Инференс сложных нейросетей может занимать секунды. Если таймаут Blackbox Exporter (параметр timeout в модуле) меньше времени ответа модели, мониторинг будет постоянно генерировать ложные алерты о недоступности.26  
Более того, существует ограничение на уровне Prometheus. Параметр scrape_timeout в конфигурации Prometheus должен быть всегда больше или равен таймауту, заданному в Blackbox Exporter. Если Prometheus перестанет ждать ответа от экспортера раньше, чем экспортер получит ответ от модели, данные будут потеряны. Рекомендуется устанавливать scrape_timeout с запасом (например, 10 секунд при таймауте пробы 5 секунд).5

## ---

**5\. Визуализация и Психология цвета: Настройка панелей Grafana**

Grafana является финальным звеном в цепочке «Тревожного Сигнала». Здесь сухие цифры метрик трансформируются в визуальные индикаторы, понятные человеку. Для отображения точности модели (значение от 0 до 1 или от 0% до 100%) идеальным инструментом является панель **Gauge**.17

### **5.1. Инверсия логики пороговых значений (Thresholds)**

Стандартная семантика цветов в системах мониторинга (и в Grafana по умолчанию) ориентирована на метрики утилизации ресурсов (CPU, Memory), где «больше» означает «хуже». Поэтому дефолтные настройки выглядят так:

- Низкие значения (Base) \-\> Зеленый (OK)
- Высокие значения (\>80%) \-\> Красный (Critical).10

Для метрики точности (Accuracy) эта логика диаметрально противоположна: высокое значение (например, 0.95) — это хорошо (Зеленый), а низкое (например, 0.6) — это критический сбой (Красный).  
Чтобы настроить «Тревожный Сигнал» правильно, необходимо инвертировать пороги.29  
**Правильная конфигурация порогов для точности:**

| Значение (Threshold)           | Цвет       | Семантика                                                                           |
| :----------------------------- | :--------- | :---------------------------------------------------------------------------------- |
| **Base** (минус бесконечность) | **Red**    | Критическая зона. Если точность упала до нуля или очень низка, это сигнал бедствия. |
| **0.80** (80%)                 | **Yellow** | Зона предупреждения. Точность приемлема, но требует внимания.                       |
| **0.90** (90%)                 | **Green**  | Нормальная работа. Модель функционирует в штатном режиме.                           |

Таким образом, шкала будет окрашиваться в красный цвет _до_ достижения порога 0.8, в желтый между 0.8 и 0.9, и в зеленый выше 0.9. Это создает интуитивно понятную картину: падение стрелки влево (вниз) приводит к покраснению панели.30

### **5.2. Проблема базового цвета в новых версиях Grafana**

В версиях Grafana 9.x и новее пользователи часто сталкиваются с проблемой, когда базовый цвет (Base) принудительно устанавливается в зеленый при использовании автоматических конфигураций или режима "Config from query results". Это может сломать логику инвертированного мониторинга.32  
Для решения этой проблемы необходимо:

1. Явно переключить режим порогов (Thresholds mode) в **Absolute**.
2. Вручную удалить все автоматически созданные пороги.
3. Установить цвет для точки **Base** в **Red**.
4. Добавить последующие пороги (0.8, 0.9) с соответствующими цветами.  
   Важно помнить, что Grafana сортирует пороги от большего к меньшему, но логика окрашивания работает по принципу «значение \>= порогу». Поэтому установка Base в Red гарантирует, что все значения, не достигшие следующего порога, будут красными.34

## ---

**6\. Альтернативные паттерны: Sidecar-контейнеры для сложной логики**

Хотя связка Prometheus \+ Blackbox Exporter покрывает 90% сценариев, существуют ситуации, когда возможностей регулярных выражений недостаточно. Например, если необходимо проверить, что сумма вероятностей всех классов равна 1, или что confidence score для предсказанного класса выше динамического порога.  
В таких случаях целесообразно использовать паттерн **Sidecar Container**.

### **6.1. Архитектура Sidecar-монитора**

Рядом с контейнером ML-сервиса запускается вспомогательный контейнер (обычно на базе Python-образа), который содержит скрипт валидации. Этот скрипт запускается по расписанию (Cron) внутри контейнера.36  
Преимущества подхода:

- **Полная гибкость:** Можно использовать requests, pandas, numpy для анализа ответа модели.
- **Сложная логика:** Валидация JSON любой сложности, сравнение чисел, проверка типов данных.
- **Изоляция:** Скрипт выполняется независимо и не нагружает основной процесс ML-сервиса.

### **6.2. Передача метрик**

Результаты работы Sidecar-скрипта (например, вычисленное значение точности 0.92) необходимо передать в Prometheus. Для этого существует два пути:

1. **Pushgateway:** Скрипт отправляет метрику методом POST в Prometheus Pushgateway, откуда Prometheus её забирает.
2. **Textfile Collector:** Скрипт записывает метрику в файл .prom в общей директории (volume), которую читает Node Exporter.

Этот подход позволяет реализовать максимально детализированный «Тревожный Сигнал», реагирующий на тончайшие отклонения в поведении модели, которые невозможно уловить простым HTTP-зондированием.24

## ---

**Заключение**

Реализация ритуала «Тревожного Сигнала» трансформирует ML-модель из непрозрачного «черного ящика» в наблюдаемый и управляемый актив. Мы проанализировали путь от теоретического понимания дрейфа данных до конкретных инженерных решений на базе FastAPI, Prometheus и Grafana.  
Ключевые выводы исследования:

1. **Мониторинг качества первичен:** Техническая доступность сервиса не гарантирует его бизнес-ценности. Активное зондирование на эталонных данных — необходимый минимум для критических систем.
2. **Архитектурная согласованность:** Настройка таймаутов, сетевого взаимодействия в Docker Compose и типов метрик (Gauge vs Counter) требует системного подхода. Ошибка в одном компоненте (например, блокировка Event Loop) может скомпрометировать всю систему мониторинга.
3. **Визуальная ясность:** Правильная настройка порогов в Grafana с инверсией цветов является критическим элементом UX для операторов системы, позволяя мгновенно идентифицировать деградацию.

Внедрение описанных практик позволяет создать надежный контур обратной связи, минимизируя время реакции на сбои и обеспечивая долгосрочную стабильность ML-решений в агрессивной среде реальной эксплуатации.

#### **Источники**

1. MLOps Principles, дата последнего обращения: декабря 22, 2025, [https://ml-ops.org/content/mlops-principles](https://ml-ops.org/content/mlops-principles)
2. Monitoring ML systems in production. Which metrics should you track? \- Evidently AI, дата последнего обращения: декабря 22, 2025, [https://www.evidentlyai.com/blog/ml-monitoring-metrics](https://www.evidentlyai.com/blog/ml-monitoring-metrics)
3. A Comprehensive Guide on How to Monitor Your Models in Production \- Neptune.ai, дата последнего обращения: декабря 22, 2025, [https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide](https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide)
4. Machine learning model monitoring: Best practices \- Datadog, дата последнего обращения: декабря 22, 2025, [https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/)
5. prometheus/blackbox_exporter: Blackbox prober exporter \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/prometheus/blackbox_exporter](https://github.com/prometheus/blackbox_exporter)
6. Prometheus Blackbox Exporter and POST calls \- @abiydv, дата последнего обращения: декабря 22, 2025, [https://abiydv.github.io/posts/prometheus-blackbox-monitor-post-api/](https://abiydv.github.io/posts/prometheus-blackbox-monitor-post-api/)
7. FastAPI Observability Lab with Prometheus and Grafana: Complete Guide \- Towards AI, дата последнего обращения: декабря 22, 2025, [https://pub.towardsai.net/fastapi-observability-lab-with-prometheus-and-grafana-complete-guide-f12da15a15fd](https://pub.towardsai.net/fastapi-observability-lab-with-prometheus-and-grafana-complete-guide-f12da15a15fd)
8. Concurrency and async / await \- FastAPI, дата последнего обращения: декабря 22, 2025, [https://fastapi.tiangolo.com/async/](https://fastapi.tiangolo.com/async/)
9. Getting Started: Monitoring a FastAPI App with Grafana and Prometheus \- A Step-by-Step Guide \- DEV Community, дата последнего обращения: декабря 22, 2025, [https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-grafana-and-prometheus-a-step-by-step-guide-3fbn](https://dev.to/ken_mwaura1/getting-started-monitoring-a-fastapi-app-with-grafana-and-prometheus-a-step-by-step-guide-3fbn)
10. Configure thresholds | Grafana Cloud documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana-cloud/visualizations/panels-visualizations/configure-thresholds/](https://grafana.com/docs/grafana-cloud/visualizations/panels-visualizations/configure-thresholds/)
11. Connecting services with Docker Compose, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/guides/go-prometheus-monitoring/compose/](https://docs.docker.com/guides/go-prometheus-monitoring/compose/)
12. Monitoring a Linux host with Prometheus, Node Exporter, and Docker Compose \- Grafana, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-prometheus/prometheus-config-examples/docker-compose-linux/](https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-prometheus/prometheus-config-examples/docker-compose-linux/)
13. stfsy/prometheus-grafana-blackbox-exporter: A docker-compose stack for Prometheus monitoring \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/stfsy/prometheus-grafana-blackbox-exporter](https://github.com/stfsy/prometheus-grafana-blackbox-exporter)
14. Understanding and using the multi-target exporter pattern \- Prometheus, дата последнего обращения: декабря 22, 2025, [https://prometheus.io/docs/guides/multi-target-exporter/](https://prometheus.io/docs/guides/multi-target-exporter/)
15. trallnag/prometheus-fastapi-instrumentator: Instrument your ... \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/trallnag/prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
16. Sending Custom Metrics from Python to Prometheus | by K Shekar \- Medium, дата последнего обращения: декабря 22, 2025, [https://observabilityfeed.medium.com/sending-custom-metrics-from-python-app-to-prometheus-722211bedfe9](https://observabilityfeed.medium.com/sending-custom-metrics-from-python-app-to-prometheus-722211bedfe9)
17. Gauge | Grafana documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/gauge/](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/gauge/)
18. How to add middleware to the Fast API to create metrics to track time spent and requests made? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/77338597/how-to-add-middleware-to-the-fast-api-to-create-metrics-to-track-time-spent-and](https://stackoverflow.com/questions/77338597/how-to-add-middleware-to-the-fast-api-to-create-metrics-to-track-time-spent-and)
19. FastAPI's Async Superpowers: Don't Be That Developer Who Blocks the Event Loop\! | by Sarthak Shah | Dec, 2025 | Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@sarthakshah1920/fastapis-async-superpowers-don-t-be-that-developer-who-blocks-the-event-loop-651be5ac1384](https://medium.com/@sarthakshah1920/fastapis-async-superpowers-don-t-be-that-developer-who-blocks-the-event-loop-651be5ac1384)
20. FastAPI \- Why does synchronous code do not block the event Loop? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/79382645/fastapi-why-does-synchronous-code-do-not-block-the-event-loop](https://stackoverflow.com/questions/79382645/fastapi-why-does-synchronous-code-do-not-block-the-event-loop)
21. FastAPI is blocked when an endpoint takes longer \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/FastAPI/comments/1euhq69/fastapi_is_blocked_when_an_endpoint_takes_longer/](https://www.reddit.com/r/FastAPI/comments/1euhq69/fastapi_is_blocked_when_an_endpoint_takes_longer/)
22. FastAPI Mistakes That Kill Your Performance \- DEV Community, дата последнего обращения: декабря 22, 2025, [https://dev.to/igorbenav/fastapi-mistakes-that-kill-your-performance-2b8k](https://dev.to/igorbenav/fastapi-mistakes-that-kill-your-performance-2b8k)
23. Blackbox Post Request SOAP \- Google Groups, дата последнего обращения: декабря 22, 2025, [https://groups.google.com/g/prometheus-users/c/D_uRS7fx51Y](https://groups.google.com/g/prometheus-users/c/D_uRS7fx51Y)
24. Promethues check external API json response \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/74759672/promethues-check-external-api-json-response](https://stackoverflow.com/questions/74759672/promethues-check-external-api-json-response)
25. Is it possible to support HTTP body response? · Issue \#427 · prometheus/blackbox_exporter, дата последнего обращения: декабря 22, 2025, [https://github.com/prometheus/blackbox_exporter/issues/427](https://github.com/prometheus/blackbox_exporter/issues/427)
26. prometheus.exporter.blackbox | Grafana Alloy documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/alloy/latest/reference/components/prometheus/prometheus.exporter.blackbox/](https://grafana.com/docs/alloy/latest/reference/components/prometheus/prometheus.exporter.blackbox/)
27. Configuring Blackbox exporter timeouts \- Robust Perception, дата последнего обращения: декабря 22, 2025, [https://www.robustperception.io/configuring-blackbox-exporter-timeouts/](https://www.robustperception.io/configuring-blackbox-exporter-timeouts/)
28. blackbox_exporter timeout configuration (status 0, context deadline exceeded), дата последнего обращения: декабря 22, 2025, [https://groups.google.com/g/prometheus-users/c/SEMGJg9O2Sk](https://groups.google.com/g/prometheus-users/c/SEMGJg9O2Sk)
29. How To? Configure Gauge thresholds lower values reduce, not increase, дата последнего обращения: декабря 22, 2025, [https://community.grafana.com/t/how-to-configure-gauge-thresholds-lower-values-reduce-not-increase/100945](https://community.grafana.com/t/how-to-configure-gauge-thresholds-lower-values-reduce-not-increase/100945)
30. Correctly tweaking thresholds in Grafana Gauges, дата последнего обращения: декабря 22, 2025, [https://community.grafana.com/t/correctly-tweaking-thresholds-in-grafana-gauges/71286](https://community.grafana.com/t/correctly-tweaking-thresholds-in-grafana-gauges/71286)
31. Setting threshold from maximum to minimum \- Grafana Labs Community Forums, дата последнего обращения: декабря 22, 2025, [https://community.grafana.com/t/setting-threshold-from-maximum-to-minimum/732](https://community.grafana.com/t/setting-threshold-from-maximum-to-minimum/732)
32. Gauge Panel Dynamic Threshold "Config from query" Transformation Problem With Base Color and Doesn't Allow Multiple Thresholds \- Grafana, дата последнего обращения: декабря 22, 2025, [https://community.grafana.com/t/gauge-panel-dynamic-threshold-config-from-query-transformation-problem-with-base-color-and-doesnt-allow-multiple-thresholds/132532](https://community.grafana.com/t/gauge-panel-dynamic-threshold-config-from-query-transformation-problem-with-base-color-and-doesnt-allow-multiple-thresholds/132532)
33. How can I change the threshold colors from config query? \- Time Series Panel, дата последнего обращения: декабря 22, 2025, [https://community.grafana.com/t/how-can-i-change-the-threshold-colors-from-config-query/55003](https://community.grafana.com/t/how-can-i-change-the-threshold-colors-from-config-query/55003)
34. Thresholds \- Amazon Managed Grafana \- AWS Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.aws.amazon.com/grafana/latest/userguide/thresholds.html](https://docs.aws.amazon.com/grafana/latest/userguide/thresholds.html)
35. Configure thresholds | Grafana documentation, дата последнего обращения: декабря 22, 2025, [https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/configure-thresholds/](https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/configure-thresholds/)
36. Running a Python script periodically in a Docker container using cron \- Andrew Conlin, дата последнего обращения: декабря 22, 2025, [https://andrewconl.in/til/running-python-in-cron-in-docker/](https://andrewconl.in/til/running-python-in-cron-in-docker/)
37. How to run a cron job inside a docker container? \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/37458287/how-to-run-a-cron-job-inside-a-docker-container](https://stackoverflow.com/questions/37458287/how-to-run-a-cron-job-inside-a-docker-container)
38. Running scheduled Python tasks in a Docker container | by Nils Schröder \- Medium, дата последнего обращения: декабря 22, 2025, [https://nschdr.medium.com/running-scheduled-python-tasks-in-a-docker-container-bf9ea2e8a66c](https://nschdr.medium.com/running-scheduled-python-tasks-in-a-docker-container-bf9ea2e8a66c)
39. Running a Sidecar container as a cron job : r/devops \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/devops/comments/1ea8z43/running_a_sidecar_container_as_a_cron_job/](https://www.reddit.com/r/devops/comments/1ea8z43/running_a_sidecar_container_as_a_cron_job/)
