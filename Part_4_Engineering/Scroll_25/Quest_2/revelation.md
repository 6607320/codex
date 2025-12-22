# **Отчет об исследовании: Архитектурная оптимизация контейнерных артефактов в экосистеме Python: Сравнительный анализ многоступенчатых и монолитных сборок**

## **Аннотация**

В современной парадигме разработки программного обеспечения контейнеризация стала стандартом де\-факто для упаковки и доставки приложений. Однако по мере роста сложности микросервисных архитектур и внедрения моделей машинного обучения (ML) и Data Science в производственные среды, проблема размера контейнерных образов (Docker images) приобрела критическое значение. В контексте языка Python, требующего наличия компиляторов и системных библиотек для сборки расширений C/C++, использование наивных («монолитных») подходов к созданию Dockerfile приводит к появлению избыточных, небезопасных и дорогостоящих в эксплуатации артефактов.  
Данный отчет представляет собой исчерпывающее исследование методологий оптимизации Docker-образов для Python-приложений. Основное внимание уделяется переходу от монолитных сборок к многоступенчатым (multi-stage builds). В работе детально рассматриваются механика файловых систем UnionFS, специфика управления зависимостями в Python (pip, poetry, uv), проблемы совместимости стандартных библиотек C (glibc vs musl), а также влияние размера образа на безопасность (поверхность атаки, CVE) и экономику облачной инфраструктуры (время масштабирования Kubernetes, затраты на хранение и трафик). Исследование базируется на агрегированных технических данных, бенчмарках и актуальных рекомендациях индустрии по состоянию на 2024-2025 годы.

## **1\. Введение: Кризис «раздутых» контейнеров в современной разработке**

### **1.1. Эволюция подходов к сборке артефактов**

Переход от виртуальных машин к контейнерам обещал легковесность и скорость. Однако на практике многие команды разработки переносят ментальную модель работы с VM на контейнеры, создавая так называемые «толстые» (fat) или монолитные контейнеры. Монолитная сборка подразумевает использование единственного сценария (Dockerfile), который последовательно выполняет все этапы: подготовку ОС, установку инструментов сборки, компиляцию кода и запуск приложения.1  
В экосистеме Python эта проблема стоит особенно остро. Python — интерпретируемый язык, но его мощь во многом базируется на библиотеках с нативными расширениями (NumPy, Pandas, Cryptography, PyTorch). Для их установки часто требуется полноценный набор инструментов компиляции (GNU Compiler Collection — GCC, заголовочные файлы Linux headers, make), которые весят сотни мегабайт.2 В монолитной парадигме эти инструменты остаются в финальном образе, создавая «мертвый груз», который никогда не используется во время выполнения приложения (runtime), но потребляет ресурсы хранения и увеличивает риски безопасности.3

### **1.2. Технический долг монолитных образов**

Использование образов размером 1 ГБ и более для запуска простых микросервисов создает значительный технический долг. Это проявляется в увеличении времени прохождения CI/CD пайплайнов (задержки на этапах docker push/docker pull), росте расходов на хранение в реестрах (Amazon ECR, Google GCR) и, что критически важно, в замедлении реакции систем оркестрации (Kubernetes) на изменения нагрузки.4  
Кроме того, монолитные образы нарушают принцип минимальных привилегий и минимальной поверхности атаки. Наличие компилятора и пакетного менеджера в продуктовом контуре предоставляет злоумышленнику, получившему доступ к контейнеру (RCE), готовый инструментарий для эскалации привилегий и бокового перемещения внутри кластера.2

## **2\. Архитектурная механика Docker и проблема монолитов**

Для глубокого понимания неэффективности монолитных сборок необходимо проанализировать работу файловой системы Docker на низком уровне.

### **2.1. Union File System и неизменяемость слоев**

Docker использует объединяемые файловые системы (UnionFS), чаще всего драйвер overlay2 в Linux. Образ Docker состоит из серии доступных только для чтение (read-only) слоев. Каждая инструкция в Dockerfile (RUN, COPY, ADD), изменяющая файловую систему, создает новый слой, который накладывается поверх предыдущего.7  
В монолитной сборке разработчики часто пытаются оптимизировать размер, выполняя установку зависимостей и их удаление в разных инструкциях. Например:

1. RUN apt-get install gcc (Слой А: \+200 МБ)
2. RUN pip install numpy (Слой Б: \+100 МБ)
3. RUN apt-get remove gcc (Слой В: \+0 МБ, "whiteout" файлов)

Из-за иммутабельности слоев, удаление файлов в слое В не физически стирает данные из слоя А. Файловая система лишь помечает файлы как «удаленные» (создает whiteout-файлы), делая их невидимыми для процесса контейнера. Однако на диске и при передаче по сети образ продолжает содержать все 200 МБ компилятора GCC.8 Это фундаментальное ограничение делает невозможным эффективную очистку образа в рамках одной стадии сборки, если удаление не происходит в той же инструкции RUN, что и установка (с использованием цепочки команд через &&). Но даже при использовании цепочек команд поддержка Dockerfile становится кошмаром читаемости и сопровождения.9

### **2.2. Контекст сборки и неэффективность кэширования**

Еще одной проблемой монолитных сборок является неэффективное использование кэша сборки (Build Cache). Docker кэширует слои на основе хеша содержимого. Если в монолитном Dockerfile код приложения копируется (COPY..) перед установкой зависимостей, то любое изменение в исходном коде (даже комментарий) инвалидирует кэш для всех последующих команд.8  
Это приводит к тому, что при каждой правке кода запускается процесс переустановки всех зависимостей, что для тяжелых проектов (например, с ML-библиотеками) может занимать 10-20 минут. Это разрушает цикл быстрой обратной связи (feedback loop), критичный для Agile-разработки и DevOps-практик.10

## **3\. Многоступенчатые сборки (Multi-Stage Builds): Парадигма разделения**

### **3.1. Концептуальная модель**

Многоступенчатые сборки, внедренные в Docker 17.05, представляют собой архитектурный паттерн, позволяющий использовать несколько инструкций FROM в одном Dockerfile.1 Каждая инструкция FROM инициализирует новую стадию сборки, которая может базироваться на совершенно другом базовом образе.  
Ключевая особенность заключается в возможности копировать артефакты (скомпилированные бинарные файлы, виртуальные окружения, статические ассеты) из одной стадии в другую, используя инструкцию COPY \--from=\<stage_name\>. При этом все промежуточные слои, временные файлы и инструменты, оставшиеся в предыдущих стадиях, отбрасываются и не попадают в финальный образ.9

### **3.2. Роль BuildKit и графов зависимостей**

Современные версии Docker используют движок сборки BuildKit, который строит граф зависимостей стадий (DAG \- Directed Acyclic Graph). Это позволяет не только уменьшать размер, но и распараллеливать сборку. Если стадии не зависят друг от друга (например, сборка фронтенда и бэкенда), BuildKit может выполнять их одновременно.9  
В отличие от классического сборщика, который выполнял все инструкции последовательно, BuildKit пропускает стадии, которые не требуются для целевой стадии (target stage), что позволяет создавать универсальные Dockerfile для разработки, тестирования и продакшна.12

### **3.3. Анатомия оптимизированного Python-образа**

В контексте Python многоступенчатая сборка разделяет жизненный цикл приложения на две фазы:

1. **Builder Stage (Стадия сборщика):** Использует «тяжелый» образ (часто суффикс \-slim с дополнительно установленными build-essential), содержащий компиляторы, заголовочные файлы ядра и библиотеки разработчика (libpq-dev, libffi-dev). Здесь происходит создание виртуального окружения (virtualenv) и компиляция всех wheel-пакетов.13
2. **Runtime Stage (Стадия выполнения):** Использует минималистичный образ (обычно python:3.X-slim или distroless), содержащий только интерпретатор Python и необходимые runtime-библиотеки (libpq5). В эту стадию копируется только подготовленное виртуальное окружение из стадии сборщика.3

Такой подход гарантирует, что в продакшн попадает только то, что необходимо для работы приложения, снижая размер образа на 50-90% по сравнению с монолитным подходом.13

## **4\. Стратегии реализации для Python: Углубленный анализ**

### **4.1. Управление зависимостями: Virtualenv против User Install**

Существует два основных паттерна переноса зависимостей между стадиями в многоступенчатой сборке.

#### **Паттерн А: Перенос виртуального окружения (Рекомендуемый)**

Использование модуля venv является стандартом де\-факто для изоляции зависимостей.

- **Механизм:** В стадии builder создается виртуальное окружение (например, в /opt/venv). Все пакеты устанавливаются в него. Затем директория /opt/venv целиком копируется в финальную стадию.15
- **Преимущества:** Полная изоляция от системных пакетов. Простота копирования (одна директория). Гарантия целостности путей интерпретатора, так как venv содержит свою структуру bin/, lib/ и include/.
- **Реализация:**  
  Dockerfile  
  \# Stage 1: Builder  
  FROM python:3.10-slim as builder  
  RUN apt-get update && apt-get install \-y build-essential  
  RUN python \-m venv /opt/venv  
  ENV PATH="/opt/venv/bin:$PATH"  
  COPY requirements.txt.  
  RUN pip install \-r requirements.txt

  \# Stage 2: Runtime  
  FROM python:3.10-slim  
  COPY \--from=builder /opt/venv /opt/venv  
  ENV PATH="/opt/venv/bin:$PATH"

  В данном примере переменная окружения PATH играет критическую роль: она заставляет систему искать исполняемые файлы (например, uvicorn, gunicorn) в директории виртуального окружения, а не в системных путях.15

#### **Паттерн Б: Сборка Wheel-пакетов**

Этот метод предполагает сборку бинарных пакетов (.whl) в первой стадии и их установку во второй.

- **Механизм:** pip wheel \-w /wheels \-r requirements.txt в сборщике, затем pip install \--no-index /wheels/\* в финальном образе.8
- **Недостатки:** Требует наличия pip в финальном образе (что не всегда желательно для distroless образов) и повторного процесса установки, который, хоть и быстр, но создает дополнительные слои файловой системы.

#### **Паттерн В: Pip Install Target**

Использование флага \--target позволяет установить пакеты в произвольную директорию.

- **Проблема:** Этот метод часто приводит к проблемам с путями импорта (PYTHONPATH) и отсутствием бинарных файлов в PATH. Также он может конфликтовать со структурой системных пакетов Debian.17 Исследование показывает, что метод venv более надежен и предсказуем для Docker-сборок.19

### **4.2. Современные пакетные менеджеры: Poetry и UV**

С появлением современных инструментов управления зависимостями, таких как Poetry и uv (от Astral), стратегии многоступенчатых сборок эволюционировали.

- **Poetry:** Требует установки самого Poetry в стадии сборки. Лучшая практика — экспортировать зависимости в requirements.txt или использовать настройку virtualenvs.in-project \= true, чтобы создать venv внутри проекта и скопировать его.20 Однако, Poetry добавляет оверхед при сборке из\-за разрешения графа зависимостей.
- **UV:** Новый инструмент uv, написанный на Rust, позиционируется как сверхбыстрая замена pip. В Docker-сборках uv демонстрирует значительное ускорение установки зависимостей (в 10-100 раз быстрее pip). uv поддерживает специальный режим копирования venv, оптимизированный для Docker, и компиляцию байт-кода (.pyc), что ускоряет старт контейнера.19 Использование uv в многоступенчатых сборках позволяет сократить время стадии builder до минимума.

### **4.3. Базовые образы: Alpine против Slim vs Distroless**

Выбор базового образа является фундаментальным решением, влияющим на размер, производительность и совместимость.

| Тип образа                      | Описание                                                               | Размер (сжатый) | Совместимость (C-extensions) | Рекомендация             |
| :------------------------------ | :--------------------------------------------------------------------- | :-------------- | :--------------------------- | :----------------------- |
| **Full (python:3.10)**          | Основан на Debian (Bookworm/Bullseye). Содержит все системные утилиты. | \~300 МБ        | Идеальная (glibc)            | Только для разработки    |
| **Slim (python:3.10-slim)**     | Урезанная версия Debian. Удалены мануалы, документация, лишние либы.   | \~45-50 МБ      | Идеальная (glibc)            | **Золотой стандарт**     |
| **Alpine (python:3.10-alpine)** | Основан на Alpine Linux (musl libc). Экстремально мал.                 | \~15-20 МБ      | Проблематичная (musl)        | Только для экспертов     |
| **Distroless**                  | Google образы без Shell. Только runtime.                               | \~20 МБ         | Хорошая (glibc)              | Для высокой безопасности |

**Проблема Alpine Linux в Python:** Вопреки популярному мифу, Alpine часто является _плохим_ выбором для Python.23 Python-колеса (wheels) в PyPI обычно скомпилированы под manylinux (использующий glibc). Alpine использует musl libc. Это означает, что pip не может использовать бинарные колеса и вынужден компилировать всё из исходников. Это приводит к:

1. Гигантскому времени сборки (компиляция pandas/numpy может занять 20+ минут).
2. Необходимости устанавливать полный набор компиляторов в образ (делая Alpine тяжелее Slim на этапе сборки).
3. Потенциальным багам производительности и несовместимости на уровне C-API.23

Сравнительные данные показывают, что python:3.10-slim обеспечивает лучший баланс между размером и совместимостью, избавляя от головной боли с кросс-компиляцией musl.24

## **5\. Аспект безопасности: CVE и поверхность атаки**

### **5.1. Анализ уязвимостей (CVE)**

Размер образа имеет прямую корреляцию с количеством уязвимостей. Монолитные образы, основанные на полноценных дистрибутивах (Ubuntu, Debian), содержат тысячи пакетов (Perl, systemd утилиты, mount, gnupg и т.д.), многие из которых могут иметь известные уязвимости (CVE). Даже если приложение не использует эти библиотеки, сканеры безопасности (Snyk, Trivy, Grype) будут блокировать деплой таких образов в соответствии с политиками безопасности компании.26  
Сравнительная статистика уязвимостей (данные 2024-2025 гг.):  
По данным отчетов сканирования Snyk и OpenCVE:

- **Python 3.10 Full (Debian 11/12):** Обычно содержит **сотни** уязвимостей различной критичности из\-за широкого набора системных библиотек.28
- **Python 3.10 Slim:** Количество уязвимостей радикально ниже. Например, отчет Snyk для python:3.10-slim показывает около **23 уязвимостей**, большинство из которых имеют низкий (Low) уровень критичности и часто не имеют фикса в upstream Debian (например, незначительные баги в glibc или tar).29
- **Python 3.10 Alpine:** Часто показывает **0-5 уязвимостей**, что делает его привлекательным для compliance-отчетов, однако операционные риски (см. раздел 4.3) часто перевешивают этот плюс.30

Использование многоступенчатой сборки с финальным образом slim позволяет отфильтровать уязвимости инструментов сборки (GCC, git, ssh-client), которые часто имеют высокий рейтинг критичности. Например, уязвимости в git или curl, необходимые только для получения зависимостей, не попадут в продовый образ.3

### **5.2. Минимизация поверхности атаки**

Помимо CVE, существует концепция «поверхности атаки» (Attack Surface). В случае компрометации приложения (например, через уязвимость Log4Shell или SQL Injection), злоумышленник получает доступ к оболочке контейнера. В монолитном образе он находит богатый инструментарий: wget для скачивания эксплойтов, gcc для их компиляции на месте, netcat для организации обратного шелла.  
В оптимизированном многоступенчатом образе (особенно на базе Distroless или минимального Slim) эти инструменты отсутствуют. Злоумышленник оказывается в ограниченной среде, где нет даже пакетного менеджера для установки инструментов (apt/apk отсутствуют или кэши очищены). Это реализует принцип Defense in Depth (эшелонированная защита).3

## **6\. Экономическая и операционная эффективность**

### **6.1. Экономика облачного хранения и трафика**

Оптимизация образов имеет прямое денежное выражение. Облачные провайдеры (AWS, Google Cloud, Azure) тарифицируют хранение образов в реестрах (ECR, GCR, ACR) и исходящий трафик.  
Расчет совокупной стоимости владения (TCO):  
Рассмотрим сценарий активной разработки: 20 микросервисов, 10 билдов в день на каждый сервис.

- **Монолитный подход (1.2 ГБ/образ):**
  - 20 сервисов \* 10 билдов \* 1.2 ГБ \= 240 ГБ новых данных в день.
  - За месяц: \~7.2 ТБ данных.
- **Многоступенчатый подход (150 МБ/образ):**
  - 20 сервисов \* 10 билдов \* 0.15 ГБ \= 30 ГБ в день.
  - За месяц: \~900 ГБ данных.

**Стоимость AWS ECR (2025):**

- Хранение: $0.10 за ГБ/месяц.32
- Трафик (Data Transfer Out): \~$0.09 за ГБ (при передаче в другие регионы или интернет).32

Разница в затратах на хранение и трафик может достигать тысяч долларов в год для крупных организаций, просто за счет удаления «воздуха» из образов. Кроме того, новые классы хранилищ, такие как ECR Archive, позволяют экономить на старых образах, но снижение исходного размера дает немедленный эффект.33

### **6.2. Производительность Kubernetes и масштабирование**

Наиболее критичный параметр для высоконагруженных систем — **время холодного старта** (Cold Start). При масштабировании кластера Kubernetes (HPA) новые узлы (Nodes) должны сначала скачать образ, прежде чем запустить под.

- Скачивание 1.2 ГБ по сети 1 Gbps занимает теоретически \~10 секунд, на практике с учетом распаковки слоев и I/O диска — до 30-40 секунд.4
- Скачивание 150 МБ занимает \~2-3 секунды.

В ситуации резкого всплеска трафика (Spike traffic) задержка в 30 секунд может привести к отказу в обслуживании (503 Service Unavailable) и потере клиентов. Оптимизированные образы позволяют системе авто-масштабирования реагировать на изменения нагрузки почти мгновенно. Также снижается нагрузка на дисковую подсистему узлов (Disk Pressure), что повышает стабильность работы других подов на том же узле.5

## **7\. Практическое руководство: Бенчмарки и сравнение**

Для демонстрации эффективности сравним сборку типичного приложения FastAPI с зависимостями для Data Science (Pandas, NumPy).

### **7.1. Монолитная сборка (Anti-Pattern)**

**Dockerfile:**

Dockerfile

FROM python:3.10  
WORKDIR /app  
COPY requirements.txt.  
RUN pip install \-r requirements.txt  
COPY..  
CMD \["uvicorn", "main:app", "--host", "0.0.0.0"\]

**Характеристики:**

- **Базовый образ:** python:3.10 (\~300 МБ сжатый / \~900 МБ распакованный).
- **Слой зависимостей:** Устанавливаются в системные пути. Компиляторы остаются.
- **Итоговый размер:** \~1.2 ГБ.
- **Количество слоев:** Минимум, но они огромные.
- **Безопасность:** Высокий риск (CVE \> 100).

### **7.2. Оптимизированная многоступенчатая сборка**

**Dockerfile:**

Dockerfile

\# Stage 1: Builder  
FROM python:3.10-slim as builder

\# Установка системных зависимостей для сборки  
RUN apt-get update && apt-get install \-y \--no-install-recommends \\  
 build-essential \\  
 libpq-dev \\  
 && rm \-rf /var/lib/apt/lists/\*

\# Создание виртуального окружения  
RUN python \-m venv /opt/venv  
ENV PATH="/opt/venv/bin:$PATH"

\# Установка Python-зависимостей  
COPY requirements.txt.  
\# Используем кэш pip для ускорения повторных сборок  
RUN \--mount=type=cache,target=/root/.cache/pip \\  
 pip install \-r requirements.txt

\# Stage 2: Runtime  
FROM python:3.10-slim

\# Установка только runtime-библиотек (без dev-headers)  
RUN apt-get update && apt-get install \-y \--no-install-recommends \\  
 libpq5 \\  
 && rm \-rf /var/lib/apt/lists/\*

\# Копирование виртуального окружения из builder  
COPY \--from=builder /opt/venv /opt/venv

\# Настройка путей и пользователя  
ENV PATH="/opt/venv/bin:$PATH"  
WORKDIR /app  
COPY..

\# Создание непривилегированного пользователя  
RUN useradd \-m appuser && chown \-R appuser /app  
USER appuser

CMD \["uvicorn", "main:app", "--host", "0.0.0.0"\]

**Характеристики:**

- **Базовый образ:** python:3.10-slim (\~50 МБ сжатый / \~120 МБ распакованный).
- **Слой зависимостей:** Только скомпилированные библиотеки и необходимые shared objects (libpq5).
- **Итоговый размер:** \~350 МБ.
- **Оптимизация:** Уменьшение размера в **3.4 раза**.
- **Безопасность:** Минимальный CVE, отсутствует GCC, запуск от non-root пользователя (appuser).11

### **7.3. Сводная таблица сравнения**

| Метрика                          | Монолитная сборка    | Многоступенчатая сборка  | Выигрыш         |
| :------------------------------- | :------------------- | :----------------------- | :-------------- |
| **Размер образа (Uncompressed)** | \~1.15 GB            | \~340 MB                 | **\-70%**       |
| **Время Pull (100 Mbps)**        | \~95 сек             | \~28 сек                 | **3x быстрее**  |
| **Количество CVE (High/Crit)**   | Высокое (\>50)       | Низкое (0-5)             | **Безопаснее**  |
| **Наличие компиляторов**         | Да (GCC, Make)       | Нет                      | **Hardening**   |
| **Использование кэша CI**        | Низкое (частые сбои) | Высокое (изоляция слоев) | **Эффективнее** |
| **Сложность Dockerfile**         | Низкая               | Средняя                  | Требует навыка  |

### **7.4. Кейс с TensorFlow и ML-моделями**

Особое внимание стоит уделить ML-контейнерам. Библиотеки вроде TensorFlow или PyTorch могут весить гигабайты. В монолитной сборке часто приходится устанавливать libhdf5-dev и другие тяжелые dev-пакеты. Исследования показывают, что правильное разделение на build/runtime stages для таких проектов (копирование только site-packages или venv) и исключение кэша pip (rm \-rf /root/.cache/pip или использование \--no-cache-dir) критично. В одном из кейсов размер образа с TensorFlow удалось снизить с 3+ ГБ до 1.5 ГБ, удалив инструментарий сборки и временные файлы.35

## **8\. Дополнительные техники оптимизации**

Для достижения предельной эффективности многоступенчатые сборки следует комбинировать с дополнительными практиками.

### **8.1. Файл .dockerignore — первая линия обороны**

Часто образы «раздуваются» из\-за случайного включения локальных файлов: директории .git (которая может весить больше самого кода), локальных виртуальных окружений (venv/), кэшей (\_\_pycache\_\_, .pytest_cache) и файлов документации. Корректно настроенный .dockerignore предотвращает передачу этих файлов в контекст сборки (Build Context), ускоряя отправку контекста демону Docker и уменьшая финальный образ.1

### **8.2. Сортировка аргументов и объединение команд**

Хотя многоступенчатые сборки нивелируют необходимость склеивания команд в _промежуточных_ стадиях, в _финальной_ стадии это остается актуальным. Команды RUN apt-get update && apt-get install... && rm \-rf /var/lib/apt/lists/\* должны выполняться в одном слое, чтобы кэш apt не оставался в слое образа.8 Также рекомендуется сортировать списки пакетов по алфавиту для удобства аудита и предотвращения дубликатов.1

### **8.3. Кэширование пакетов (BuildKit Cache Mounts)**

Использование \--mount=type=cache,target=/root/.cache/pip в стадии сборки позволяет сохранять скачанные wheel-пакеты между сборками, даже если requirements.txt изменился. Это не уменьшает размер финального образа, но радикально ускоряет время сборки в CI/CD.10

## **9\. Заключение**

Переход от монолитных к многоступенчатым сборкам в экосистеме Python не является вопросом вкуса или перфекционизма — это строгая инженерная необходимость для создания масштабируемых, безопасных и экономически эффективных облачных систем.  
Исследование подтверждает, что использование многоступенчатых сборок в сочетании с образами python-slim и виртуальными окружениями обеспечивает:

1. **Снижение размера артефактов на 60-80%**, что напрямую уменьшает облачные расходы.
2. **Укрепление контура безопасности** за счет минимизации поверхности атаки и устранения инструментов двойного назначения (компиляторов).
3. **Повышение операционной надежности** благодаря ускорению деплоя и масштабирования.

Рекомендуется принять описанный паттерн (Slim Builder \+ Venv Copy \+ Slim Runtime) в качестве корпоративного стандарта для всех Python-сервисов, заменяя устаревшие монолитные практики.

#### **Источники**

1. Building best practices \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/build/building/best-practices/](https://docs.docker.com/build/building/best-practices/)
2. How to Build Smaller Container Images: Docker Multi-Stage Builds | iximiuz Labs, дата последнего обращения: декабря 22, 2025, [https://labs.iximiuz.com/tutorials/docker-multi-stage-builds](https://labs.iximiuz.com/tutorials/docker-multi-stage-builds)
3. The Hidden Cost of Docker Images: Why Multi-Stage Builds Are Essential in 2025, дата последнего обращения: декабря 22, 2025, [https://claudiomas.medium.com/the-hidden-cost-of-docker-images-why-multi-stage-builds-are-essential-in-2025-0b9cc0339499](https://claudiomas.medium.com/the-hidden-cost-of-docker-images-why-multi-stage-builds-are-essential-in-2025-0b9cc0339499)
4. Cost-effective resources \- Container Build Lens \- AWS Documentation, дата последнего обращения: декабря 22, 2025, [https://docs.aws.amazon.com/wellarchitected/latest/container-build-lens/cost-effective-resources.html](https://docs.aws.amazon.com/wellarchitected/latest/container-build-lens/cost-effective-resources.html)
5. Small is Beautiful: How Container Size Impacts Deployment and Resource Usage, дата последнего обращения: декабря 22, 2025, [https://www.fullstack.com/labs/resources/blog/small-is-beautiful-how-container-size-impacts-deployment-and-resource-usage](https://www.fullstack.com/labs/resources/blog/small-is-beautiful-how-container-size-impacts-deployment-and-resource-usage)
6. What is Container Security and Why It Matters | Docker, дата последнего обращения: декабря 22, 2025, [https://www.docker.com/blog/container-security-and-why-it-matters/](https://www.docker.com/blog/container-security-and-why-it-matters/)
7. Short introduction to Docker \- .Stat Suite documentation \- GitLab, дата последнего обращения: декабря 22, 2025, [https://sis-cc.gitlab.io/dotstatsuite-documentation/install-docker/intro/](https://sis-cc.gitlab.io/dotstatsuite-documentation/install-docker/intro/)
8. Docker Best Practices for Python Developers \- TestDriven.io, дата последнего обращения: декабря 22, 2025, [https://testdriven.io/blog/docker-best-practices/](https://testdriven.io/blog/docker-best-practices/)
9. Multi-stage builds \- Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/build/building/multi-stage/](https://docs.docker.com/build/building/multi-stage/)
10. Improving docker build time for pip based python application \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/71187980/improving-docker-build-time-for-pip-based-python-application](https://stackoverflow.com/questions/71187980/improving-docker-build-time-for-pip-based-python-application)
11. Multi-stage builds | Docker Docs, дата последнего обращения: декабря 22, 2025, [https://docs.docker.com/get-started/docker-concepts/building-images/multi-stage-builds/](https://docs.docker.com/get-started/docker-concepts/building-images/multi-stage-builds/)
12. Harness the power of Docker multi-stage builds \- Engineering at Axis, дата последнего обращения: декабря 22, 2025, [https://engineeringat.axis.com/docker-multi-stage-builds/](https://engineeringat.axis.com/docker-multi-stage-builds/)
13. Docker Multi-Stage Builds for Python Developers: A Complete Guide \- Collabnix, дата последнего обращения: декабря 22, 2025, [https://collabnix.com/docker-multi-stage-builds-for-python-developers-a-complete-guide/](https://collabnix.com/docker-multi-stage-builds-for-python-developers-a-complete-guide/)
14. Building Faster, Smaller, and Cleaner Python Docker Images with Multi-Stage Builds, дата последнего обращения: декабря 22, 2025, [https://manabpokhrel7.medium.com/building-faster-smaller-and-cleaner-python-docker-images-with-multi-stage-builds-0da6983a0593](https://manabpokhrel7.medium.com/building-faster-smaller-and-cleaner-python-docker-images-with-multi-stage-builds-0da6983a0593)
15. Why I Still Use Python Virtual Environments in Docker \- Hynek Schlawack, дата последнего обращения: декабря 22, 2025, [https://hynek.me/articles/docker-virtualenv/](https://hynek.me/articles/docker-virtualenv/)
16. Activate python virtualenv in Dockerfile \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/48561981/activate-python-virtualenv-in-dockerfile](https://stackoverflow.com/questions/48561981/activate-python-virtualenv-in-dockerfile)
17. Multistage Dockefile \- how to copy/install dependencies the right way \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/75427676/multistage-dockefile-how-to-copy-install-dependencies-the-right-way](https://stackoverflow.com/questions/75427676/multistage-dockefile-how-to-copy-install-dependencies-the-right-way)
18. Installing packages close to the project root \- Python Discussions, дата последнего обращения: декабря 22, 2025, [https://discuss.python.org/t/installing-packages-close-to-the-project-root/6915](https://discuss.python.org/t/installing-packages-close-to-the-project-root/6915)
19. Using uv in Docker \- Astral Docs, дата последнего обращения: декабря 22, 2025, [https://docs.astral.sh/uv/guides/integration/docker/](https://docs.astral.sh/uv/guides/integration/docker/)
20. Document docker poetry best practices · python-poetry · Discussion \#1879 \- GitHub, дата последнего обращения: декабря 22, 2025, [https://github.com/orgs/python-poetry/discussions/1879](https://github.com/orgs/python-poetry/discussions/1879)
21. Integrating Python Poetry with Docker \- Stack Overflow, дата последнего обращения: декабря 22, 2025, [https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker](https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker)
22. Best practices for using Python & uv inside Docker \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/Python/comments/1o3p4bf/best_practices_for_using_python_uv_inside_docker/](https://www.reddit.com/r/Python/comments/1o3p4bf/best_practices_for_using_python_uv_inside_docker/)
23. Alpine vs python-slim for deploying python data science stack? : r/docker \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/docker/comments/g5hb93/alpine_vs_pythonslim_for_deploying_python_data/](https://www.reddit.com/r/docker/comments/g5hb93/alpine_vs_pythonslim_for_deploying_python_data/)
24. The best Docker base image for your Python application (May 2024), дата последнего обращения: декабря 22, 2025, [https://pythonspeed.com/articles/base-image-python-docker-images/](https://pythonspeed.com/articles/base-image-python-docker-images/)
25. Alpine, Slim, Bullseye, Bookworm, Noble — Different Docker Images Explained \- Medium, дата последнего обращения: декабря 22, 2025, [https://medium.com/@faruk13/alpine-slim-bullseye-bookworm-jammy-noble-differences-in-docker-images-explained-d9aa6efa23ec](https://medium.com/@faruk13/alpine-slim-bullseye-bookworm-jammy-noble-differences-in-docker-images-explained-d9aa6efa23ec)
26. Docker Image Optimization: A Comprehensive Guide to Creating Smaller and More Efficient Containers \- DEV Community, дата последнего обращения: декабря 22, 2025, [https://dev.to/rajeshgheware/docker-image-optimization-a-comprehensive-guide-to-creating-smaller-and-more-efficient-containers-2g70](https://dev.to/rajeshgheware/docker-image-optimization-a-comprehensive-guide-to-creating-smaller-and-more-efficient-containers-2g70)
27. Corporate IT have banned all versions of python lower than the latest \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/Python/comments/180lq7i/corporate_it_have_banned_all_versions_of_python/](https://www.reddit.com/r/Python/comments/180lq7i/corporate_it_have_banned_all_versions_of_python/)
28. Vulnerability report for Docker python:3.10.0a5 | Snyk, дата последнего обращения: декабря 22, 2025, [https://snyk.io/test/docker/python%3A3.10.0a5](https://snyk.io/test/docker/python%3A3.10.0a5)
29. Vulnerability report for Docker python:3.10-slim \- Snyk, дата последнего обращения: декабря 22, 2025, [https://snyk.io/test/docker/python%3A3.10-slim](https://snyk.io/test/docker/python%3A3.10-slim)
30. Vulnerability report for Docker python:3.10.4-alpine \- Snyk, дата последнего обращения: декабря 22, 2025, [https://snyk.io/test/docker/python%3A3.10.4-alpine](https://snyk.io/test/docker/python%3A3.10.4-alpine)
31. The Hidden Cost of Docker Images: Why Multi-Stage Builds Are Essential in 2025, дата последнего обращения: декабря 22, 2025, [https://dev.to/klaus82/the-hidden-cost-of-docker-images-why-multi-stage-builds-are-essential-in-2025-3knk](https://dev.to/klaus82/the-hidden-cost-of-docker-images-why-multi-stage-builds-are-essential-in-2025-3knk)
32. Amazon ECR Pricing \- Elastic Container Registry \- AWS, дата последнего обращения: декабря 22, 2025, [https://aws.amazon.com/ecr/pricing/](https://aws.amazon.com/ecr/pricing/)
33. Amazon ECR introduces archive storage class for rarely accessed container images \- AWS, дата последнего обращения: декабря 22, 2025, [https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-ecr-archive-storage-class-container-images/](https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-ecr-archive-storage-class-container-images/)
34. Do large containers (\>600MB) affect performance? : r/kubernetes \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/kubernetes/comments/1cm8g4d/do_large_containers_600mb_affect_performance/](https://www.reddit.com/r/kubernetes/comments/1cm8g4d/do_large_containers_600mb_affect_performance/)
35. Optimizing Dockerized FastAPI with TensorFlow: How to reduce a 1.57GB Image Size?, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/FastAPI/comments/1e1lal6/optimizing_dockerized_fastapi_with_tensorflow_how/](https://www.reddit.com/r/FastAPI/comments/1e1lal6/optimizing_dockerized_fastapi_with_tensorflow_how/)
36. How I Reduced Docker Image Size from 588 MB to Only 47.7 MB \- A whomping 91.89 \- Reddit, дата последнего обращения: декабря 22, 2025, [https://www.reddit.com/r/docker/comments/1f1wqnb/how_i_reduced_docker_image_size_from_588_mb_to/](https://www.reddit.com/r/docker/comments/1f1wqnb/how_i_reduced_docker_image_size_from_588_mb_to/)
