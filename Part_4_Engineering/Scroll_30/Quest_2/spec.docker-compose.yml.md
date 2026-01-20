# docker-compose.yml Specification

1. Meta Information

- Domain: Infrastructure
- Complexity: Medium
- Language: Bash
- Frameworks: Docker, Docker Compose
- Context: Independent Artifact

2. Goal & Purpose (Цель и Назначение)
   Легенда: свиток оркестрации призывает в одну citadel три духа командного мира: app — Артефакт, исполняющий логику сервиса; prometheus — Страж измерений, собирающий метрики; grafana — Провидец зеркал, визуализирующий данные. Они соединены единым магическим каналом monitoring-net; порталы открыты на портах 8000, 9090 и 3000. Цель файла — обеспечить единое развёртывание и мониторинг этих компонентов через docker-compose, с предписанной последовательностью старта и правильной подстановкой конфигураций.

3. Interface Contract (Интерфейсный Контракт)

3.1 Inputs (Входы)

- Source: CLI Args | STDIN | API Request
- Format: JSON
- Schema (TypeScript-подобное описание форм Shapes):
  - interface InputData {
    version: string;
    services: Record<string, ServiceSpec>;
    networks?: Record<string, NetworkSpec>;
    }
  - interface ServiceSpec {
    image?: string;
    build?: { context?: string; dockerfile?: string };
    ports?: string[];
    networks?: string[];
    environment?: string[] | Record<string, string>;
    container_name?: string;
    depends_on?: string[];
    volumes?: string[];
    }
  - interface NetworkSpec {
    driver?: string;
    }

    3.2 Outputs (Выходы)

- Destination: STDOUT | File | API Response
- Format: JSON | CSV | Text
- Success Criteria: Exit Code 0
- Schema (TypeScript-подобное описание форм Shapes):
  - interface OutputResult {
    success: boolean;
    message?: string;
    data?: any;
    }

4. Implementation Details (The Source DNA / Исходный Код)

4.1 Algorithmic Logic (Для исполняемого кода)

1. Загрузить конфигурационный документ docker-compose.yml и разобрать его содержимое.
2. Проверить, что версия манифеста поддерживается исполнительной средой (например, версия 3.8).
3. Пройти по каждому сервису в секции services и аппроксимировать его требования: образ или билд контекст, порты, сети, переменные окружения, тома и зависимости.
4. Проверить наличие секции networks и корректность указанных сетевых связей; убедиться, что для каждого сервиса перечислены сети, к которым он должен подключаться.
5. В случае использования Build контекста подтвердить путь к контексту и допустимость файловой структуры проекта.
6. При запуске команды docker-compose up выполнить старт сервисов с учётом зависимостей (depends_on) и сетевых связей, позволяя частичному параллельному развертыванию там, где зависимости не ограничены.
7. Обеспечить корректное отображение логов, ошибок и статусов запуска для последующей диагностики.

4.2 Declarative Content (Для конфигураций и данных)

- Объекты и их связи соответствуют исходному артефакту: три сервиса app, prometheus, grafana, и единственная сеть monitoring-net, как описано в файле.
- Пояснения к связям и настройкам — без изменений синтаксиса и без копирования самого YAML.

5. Structural Decomposition (Декомпозиция структуры)

- version: версия манифеста
- services: словарь из трёх служб
  - app: image, build.context, ports, networks, environment
  - prometheus: image, container_name, volumes, ports, depends_on, networks
  - grafana: image, container_name, ports, depends_on, networks
- networks: monitoring-net с driver bridge

6. System Context & Constraints (Системный контекст и Ограничения)

6.1 Technical Constraints

- Performance: Standard CPU
- Concurrency: Async (Docker Compose разворачивает контейнеры параллельно там, где зависимости не ограничивают порядок)
- Dependencies: Docker, Docker Compose, доступ к образам prom/prometheus:latest, grafana/grafana:latest, codex/amulet-30-2 и т. д.; локальные файлы проекта (prometheus.yml)

  6.2 Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main orchestration loop
- DO NOT wrap конфигурационные файлы (.yaml, .json) в исполняемые скрипты
- DO NOT менять версии или пути во время реконструкции

7. Verification & Testing (Верификация)

Геркин сценарии

Feature: Docker Compose Deployment
Scenario: Successful deployment
Given Docker daemon is running and docker-compose.yml is valid
When docker-compose up -d is executed
Then app, prometheus и grafana контейнеры запускаются
And порты 8000, 9090 и 3000 доступны
And сеть monitoring-net создана

Scenario: Failure due to missing prometheus.yml
Given Docker daemon is running and prometheus.yml не монтирован
When docker-compose up -d выполняется
Then запуск Prometheus завершается с ошибкой и соответствующий контейнер помечается как не готовый

ИССЛЕДУЕМЫЙ АРТЕФАКТ: docker-compose.yml

ИСТОЧНЫЙ КОД

- Гримуар: Пробуждение Цитадели
- Версия: 3.8
- Службы:
  - app
    образ: codex/amulet-30-2
    build context: .
    ports: 8000:8000
    networks: monitoring-net
    environment: PORT=8000
  - prometheus
    образ: prom/prometheus:latest
    container_name: prometheus_guardian
    volumes: ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports: 9090:9090
    depends_on: app
    networks: monitoring-net
  - grafana
    образ: grafana/grafana:latest
    container_name: grafana_seer
    ports: 3000:3000
    depends_on: prometheus
    networks: monitoring-net
- Сети:
  - monitoring-net: bridge

ИНСТРУКЦИИ ПО РАЗДЕЛАМ

- Meta Information: Определяй Стихию (Язык). Context укажи только если есть явные импорты.
- Goal & Purpose: Легенда — без вступлений. Зачем нужен этот файл.
- Interface Contract: Описывай входящие/исходящие данные (TypeScript/Python types) без вставки кода в реализации.
- Implementation Details: Пошаговый алгоритм без квадратных скобок и заглушек.
- Declarative Content: INVENTORY — преврати данные в маркированный список с эмодзи. Не копируй код.

Примечание синергии:

- Скрипт стал бы подлинной частью инфраструктурного артефакта, в котором каждый компонент несёт свой характер: app — сердце операции, prometheus — хроникер измерений, grafana — оракул визуализации. Их слияние образует устойчивую экосистему мониторинга, управляемую магией docker-compose.
