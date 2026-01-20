# docker-compose.yml Specification

## 1. Meta Information

- Domain: Infrastructure
- Complexity: Medium
- Language: Bash
- Frameworks: Docker, Docker Compose
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Context for Creator: Этот Ритуал-скрижаль разворачивает мониторинговый трезубец из трех духов: приложение (app), Стража Метрик (prometheus) и Мудрецу Видения (grafana). Все они связаны единым эфиром monitoring-net и слушают порты 8000, 9090 и 3000 соответственно.  
Instruction for AI: Этот раздел объясняет «ПОЧЕМУ» — зачем нужен этот файл и какие бизнес-задачи он решает.

Легенда: С помощью docker-compose.yml создаётся автономная цитадель наблюдения, где приложение разворачивается, метрики собираются Стражем, а Ведающая Графика Графана превращает данные в видения для команды.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: JSON
- Schema:
  - InputData
    - path: string — путь к docker-compose.yml
    - action?: string — возможное действие (например, "deploy", "validate")
    - dryRun?: boolean — симулятивный режим без запуска

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: JSON
- Success Criteria: Exit Code 0
- Schema:
  - OutputResult
    - success: boolean
    - exitCode: number
    - message?: string
    - details?: {
      servicesRunning: string[]
      networksPresent: string[]
      }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Принять путь к файлу docker-compose.yml из входных данных и проверить доступность файла.
2. Распознать версию скрипта: убедиться, что версия составлена как "3.8" и что файл читается без ошибок.
3. Просмотреть секцию services и зафиксировать три существа: app, prometheus, grafana.
4. Для service app проверить: build.context равно ".", ports содержит 8000:8000, network monitoring-net, переменная PORT=8000 в environment.
5. Для service prometheus проверить: image равен prom/prometheus:latest, container_name равен prometheus_guardian, volumes содержит ./prometheus.yml:/etc/prometheus/prometheus.yml, ports содержит 9090:9090, depends_on включает app, network monitoring-net.
6. Для service grafana проверить: image равен grafana/grafana:latest, container_name равен grafana_seer, ports содержит 3000:3000, depends_on включает prometheus, network monitoring-net.
7. Для секции networks проверить наличие monitoring-net с driver: bridge.
8. Верифицировать связи между сервисами: app <- prometheus <- grafana, и что все порты открыты как указано.
9. Вернуть итог в виде Outputs, включив статус и детальную сводку по запущенным сервисам и сетям.

### 4.2. Declarative Content (Для конфигураций и данных)

Указ Ткачу и точные данные для 1-в-1:

- Версия: 3.8
- Сети: monitoring-net (bridge)
- Сервисы:
  - app
    - Build context: .
    - Ports: "8000:8000"
    - Networks: monitoring-net
    - Environment: PORT=8000
  - prometheus
    - Image: prom/prometheus:latest
    - Container name: prometheus_guardian
    - Volumes: ./prometheus.yml:/etc/prometheus/prometheus.yml
    - Ports: "9090:9090"
    - Depends_on: app
    - Networks: monitoring-net
  - grafana
    - Image: grafana/grafana:latest
    - Container name: grafana_seer
    - Ports: "3000:3000"
    - Depends_on: prometheus
    - Networks: monitoring-net

## 5. Structural Decomposition (Декомпозиция структуры)

- Version: "3.8"
- Services:
  - app
  - prometheus
  - grafana
- Build:
  - app.context = .
- Images:
  - prom/prometheus:latest
  - grafana/grafana:latest
- Networks:
  - monitoring-net (driver: bridge)
- Volumes:
  - ./prometheus.yml:/etc/prometheus/prometheus.yml (для prometheus)
- Dependencies:
  - prometheus зависит от app
  - grafana зависит от prometheus
- Container Names:
  - prometheus_guardian
  - grafana_seer

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Version: 3.8
- Performance: Стандартная конфигурация Docker. Ожидается нормальное поведение на обычной машине (CPU/память по умолчанию для докер-демона).
- Concurrency: Параллельный запуск допускается, но порядок обеспечивается depends_on (prometheus после app, grafana после prometheus).
- Dependencies: Docker Engine + Docker Compose; сеть bridge для monitoring-net.

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде; используйте среды (.env) там, где требуется секретная информация.
- DO NOT выводить сырые данные в консоль в продакшн-режиме.
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты.
- DO NOT менять версии или пути во время реконструкции.
- DO NOT изменять имена или маршрутизацию портов без явной необходимости.

## 7. Verification & Testing (Верификация)

1. Герхин-описания (1-2 сценария):
   Feature: Docker Compose Deployment of Monitoring Stack
   Scenario: Successful deployment
   Given Docker and docker-compose installed
   When запускать docker-compose up -d в директории с docker-compose.yml
   Then все сервисы запускаются: app на порту 8000, Prometheus на 9090, Grafana на 3000, сеть monitoring-net создана, возвращается код 0

Feature: Deployment failure when prometheus.yml is missing
Scenario: Missing prometheus.yml
Given файл prometheus.yml отсутствует или недоступен по указанному пути
When запускать docker-compose up -d
Then Prometheus не запускается, выводится ошибка в логе, возвращается не-нулевой код

ИЗГОТОВЛЕННЫЙ АРТЕФАКТ: docker-compose.yml

ИСТОЧНЫЙ КОД ПРЕДОСТАВЛЕН В ВОРКЕ ВЫШЕ КАК ЭТАЛОННЫЙ РИТУАЛ, СКРЫТЫЙ ПЛАНЕТАРИЙИ ДУХОВ.
