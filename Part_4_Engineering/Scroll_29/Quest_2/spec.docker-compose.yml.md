# docker-compose.yml Specification

## 1. Meta Information

- Domain: Infrastructure
- Complexity: Medium
- Language: Bash
- Frameworks: Docker, Docker Compose
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Легенда: На свитке разворачивается тройственный ритуал наблюдения над полем сервиса. Первый Голем — Говорящий Амулет app — открывает портал к своему миру и несет порты 8000 наружу. Второй Страж — Prometheus — питается данными с Амулета и держит жезл мониторинга на порту 9090, собирая хроники эфира. Третий Провидец — Grafana — сворачивает эти хроники в визуальные предзнаменования на порту 3000. Все соединено в тайной сети monitoring-net, дабы хаос не прорвался в Свиток. Этот файл задает ритуал разворачивания и взаимопроверки этих сущностей в единый канал наблюдения.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args | STDIN | API Request | Kafka Topic | Smart Contract Call
- Format: JSON | Text | Binary | Stream
- Schema:
  интерфейс InputData {
  source: 'CLI' | 'STDIN' | 'API' | 'Kafka' | 'SmartContract';
  payload?: string | object;
  options?: {
  overrides?: string[];
  dryRun?: boolean;
  };
  }

### 3.2. Outputs (Выходы)

- Destination: STDOUT | File | Database | API Response | Event Log
- Format: JSON | CSV | Text
- Success Criteria: Exit Code 0 | 200 OK | File Created
- Schema:
  интерфейс OutputResult {
  destination: 'STDOUT' | 'File' | 'Database' | 'API Response' | 'Event Log';
  format: 'JSON' | 'CSV' | 'Text';
  success: boolean;
  exitCode?: number;
  payload?: string | object;
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Загрузить структуру конфигурации из docker-compose.yml как текстовый Эфир.
2. Проверить версию ритуала — должна быть 3.8. Если не совпадает, вернуть хаос с пометкой несовместимости.
3. Найти блок Services и убедиться, что в нем присутствуют три Голема: app, prometheus, grafana.
4. Для каждого Голема проверить основные ритуальные параметры:
   - app: build.context должен быть точкой текущего свитка (.). image должен соответствовать codex/app-29-2-telemetry; ports содержит 8000:8000; networks включает monitoring-net.
   - prometheus: image — prom/prometheus:latest; container_name — не обязателен, но может быть задан; volumes содержит ./prometheus.yml:/etc/prometheus/prometheus.yml; ports содержит 9090:9090; depends_on — app; networks — monitoring-net.
   - grafana: image — grafana/grafana:latest; container_name — grafana_seer; ports — 3000:3000; depends_on — prometheus; networks — monitoring-net.
5. Проверить сеть: networks.monitoring-net должен существовать и быть типа bridge.
6. Проверить взаимозависимости: app — независим в старте, prometheus — после app, grafana — после prometheus.
7. Вернуть результат в виде OutputResult со статусом успеха и, при необходимости, журналами действий.

### 4.2. Declarative Content (Для конфигураций и данных)

- Версия ритуала: 3.8
- Схема пространств:
  - Сервисы: app, prometheus, grafana
  - Сети: monitoring-net
- Детали Големов:
  - app
    - build.context: .
    - image: codex/app-29-2-telemetry
    - ports: 8000:8000
    - networks: monitoring-net
    - environment: PORT=8000
  - prometheus
    - image: prom/prometheus:latest
    - container_name: prometheus_guardian
    - volumes: ./prometheus.yml:/etc/prometheus/prometheus.yml
    - ports: 9090:9090
    - depends_on: app
    - networks: monitoring-net
  - grafana
    - image: grafana/grafana:latest
    - container_name: grafana_seer
    - ports: 3000:3000
    - depends_on: prometheus
    - networks: monitoring-net
- Сетевые принципы: monitoring-net — мостовая сеть

## 5. Structural Decomposition (Декомпозиция структуры)

### 5.1. For code (если бы это был код)

- Основные блоки: version, services, networks
- Внутри services:
  - app: build, image, ports, networks, environment
  - prometheus: image, container_name, volumes, ports, depends_on, networks
  - grafana: image, container_name, ports, depends_on, networks
- В networks: monitoring-net

### 5.2. For config (это YAML конфигурация)

- version: строка версии ритуала
- services: список Големов с их свойствами
- networks: магические каналы связи

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Версия конфигурации: 3.8
- Используемые образы: prom/prometheus:latest, grafana/grafana:latest
- Порты внешние/внутренние: 8000, 9090, 3000
- Сети: monitoring-net, тип сети — bridge
- Зависимости: Prometheus зависит от app; Grafana зависит от Prometheus
- Среда выполнения: Docker Engine, Docker Compose

### 6.2. Prohibited Actions (Negative Constraints)

- НЕ хранить секреты в открытом виде в env без использования безопасных механизмов (env-файлы, секреты).
- НЕ выводить сырой конфигурационный эфир в консоль в продакшн-режиме.
- НЕ использовать синхронные сетевые вызовы в цикле запуска служб.
- НЕ встраивать конфигурационные файлы YAML/JSON внутрь скриптов, если это не оправдано архитектурой.
- НЕ менять версии образов или пути конфигураций во время реконструкции без явного обоснования.

## 7. Verification & Testing (Верификация)

### Герхин сценарии

```gherkin
Feature: Docker Compose orchestration
  Scenario: Successful startup and exposure of services
    Given a clean Docker environment with docker-compose.yml present
    When the command docker-compose up -d is executed
    Then app port 8000 is accessible and responds to requests
    And Prometheus port 9090 serves metrics
    And Grafana port 3000 is reachable for dashboards
```

```gherkin
Feature: Startup with missing dependencies
  Scenario: Missing required image or network
    Given docker-compose.yml references a unavailable image or a missing network
    When docker-compose up -d is executed
    Then the startup should fail gracefully with a clear error indicating the missing item
```
