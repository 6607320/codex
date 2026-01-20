# prometheus.yml Specification

1. Meta Information

- Domain: Infrastructure
- Complexity: Low
- Language: YAML
- Frameworks: Docker-Compose, Prometheus, Kubernetes
- Context: Independent Artifact

2. Goal & Purpose (Цель и Назначение)
   Context for Creator: Этот Ритуал Скрижали задаёт правила мониторинга для нашего стража‑«Летописца», чтобы он регулярно собирал Эфир метрик из FastAPI‑приложения и направлял их в Хранилище наблюдений. Instruction for AI: Этот раздел объясняет WHY — зачем нужна эта Скрижаль и что она обеспечивает.

Легенда: Скрижаль prometheus.yml устанавливает глобальный интервал сбора метрик и указывает, что страж должен наблюдать за FastAPI‑системой через конкретный пункт входа (/metrics) на порту 8000. Это позволяет поддерживать живое зрение над состоянием сервиса и быстро замечать Хаос в работе приложения.

3. Interface Contract (Интерфейсный Контракт)

3.1 Inputs (Входы)

- Source: CLI Args
- Format: YAML
- Schema: Входной Эфир описывает источник конфигурации Prometheus. В формате TypeScript‑подобной формы:
  - InputData имеет поля:
    - source: "CLI Args"
    - format: "YAML"
    - path: строка, путь к файлу prometheus.yml

    3.2 Outputs (Выходы)

- Destination: STDOUT
- Format: JSON
- Success Criteria: Exit code 0
- Schema: OutputResult имеет поля:
  - status: "success" или "failure"
  - path: строка, путь к проверяемому файлу
  - message?: строка, детальное сообщение об успехе или причине сбоя

4. Implementation Details (The Source DNA / Исходный Код)

4.1 Algorithmic Logic (Для исполняемого кода)
Шаг за шагом ritual:

1. Взять путь к файлу prometheus.yml либо из параметров CLI, либо из окружения.
2. Прочитать содержимое файла и преобразовать его в внутризначную Эфирную структуру YAML.
3. Проверить наличие глобального раздела – global — и внутри него убедиться, что scrape_interval задан как 15s.
4. Найти раздел scrape_configs как главный свиток: внутри него должен быть один элемент с job_name равным fastapi-app.
5. Внутри этого элемента проверить наличие секции static_configs и массива targets, и чтобы среди целей присутствовал элемент "app:8000".
6. При отсутствии любого из критических элементов вернуть Хаос с понятной детализацией ошибки; иначе продолжить.
7. Зафиксировать результат проверки как успешный и вернуть соответствующий статус и путь к файлу.
8. Не модифицировать содержимый Скрижаль: данный процесс только читает и валидирует структуру.
9. В случае успеха не создаются новые артефакты, файл может быть помечен как валидированный; в случае ошибки — вернуть ясное сообщение об источнике несоответствия.

4.2 Declarative Content (Для конфигураций и данных)
Имущество Скрижали prometheus.yml как часть портала наблюдений:

- Глобальная оправа: глобальный блок, который устанавливает режим слежения и интервал — 15 секунд — для непрерывного сбора Эфира.
- Свиток Слежения: раздел scrape_configs, содержащий одну запись с именем fastapi-app, что служит указателем для летописьей о том, чьих подопечных слушать.
- Талисман цели: static_configs, внутри которого прописаны точные координаты цели — targets, включая сущность app:8000.
- Лексема рисков: возможный Хаос, если структура нарушена или цель не найдена; в таких случаях ритуал прерывается и возвращает пояснение.

5. Structural Decomposition (Декомпозиция структуры)

- global: глобальные параметры мониторинга, включая scrape_interval
- scrape_configs: массив конфигураций наблюдения
  - job_name: идентификатор задачи наблюдения (fastapi-app)
  - static_configs: конкретные точки наблюдения
    - targets: список адресов в формате хоста:порта (например, app:8000)
- Связанные элементы: нет дополнительных узлов при текущем наборе

6. System Context & Constraints (Системный контекст и Ограничения)

6.1 Technical Constraints

- Performance: Ритуал оптимизирован под быстрый читательский доступ; базовый интервал сбора — 15s, характерный для стабильного мониторинга.
- Concurrency: Async в рамках самой системы мониторинга; конфигурация задаёт расписание, а исполнение может происходить асинхронно в рамках движка сбора метрик.
- Dependencies: Prometheus как движок сбора, возможность интеграции через Docker-Compose или Kubernetes; целевая система должна экспонировать метрики по адресу /metrics на порту 8000.

  6.2 Prohibited Actions (Negative Constraints)

- НЕ размещать секреты в явном виде в конфигурации; чувствительные данные держать отдельно.
- НЕ печатать в консоль необработанные данные в продакшене.
- НЕ использовать синхронные сетевые вызовы в критических путях сбора метрик.
- НЕ оборачивать конфигурационные файлы YAML или JSON в скрипты.
- НЕ менять версии ПО, пути или целевые порты без явного разрешения реконструкции.

7. Verification & Testing (Верификация)

1. Gherkin — счастливый сценарий
   Feature: Prometheus YAML validation
   Scenario: Successful execution
   Given a prometheus.yml with a global scrap_interval of 15s and a single scrape_configs entry
   When the system reads and validates the configuration
   Then the configuration is accepted and ready for Prometheus to scrape

1. Gherkin — ошибка конфигурации
   Feature: Prometheus YAML validation
   Scenario: Failure when targets are missing or malformed
   Given a prometheus.yml that lacks a valid targets entry or contains an incorrect endpoint
   When the system validates the configuration
   Then a descriptive error is produced indicating the missing or invalid targets and the reason

ИССЛЕДУЕМЫЙ АРТЕФАКТ: prometheus.yml
ИСХОДНЫЙ КОД (для контекста, без копирования кода): глобальная глава с полем global.scrape_interval: 15s; в свитке scrape_configs задано задание fastapi-app; внутри static_configs указан один элемент targets: ["app:8000"]. Это образец структуры, который мы валидируем и как указанные поля сохраняются в целом виде без модификации.
