# deploy-pages.yml Specification

## 1. Meta Information

- Domain: Infrastructure | Scripting
- Complexity: Low
- Language: Bash
- Frameworks: GitHub Actions; actions/checkout@v4; actions/configure-pages@v4; actions/upload-pages-artifact@v3; actions/deploy-pages@v4
- Context: Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Context for Creator: Этот ритуал автоматически собирает и выкладывает статический сайт на GitHub Pages. Он активируется при событии push в ветку main и срабатывает только если изменения затронули священные артефакты в каталоге tools/site_builder/static.
Instruction for AI: Этот раздел объясняет «ПОЧЕМУ» кода — зачем нужен этот ритуал и какие бизнес-цели он поддерживает.

Легенда: Ритуал Deploy Static Site to GitHub Pages принимает эфир из событий репозитория, отбирает путь статических материалов и аккуратно запечатывает их в хранение на публичном алтаре GitHub Pages, чтобы сайтмир был обновлён и обретал новый облик.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: API Request
- Format: JSON
- Schema: InputData
  - InputData включает следующие поля: eventName (тип строка, например push), repository (строка с именем репозитория), ref (строка ссылки на ветку), pathChanges (массив строк с изменёнными путями, опционально), sitePath (опционально путь к каталогу сайта).

### 3.2. Outputs (Выходы)

- Destination: Event Log | API Response
- Format: JSON
- Success Criteria: Exit успешно 0 | Deployment завершён
- Schema: OutputResult
  - OutputResult содержит поля: status (строка: "success" или "failure"), pagesUrl (строка с адресом страницы, опционально), artifactPath (строка с путём артефакта, опционально), runId (строка идентификатора выполнения, опционально)

## 4. Implementation Details (The Source DNA / Исходный ДНК)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Ритуал прослушивает событие push и распознаёт клятву ветки main. Он внимательно проверяет, что изменения коснулись артефактов в пути tools/site_builder/static/\*\* и что ветка соответствует канону. Если условия не соблюдены, сущность-ритуал уходит в Хаос и не выполняет дальнейших действий.
2. Когда условия выполнены, ритуал призывает призрак реpositория и кладёт точный точечный образ кристалла кода в рабочее пространство (Checkout repository).
3. Затем призывает Скрижаль “Setup Pages” для подготовки окружения, необходимых прав и корректной средой для публикации на GitHub Pages.
4. Далее Ритуал собирает или берёт из готовой ленты статические материалы из указанного каталога (tools/site_builder/static) и упаковывает их как Артефакт Эфира (артефакт для телепортации).
5. Наконец вызывается Заклинание Deploy Pages, которое переносит запечатанный Артефакт Эфира на публичный алтарь GitHub Pages, делая сайт доступным по адресу вашего репозитория.
6. В случае ошибок во время исполнения ритуал записывает Хаос и возвращает соответствующий статус, а при успехе — фиксирует путь к страницам и идентификатор выполнения.

### 4.2. Declarative Content (Для конфигураций и данных)

- Скрижаль триггера: ритуал активируется на событие push в ветку main и с условием, что изменения затронули статический контент в пути tools/site_builder/static/\*\*.
- Разрешения (Permissions): contents — read; pages — write; id-token — write.
- Алтарь миссии (Jobs): deploy
  - Runs-on: ubuntu-latest
  - Шаги:
    - Checkout repository: призывает заклинание actions/checkout@v4 для точной копии кода.
    - Setup Pages: призывает заклинание actions/configure-pages@v4 для подготовки окружения публикации.
    - Upload artifact: призывает заклинание actions/upload-pages-artifact@v3, упаковывая содержимое пути tools/site_builder/static в артефакт.
      - path: "tools/site_builder/static"
    - Deploy to GitHub Pages: призывает заклинание actions/deploy-pages@v4, которое публикует артефакт на GitHub Pages.

## 5. Structural Decomposition (Декомпозиция структуры)

- Триггер: блок On — push к main с фильтром путей static
- Разрешения: Contents, Pages, Id-token
- Задание (Job): deploy
  - Среда выполнения: ubuntu-latest
  - Шаги:
    - Checkout repository
    - Setup Pages
    - Upload artifact
    - Deploy to GitHub Pages

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Синхронная последовательность шагов (Step-by-step execution)
- Dependencies: actions/checkout@v4; actions/configure-pages@v4; actions/upload-pages-artifact@v3; actions/deploy-pages@v4

### 6.2. Prohibited Actions (Negative Constraints)

- НЕ хранить секреты в открытом виде (используйте секреты GitHub).
- НЕ выводить в консоль сырые данные в продакшн-режиме.
- НЕ выполнять сетевые вызовы синхронно в главном цикле ритуала, если это может блокировать поток.
- НЕ заворачивать конфигурационные файлы (.yaml, .json) в скрипты.
- НЕ менять версии или пути во время реконструкции артефакта.

## 7. Verification & Testing (Верификация)

1. Гхи́ркин-сценарий: Успешная публикация
   Функционал: Deploy Static Site to GitHub Pages выполняется после пуша в main с изменениями в tools/site_builder/static.
   Контекст: Дух выполняет все шаги: Checkout, Setup Pages, Upload artifact, Deploy Pages. Ожидаемый результат: артефакт упакован и размещён на GitHub Pages; сайт доступен; внутри лога отображается статус успеха.

2. Гхи́ркин-сценарий: Нет триггера из-за пути
   Функционал: Пуш в main, но изменения не касаются пути static (например, tools/site_builder/assets/). Ритуал не должен запускаться или не достигать шага деплоймента. Ожидаемый результат: нет деплоймента; изменения остаются в истории репозитория, логи не содержат публикацию на GitHub Pages.

Индивидуальные заметки: Этот артефакт называется deploy-pages.yml и действует как канонический Ритуал на скитании Эфира между репозиторием и публичным алтарём GitHub Pages. Все связи остаются в рамках независимого артефакта — Independent Artifact.
