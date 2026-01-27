# Dockerfile Specification

## 1. Meta Information

- **Domain:** Infrastructure
- **Complexity:** Medium
- **Language:** Bash
- **Frameworks:** PyTorch, CUDA, cuDNN, Docker
- **Context:** Independent Artifact

## 2. Goal & Purpose (Цель и Назначение)

Легенда: этот пергамент описывает создание боевой мастерской для вычислительных чар. Он превращает официальный базовый образ PyTorch с поддержкой CUDA в готовый к работе артефакт, который устанавливает нужные библиотеки, помещает источник амулета main.py в рабочее место, открывает врата порта 8000 и запускает сервис uvicorn для обслуживания запросов.

---

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- **Source:** CLI Args
- **Format:** Text
- **Schema:**
  interface InputData {
  baseImage: string;
  workDir: string;
  copies: Array<{ source: string; destination: string }>;
  installCommand: string;
  exposedPorts: number[];
  entrypoint: string[];
  }

---

### 3.2. Outputs (Выходы)

- **Destination:** File
- **Format:** JSON
- **Success Criteria:** Exit Code 0
- **Schema:**
  interface OutputResult {
  imageName: string;
  imageTag: string;
  sizeMB?: number;
  digest?: string;
  layers?: number;
  }

---

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Выбираем основную заготовку из официальной гильдии PyTorch: устанавливаем базовый образ с CUDA и cuDNN на Ubuntu.
2. Создаем рабочую мастерскую внутри Голема по имени /app и переходим в нее.
3. Копируем свиток libraries.list в мастерскую, чтобы духи зависимостей стали доступны.
4. Взываем духа пакетирования и разрешаем ему установить все библиотеки из libraries.list без сохранения лишнего мусора.
5. Копируем чертёж Амулета main.py в рабочее пространство.
6. Призываем врата сети: открываем порт 8000 для внешнего общения.
7. Устанавливаем Главного Заклинателя: запускаем uvicorn с указанием хоста 0.0.0.0 и порта 8000, чтобы мир мог обратиться к службе.

### 4.2. Declarative Content (Для конфигураций и данных)

Inventory из мира артефактов:

- Базовый образ: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
- Рабочее место: /app
- Копируемые амулеты: libraries.list в корень; main.py в корень
- Заклятие установки зависимостей: pip install --no-cache-dir -r libraries.list
- Окна связи: порт 8000
- Главный обряд запуска: uvicorn main:app --host 0.0.0.0 --port 8000

---

## 5. Structural Decomposition (Декомпозиция структуры)

- 🏰 Базовый образ (Base Image): pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
- 🛡️ Рабочее место (Working Directory): /app
- 🗺 Карты копирования (Copy Steps): libraries.list -> .; main.py -> .
- 🪄 Заклятие установки зависимостей (Install Step): pip install --no-cache-dir -r libraries.list
- 🧭 Ворота сети (Ports): 8000
- 🕯 Запуск сервиса (Entrypoint): uvicorn main:app --host 0.0.0.0 --port 8000

---

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- **Performance:** Оптимизировано под GPU-выполнение через базовый образ PyTorch с CUDA 11.8 и cuDNN8-runtime.
- **Concurrency:** Последовательная сборка и слой за слоем кэширование образа; сборка детерминирована.
- **Dependencies:** Ядро — PyTorch 2.1.0, CUDA 11.8, cuDNN8, Python окружение внутри образа; зависимости задаются в libraries.list.

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT хранить секреты в открытом виде внутри образа или Dockerfile.
- DO NOT выводить чувствительные данные в вывод сборки или в логи продакшн-режиме.
- DO NOT вставлять синхронные сетевые вызовы в основную логику запуска сервиса внутри образа.
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты (как в Python/Bash).
- DO NOT менять версии образов или пути к файлам во время реконструкции.

---

## 7. Verification & Testing (Верификация)

```gherkin
Feature: Dockerfile Build and Run
  Scenario: Successful build and run
    Given a proper build context containing libraries.list and main.py and Dockerfile
    When docker build -t app:latest . and docker run -p 8000:8000 app:latest
    Then a container starts and uvicorn serves on http://0.0.0.0:8000

  Scenario: Build failure due to missing dependencies file
    Given the build context is missing libraries.list
    When docker build is executed
    Then the build fails with an error indicating libraries.list is missing or unreadable
```
