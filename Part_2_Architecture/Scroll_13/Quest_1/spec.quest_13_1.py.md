# quest_13_1.py Specification

## 1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language / Стихия: Python
- Frameworks: PyTorch, torch.nn
- Context: ../AGENTS.md

## 2. Goal & Purpose (Цель и Назначение)

Context for Creator: Этот файл реализует и демонстрирует экономный строительный блок для мобильных нейронных сетей — Depthwise Separable Convolution — и наглядно сравнивает его параметрическую экономию с обычной сверточной операцией. Instruction for AI: Высокий уровень замысла — понять, почему раздельная свертка снижает число параметров и сохраняет форму выходных карт, подтверждая корректность вывода двумя методами.

Описание на русском языке:
Этот модуль создает два артефактных слоя свертки: экономный Depthwise Separable Conv и привычный стандартный Conv. Он затем запускает простой эксперимент: строит тестовый тензор, подсчитывает общее число параметров у каждого подхода, вычисляет коэффициент эффективности и проверяет, что выходные формы совпадают. В итоге мы получаем наглядное подтверждение того, что применение раздельной свертки даёт значительную экономию параметров без потери совместимости форм выхода.

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: Text
- Schema: interface InputData {
  // Этот артефакт не требует внешних входов во время выполнения
  }

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: Text
- Success Criteria: Exit Code 0
- Schema: interface OutputResult {
  // Формы вывода не закреплены жёстко в контракте; вывод через консоль
  }

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

- Ритуал подготовки инструментов: Импортируется магия вычислений из PyTorch — токи и узлы чар NN.
- Черепаха-Чертеж DepthwiseSeparableConv:
  - Инициализация заклинания принимает in_channels, out_channels, kernel_size, padding.
  - Этап 1 Глубинная Свертка (Depthwise): создаётся слой nn.Conv2d с группировкой по каждому входному каналу (groups = in_channels), сохраняющий число выходных карт равным входным.
  - Этап 2 Точечная Свертка (Pointwise): создаётся слой nn.Conv2d с kernel_size = 1, который объединяет карты признаков в требуемое число выходных карт.
- Проход заклинания (Forward): данные сперва проходят через depthwise, затем через pointwise — и возвращается финальная карта.
- Акт сравнения two рудокопов:
  - Задаются параметры тестового образца: in_channels = 3, out_channels = 16, image_size = 32.
  - input_tensor = случайный тензор размерности (1, 3, 32, 32).
  - Создаётся стандартный Conv: Conv2d(3, 16, kernel_size=3, padding=1).
  - Подсчитываются стандартные руны (standard_params) как сумма.numel() по всем параметрам standard_conv.
  - Создаётся наш Экономный Рудокоп: separable_conv = DepthwiseSeparableConv(3, 16, kernel_size=3, padding=1).
  - Подсчитываются руны экономного артефакта (separable_params) аналогично.
- Акт 4 Вердикт:
  - Выводится заголовок сравнения и значения параметров для обоих подходов.
  - Вычисляется эффективность: standard_params / separable_params.
  - Прогоняются оба рудокопа на одном и том же input_tensor.
  - Выводятся формы выходов и заявляется, что формы совпадают и артефакт работает корректно.

### 4.2. Declarative Content (Для конфигураций и данных)

Inventory является частью описания данных, которые задействованы в эксперименте. Здесь зафиксированы следующие данные и параметры:

- 3 входных канала (in_channels) и 16 выходных карт (out_channels) для порождения тестовой карты признаков.
- Ядро свертки 3x3 и паддинг 1, чтобы сохранить пространственный размер.
- Формат входного образа: 1 образ с размерностью 3 x 32 x 32 и рандомизированный эфир (input_tensor = torch.randn(1, 3, 32, 32)).
- Стандартный сверточный слой: Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1).
- Экономный сверточный блок: DepthwiseSeparableConv(in_channels=3, out_channels=16, kernel_size=3, padding=1).
- Метрики: общее число параметров у каждого подхода, коэффициент экономии, формы выходов.

## 5. Structural Decomposition (Декомпозиция структуры)

- Класс DepthwiseSeparableConv
  - **init**(self, in_channels, out_channels, kernel_size, padding)
  - forward(self, x)
- Объекты и модули в тестовом сценарии:
  - standard_conv: Conv2d
  - separable_conv: DepthwiseSeparableConv
  - input_tensor: Tensor
- Логика подсчёта параметров: суммирование p.numel() по всем параметрам каждого слоя
- Вердикт и верификация форм выходов: печать Shapes и сравнение

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Sync
- Dependencies: PyTorch (torch, torch.nn)

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (используйте .env для секретов)
- DO NOT print raw data to console в продуктивной среде
- DO NOT использовать синхронные сетевые вызовы в главном цикле
- DO NOT оборачивать конфигурационные файлы (.yaml, .json) в скрипты
- DO NOT менять версии библиотек или путей во время реконструкции

## 7. Verification & Testing (Верификация)

Герник-сценарии 1-2 и один сценарий ошибки:

Feature: Script Functionality
Scenario: Successful execution
Given PyTorch environment with required modules
When скрипт выполняется и строит оба блока свертки (Standard Conv и Depthwise Separable Conv)
Then выводятся числа параметров и формы выходов, и формы совпадают

Scenario: Shape mismatch leads to error (invalid configuration)
Given модификация входного размера или параметров, приводящая к несовпадению форм
When скрипт выполняется
Then выводится сообщение о несовпадении форм и тест считается неуспешным

Инцидентом артефакт подтверждает: эксперимент завершился успешно, формы совпадают, и экономный блок действительно экономит параметры относительно стандартной свёртки.
