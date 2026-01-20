# quest_9_2.py Specification

1. Meta Information

- Domain: Scripting
- Complexity: Medium
- Language: Python
- Frameworks: PyTorch
- Context: https://pytorch.org/docs/stable/nn.html

2. Goal & Purpose (Цель и Назначение)
   Context for Creator: Этот файл объявляет самодостаточный артефакт под названием MyLinear, который повторяет логику простого линейного слоя и демонстрирует работу обучаемых рун веса и смещения внутри PyTorch.
   Instruction for AI: Этот раздел поясняет WHY кода: построение небольшого модуля на основе nn.Module, чтобы понять принципы хранения параметров, вызова forward и базовой аугментации архитектуры без внешних зависимостей.

Легенда: Этот модуль превращает понятие линейного оператора в самостоятельный артефакт, который можно исследовать, разворачивая магию весов и смещения в самостоятельной руне.

3. Interface Contract (Интерфейсный Контракт)

3.1. Inputs (Входы)

- Source: API Request
- Format: JSON
- Schema:
  interface InputData {
  inputSize: number;
  outputSize: number;
  inputVector?: number[]; // опционально передать входной вектор для теста
  }

  3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: JSON
- Success Criteria: Exit Code 0
- Schema:
  interface OutputResult {
  output: number[]; // результат forward как односвязный вектор
  weight: number[][]; // жестко заданная форма весов (outputSize x inputSize)
  bias: number[]; // вектор смещений (outputSize)
  }

4. Implementation Details (The Source DNA / Исходный Код)

4.1. Algorithmic Logic (Для исполняемого кода)
Шаг 1. Признание мощи: подключаем магическую книгу PyTorch, импортируем главный завет nn.
Шаг 2. Черчение артефакта: создаем класс MyLinear, наследующийся от nn.Module — это наша Рунная Каменная структура.
Шаг 3. Инициализация чар: в конструкторе принимаем inputSize и outputSize, создаем две обучаемые руны weight и bias через nn.Parameter, заполняя их случайными значениями из нормального распределения.
Шаг 4. Прямой проход: реализуем forward, который вычисляет y = x @ weight^T + bias.
Шаг 5. Сотворение и проверка: создаем экземпляр my_first_neuron с inputSize=1 и outputSize=1, выводим паспорт артефакта и параметры, затем выводим значения каждой руны.
Шаг 6. Испытание пророчества: формируем входной тензор [2.0], применяем артефакт как заклинание и получаем результат вывода.
Шаг 7. Завершение ритуала: сообщаем, что артефакт выкован и готов к дальнейшему опробованию.

4.2. Declarative Content (Для конфигураций и данных)
Указ Ткачу: весы и смещение инициализируются случайно из нормального распределения и зависят от форм входа и выхода: weight имеет форму (outputSize, inputSize), bias имеет форму (outputSize). Входной образец для теста — одномерный вектор [2.0]. Вычисление выполняется через линейное сочетание: y = x @ w^T + b. Исходный код печатает паспорт камня, параметры и результат применения заклинания, что позволяет 1-в-1 воспроизвести ритуал.

5. Structural Decomposition (Декомпозиция структуры)

- Класс MyLinear
  - **init**(input_size, output_size): создание weight и bias как обучаемых параметров
  - forward(x): вычисление y = x @ weight^T + bias
- Основной демонстрационный блок:
  - Инстанцирование MyLinear(input_size=1, output_size=1)
  - Печать паспорта артефакта
  - Перечисление параметров по именам
  - Создание входного тензора [2.0], применение forward и печать результата
  - Сообщение о завершении ритуала

6. System Context & Constraints (Системный контекст и Ограничения)

6.1. Technical Constraints

- Performance: Standard CPU
- Concurrency: Синхронное выполнение (один поток)
- Dependencies: PyTorch (torch, torch.nn)

  6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env)
- DO NOT print raw data to console in production mode
- DO NOT use synchronous network calls in the main loop
- DO NOT wrap configuration files (.yaml, .json) into scripts (like Python/Bash)
- DO NOT change versions or paths during reconstruction

7. Verification & Testing (Верификация)

Геркин-сценарии

Feature: Functionality of the MyLinear Artefact
Scenario: Successful forward pass
Given a MyLinear module configured with inputSize 1 and outputSize 1
When an input vector [2.0] is passed through the module
Then the output should be a single value computed as 2.0 times weight plus bias and produced on STDOUT

Scenario: Dimension mismatch error
Given a MyLinear module configured with inputSize 3
When an input vector of size 1 is provided to forward
Then an error indicating dimension mismatch is raised

ИССЛЕДУЕМЫЙ АРТЕФАКТ: quest_9_2.py

ИСТОРИЯ АРТЕФАКТА: Этот текст описывает и фиксирует логику, структуру и взаимодействия куска кода, который выстраивает минимальную линейную рунику внутри PyTorch и демонстрирует принципы работы параметров и forward без внешних зависимостей. Артефакт готов к внедрению в скрижаль знаний Великой Книги Кодекса.
