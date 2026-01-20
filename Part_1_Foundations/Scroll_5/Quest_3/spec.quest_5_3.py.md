# quest_5_3.py Specification

## 1. Meta Information

- Domain: ML/NLP
- Complexity: Medium
- Language: Python
- Frameworks: PyTorch, transformers, peft, bitsandbytes
- Context: ../PEFT_GUIDE.md

---

## 2. Goal & Purpose (Цель и Назначение)

- Context for Creator: Этот артефакт — Ритуал для демонстрации модульной природы PEFT и развёртывания дообученного адаптера на базовой модели. Его задача — проверить, как инференс обученного PEFT-адаптера может давать новые знания и оценивать выход Голема по заданному промпту.
- Instruction for AI: Этот файл закладывает понятный WHY: показать последовательность действий от призыва чистого Голема до экзаменационного ответа после надевания адаптера, сохраняя модульность и повторное использование артефактов.

Описание на русском языке: В рамках квеста 5.3 мы призываем базовую модель, накладываем на неё обученный адаптер, формируем экзаменационный промпт и просчитываем ответ Голема, затем оцениваем качество и демонстрируем принцип работы PEFT.

---

## 3. Interface Contract (Интерфейсный Контракт)

### 3.1. Inputs (Входы)

- Source: CLI Args
- Format: JSON
- Schema:

```typescript
interface InputData {
  modelName: string; // имя базовой модели, например "distilgpt2"
  adapterPath: string; // путь к артефакту адаптера, например "../Quest_2/results/checkpoint-250"
  prompt?: string; // необязательный промпт для экзамена
  maxNewTokens?: number; // ограничение длины ответа
  device?: string; // устройство выполнения, по умолчанию "auto"
  quantization?: {
    loadIn4Bit: boolean; // использовать 4-битное квантование
    bnb4BitComputeDtype?: string; // тип вычислений, например "torch.float16"
  };
}
```

### 3.2. Outputs (Выходы)

- Destination: STDOUT
- Format: Text
- Success Criteria: Exit Code 0
- Schema:

```typescript
interface OutputResult {
  status: "success" | "failure";
  exitCode: number;
  message?: string;
  result?: string;
}
```

---

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

1. Импортируйте магические артефакты: библиотеку тьмы torch, чертеж PeftModel из peft, а также AutoModelForCausalLM, AutoTokenizer и BitsAndBytesConfig из transformers. Это и есть рабский набор для Ритуала.
2. Создайте Скрижаль квантования: задействуйте load_in_4bit = true и bnb_4bit_compute_dtype = torch.float16, чтобы Голем мог путешествовать по Эфиру с минимальной тяжестью памяти и сохранением точности.
3. Призовите базовый Голем: вызов AutoModelForCausalLM.from_pretrained с именем модели (modelName) и конфигурацией квантования, device_map="auto" — чтобы Дух Ускорителя распределял части по ресурсам.
4. Призовите Переводчика (tokenizer): AutoTokenizer.from_pretrained(modelName) и установите pad_token как eos_token для полной совместимости между мирами.
5. Обнаружьте магический артефакт (adapter): путь adapterPath — и сообщите вселенной, что нашли его.
6. Наложите адаптер: через PeftModel.from_pretrained на базовую модель, чтобы наделить Голема обученными страницами из адаптерного артефакта.
7. Экзаменационный промпт: сформируйте промпт вида "Instruction: Which genre is the hobbit?\nResponse:" и преобразуйте его в тензоры через токенизатор, отправив на CUDA.
8. Сгенерируйте ответ: вызов model.generate с input_ids и attention_mask, ограничьте до max_new_tokens (например, 20) для краткости.
9. Декодируйте и выведите: раскодируйте outputs[0] через tokenizer.decode, пропуская служебные токены, и выведите полученный ответ как финальный результат экзамена Голема.

### 4.2. Declarative Content (Для конфигураций и данных)

- База модели: distilgpt2
- Адаптер: путь ../Quest_2/results/checkpoint-250
- Квантизация: load_in_4bit = true; bnb_4bit_compute_dtype = torch.float16
- Токенизатор: AutoTokenizer.from_pretrained(modelName); pad_token установлен в eos_token
- Промпт экзамена: "Instruction: Which genre is the hobbit?\nResponse:"
- max_new_tokens: 20
- Целевая платформа исполнения: CUDA-устройство, распределение частей по ресурсам через device_map="auto"
- Сообщения: начало ритуала призыва и завершение, с печатью статусов на кристалле

---

## 5. Structural Decomposition (Декомпозиция структуры)

- Основной артефакт: Quest 5.3 скрипт (quest_5_3.py)
- Ключевые компоненты:
  - model: AutoModelForCausalLM
  - tokenizer: AutoTokenizer
  - adapter: PeftModel
- Логические блоки:
  - подготовка чар (импорт и конфигурации)
  - призыв базового Голема
  - наложение адаптера
  - формирование экзаменационного промпта
  - инференс и создание ответа
  - вывод и завершение ритуала

---

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

- Performance: Оптимизирован для GPU с поддержкой 4-битного квантования; стандартный баланс вычислений держится через bnb_4bit_compute_dtype = float16.
- Concurrency: Синхронный режим выполнения (step-by-step ритуал выполняется по очереди).
- Dependencies: torch, transformers, peft, bitsandbytes

### 6.2. Prohibited Actions (Negative Constraints)

- DO NOT store secrets in plain text (use .env).
- DO NOT print raw data to console in production mode.
- DO NOT use synchronous network calls in the main loop.
- DO NOT wrap configuration files (.yaml, .json) into scripts (like Python/Bash).
- DO NOT change versions or paths during reconstruction.

---

## 7. Verification & Testing (Верификация)

1. Гхеркин-сценарии

Feature: Quest 5.3 Script Functionality
Scenario: Successful execution
Given база моделей distilgpt2 доступна и адаптер по пути ../Quest_2/results/checkpoint-250 найден
When ритуал запускается с корректной средой и параметрами
Then получаем краткий осмысленный ответ от Голема и процесс завершается кодом 0

Scenario: Adapter path missing
Given указан путь адаптера, который не существует
When выполняется попытка наложить адаптер
Then процесс завершается с ненулевым кодом выхода и в логе записана хаотичная ошибка

Ниже — ясные признаки успеха: сообщение выполнил ритуал без хаоса и код выхода 0. В случае ошибки суток Хаос регистрирует явные тревожные сигналы и корректно прерывает исполнение.

---

ИСТОЧНИК АРТЕФАКТА: quest_5_3.py
