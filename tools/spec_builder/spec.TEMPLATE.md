# <Quest Name> Specification

## 1. Meta Information

- **Domain:** <Scripting | ML/NLP | Infrastructure | Web3 | Agent>
- **Complexity:** <Low | Medium | High>
- **Language:** <Python | TypeScript | Bash | Solidity | Go>
- **Frameworks:** <List critical libs: e.g., Pandas, Ethers.js, Docker>
- **Context:** <Ссылка на глобальный контекст, если есть, например: ../AGENTS.md>

## 2. Goal & Purpose (Цель и Назначение)

**Context for Creator:** Опишите бизнес-задачу кратко. Что делает этот модуль?
**Instruction for AI:** This section provides high-level intent. Use it to understand the "WHY" of the code.

<Описание на русском языке.>

## 3. Interface Contract (Интерфейсный Контракт)

**Instruction for AI:** Strictly implement these data structures. Do not hallucinate fields. Use TypeScript interfaces to define shapes.

### 3.1. Inputs (Входы)

- **Source:** <CLI Args | STDIN | API Request | Kafka Topic | Smart Contract Call>
- **Format:** <JSON | Text | Binary | Stream>
- **Schema:**
  ```typescript
  // Define input interfaces here
  interface InputData {
    // ...
  }
  ```

### 3.2. Outputs (Выходы)

- **Destination:** <STDOUT | File | Database | API Response | Event Log>
- **Format:** <JSON | CSV | Text>
- **Success Criteria:** <Exit Code 0 | 200 OK | File Created>
- **Schema:**
  ```typescript
  // Define output interfaces here
  interface OutputResult {
    // ...
  }
  ```

## 4. Implementation Details (The Source DNA / Исходный Код)

### 4.1. Algorithmic Logic (Для исполняемого кода)

[Сюда ИИ вставит Указ Ткачу и пошаговый алгоритм]

### 4.2. Declarative Content (Для конфигураций и данных)

[Сюда ИИ вставит Указ Ткачу и точные данные для воссоздания 1-в-1]

## 5. Structural Decomposition (Декомпозиция структуры)

**Instruction for AI:**

- Для кода: перечисли функции и классы.
- Для конфигов: перечисли основные логические блоки (например: `repos`, `services`, `build_stages`).

## 6. System Context & Constraints (Системный контекст и Ограничения)

### 6.1. Technical Constraints

**Instruction for AI:** Extract constraints based on the code (e.g. specific libraries, memory usage patterns).

- **Performance:** <e.g., "Optimized for 4GB VRAM" or "Standard CPU">
- **Concurrency:** <Sync or Async?>
- **Dependencies:** <List key external dependencies>

### 6.2. Prohibited Actions (Negative Constraints)

**Instruction for AI:** Infer what should NOT be done based on the code style (e.g., "Do not hardcode secrets").

- **DO NOT** store secrets in plain text (use .env).
- **DO NOT** print raw data to console in production mode.
- **DO NOT** use synchronous network calls in the main loop.
- **DO NOT** wrap configuration files (.yaml, .json) into scripts (like Python/Bash).
- **DO NOT** change versions or paths during reconstruction.

## 7. Verification & Testing (Верификация)

**Instruction for AI:** Generate 1-2 Gherkin scenarios that describe the happy path and one error case for this script.

```gherkin
Feature: [Script Functionality]
  Scenario: Successful execution
    Given [Preconditions]
    When [Action is taken]
    Then [Expected result]
```
