# **Архитектура Фокуса: Глубокий анализ и техническая реализация механизма Scaled Dot-Product Attention**

## **1\. Введение: От последовательности к параллелизму и эволюция "Заклинания Фокуса"**

История обработки естественного языка (NLP) и глубинного обучения — это, по сути, история поиска эффективных методов контекстуализации. До появления архитектуры Transformer, доминирующей парадигмой были рекуррентные нейронные сети (RNN) и их усовершенствованные вариации, такие как LSTM (Long Short-Term Memory) и GRU (Gated Recurrent Units). Эти архитектуры опирались на последовательную обработку данных: токен $t$ мог быть обработан только после завершения обработки токена $t-1$. Этот подход, хотя и интуитивно понятный с точки зрения человеческого восприятия времени, создавал фундаментальные инженерные ограничения.1

Во-первых, последовательная природа RNN препятствовала распараллеливанию вычислений, что критично для обучения на современных GPU. Во-вторых, сжатие всей истории последовательности в вектор скрытого состояния фиксированного размера приводило к потере информации на длинных дистанциях — так называемая проблема "бутылочного горлышка". Механизм внимания (Attention) изначально был введен как надстройка над RNN для решения этой проблемы, позволяя декодеру "подглядывать" за всеми состояниями энкодера. Однако настоящая революция произошла с выходом статьи "Attention Is All You Need" 2, где авторы постулировали, что рекуррентность не является необходимой. Достаточно одного лишь внимания.

В контексте поставленной задачи — "Квеста 11.1: Создание заклинания Фокуса" — мы рассматриваем **Scaled Dot-Product Attention** не просто как математическую формулу, а как алгоритмическое воплощение когнитивного процесса избирательности. Это "заклинание" позволяет нейронной сети динамически перераспределять информационные веса, выделяя сигналы из шума на основе их семантической релевантности, а не позиционной близости. Этот отчет представляет собой исчерпывающее техническое руководство, охватывающее теоретические основы, математические доказательства, нюансы реализации на PyTorch и современные методы оптимизации данного механизма.

## ---

**2\. Теоретический фундамент: Математика подобия и метафоры поиска**

В основе механизма внимания лежит концепция дифференцируемого поиска по словарю. Чтобы реализовать "фокус", система должна уметь отвечать на вопрос: "Насколько информация $A$ важна для понимания информации $B$?".

### **2.1 Векторная геометрия и выбор метрики подобия**

Почему именно скалярное произведение (dot product)? В векторных пространствах высокой размерности выбор метрики подобия определяет топологию семантических отношений.

Скалярное произведение двух векторов $\\mathbf{q}$ и $\\mathbf{k}$ размерности $d\_k$ определяется как:

$$\\mathbf{q} \\cdot \\mathbf{k} \= \\sum\_{i=1}^{d\_k} q\_i k\_i \= \\|\\mathbf{q}\\| \\|\\mathbf{k}\\| \\cos(\\theta)$$  
Если векторы нормализованы (имеют единичную длину), скалярное произведение эквивалентно косинусному подобию, которое измеряет косинус угла между векторами. Значение 1 означает полную коллинеарность (идентичность направлений), 0 — ортогональность (отсутствие корреляции), \-1 — противоположную направленность.3

В отличие от евклидова расстояния ($L\_2$), которое измеряет физическую "близость" точек в пространстве, скалярное произведение учитывает как направленность (семантический смысл), так и, в случае ненормализованных векторов, их магнитуду (значимость или уверенность). С вычислительной точки зрения, скалярное произведение реализуется через матричное умножение (GEMM — General Matrix Multiply), которое является одной из самых оптимизированных операций на современных аппаратных ускорителях (GPU/TPU).3 Это делает его предпочтительным выбором по сравнению с аддитивным вниманием (использовавшимся в ранних работах Bahdanau), которое требует прогона через полносвязный слой и функцию активации tanh.6

### **2.2 Концептуальная триада: Query, Key, Value**

Для реализации механизма внимания входные векторы проецируются в три различных пространства: Запрос (Query), Ключ (Key) и Значение (Value). Эта терминология заимствована из теории баз данных и информационного поиска.7

**Таблица 1\. Семантические роли компонентов внимания**

| Компонент            | Обозначение | Роль в системе                                                           | Аналогия с базой данных             | Аналогия с дейтинг-приложением                                                   |
| :------------------- | :---------- | :----------------------------------------------------------------------- | :---------------------------------- | :------------------------------------------------------------------------------- |
| **Query (Запрос)**   | $Q$         | Текущий токен, ищущий контекст. Вектор определяет, _что_ нужно найти.    | SQL-запрос (SELECT... WHERE...)     | Профиль пользователя с предпочтениями ("Ищу партнера, любящего походы")          |
| **Key (Ключ)**       | $K$         | Метка содержания. Вектор определяет, _чем_ является токен для других.    | Индекс или первичный ключ в таблице | Профиль другого пользователя с его характеристиками ("Люблю походы", "Рост 180") |
| **Value (Значение)** | $V$         | Информационная нагрузка. Вектор содержит _смысл_, который будет передан. | Содержимое строки базы данных       | Личность человека, с которым происходит знакомство, его контент                  |

Глубокий разбор аналогии с дейтинг-приложением:  
Представьте процесс мэтчинга. Пользователь $A$ (Query) имеет набор критериев. Система сканирует базу пользователей, сравнивая критерии $A$ с характеристиками $B, C, D...$ (Keys). Результатом сравнения является оценка совместимости (Attention Score). В классическом поиске это бинарный ответ (подходит/не подходит). В механизме внимания это "мягкий" рейтинг: пользователь $B$ подходит на 80%, $C$ на 10%, $D$ на 5%. Итоговый результат (Output) — это не один человек, а взвешенная сумма личностей (Values), где личность $B$ вносит наибольший вклад в формирование контекста для $A$.11

### **2.3 Каноническая формула**

Математическая формулировка Scaled Dot-Product Attention выглядит следующим образом 2:

$$\\text{Attention}(Q, K, V) \= \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V$$  
Эта формула описывает последовательность операций:

1. **Матричное умножение $QK^T$:** Вычисление попарного подобия между всеми запросами и ключами. Результат — матрица логитов (Raw Attention Scores) размера $L \\times L$ (где $L$ — длина последовательности).
2. **Масштабирование (Scaling) на $1/\\sqrt{d\_k}$:** Нормализация дисперсии логитов.
3. **Применение маски (Masking):** Исключение нежелательных связей (например, "подглядывания" в будущее).
4. **Softmax:** Преобразование логитов в вероятностное распределение (сумма по строкам равна 1).
5. **Агрегация ($ \\cdot V$):** Взвешенное суммирование векторов значений.

## ---

**3\. Критическая роль масштабирования: Почему $\\sqrt{d\_k}$?**

Термин "Scaled" (масштабированное) в названии механизма не является косметическим. Деление на корень из размерности ключа ($\\sqrt{d\_k}$) является фундаментальным требованием для сходимости модели при больших размерностях векторов.

### **3.1 Статистический анализ дисперсии**

Рассмотрим элементы векторов $q$ и $k$ как независимые случайные величины с нулевым средним ($E\[x\]=0$) и единичной дисперсией ($Var(x)=1$). Такое распределение типично для векторов после инициализации весов (например, инициализация Ксавье или Ге (He)) и после слоев нормализации (LayerNorm).

Скалярное произведение $q \\cdot k \= \\sum\_{i=1}^{d\_k} q\_i k\_i$.  
Математическое ожидание произведения двух независимых переменных равно произведению их ожиданий:  
$E\[q\_i k\_i\] \= E\[q\_i\]E\[k\_i\] \= 0 \\cdot 0 \= 0$.  
Дисперсия произведения независимых переменных:  
$Var(q\_i k\_i) \= E\[(q\_i k\_i)^2\] \- (E\[q\_i k\_i\])^2 \= E\[q\_i^2\]E\[k\_i^2\] \- 0 \= Var(q\_i)Var(k\_i) \= 1 \\cdot 1 \= 1$.  
Дисперсия суммы независимых случайных величин равна сумме их дисперсий. Следовательно, для скалярного произведения векторов размерности $d\_k$:

$$Var(q \\cdot k) \= Var\\left(\\sum\_{i=1}^{d\_k} q\_i k\_i\\right) \= \\sum\_{i=1}^{d\_k} Var(q\_i k\_i) \= \\sum\_{i=1}^{d\_k} 1 \= d\_k$$  
Таким образом, без масштабирования дисперсия (и, следовательно, стандартное отклонение) скалярного произведения растет линейно с размерностью вектора.14 Для модели с $d\_k \= 64$ стандартное отклонение равно 8\. Для $d\_k \= 512$ оно составляет $\\approx 22.6$. Это означает, что значения скалярных произведений могут варьироваться в диапазоне $\[-45, 45\]$ и более.

### **3.2 Проблема насыщения Softmax и исчезающие градиенты**

Функция Softmax определяется как:

$$\\sigma(\\mathbf{z})\_i \= \\frac{e^{z\_i}}{\\sum\_{j=1}^K e^{z\_j}}$$  
Функция Softmax чувствительна к масштабу входных данных (температуре).

- **Малые значения $z$:** Если логиты близки к нулю, распределение стремится к равномерному. Градиенты протекают свободно.
- **Большие значения $z$:** Если магнитуда логитов велика (из-за большой дисперсии $d\_k$), экспонента $e^{z\_i}$ для максимального элемента становится огромной по сравнению с остальными. Распределение становится "пиковым" (близким к one-hot вектору: одна 1, остальные 0).

В областях насыщения (где выход близок к 0 или 1\) локальная производная функции Softmax стремится к нулю. Это явление известно как **проблема исчезающих градиентов** (vanishing gradients). Если градиенты близки к нулю, обратное распространение ошибки (backpropagation) не может эффективно обновлять веса, и обучение останавливается.6

Деление на $\\sqrt{d\_k}$ нормализует дисперсию результата обратно к 1:

$$Var\\left(\\frac{q \\cdot k}{\\sqrt{d\_k}}\\right) \= \\frac{1}{(\\sqrt{d\_k})^2} Var(q \\cdot k) \= \\frac{1}{d\_k} \\cdot d\_k \= 1$$

Это удерживает значения логитов в "линейной" зоне функции Softmax, обеспечивая стабильный поток градиентов независимо от глубины или ширины модели.16

## ---

**4\. Реализация "Заклинания Фокуса": Инженерный разбор**

Переход от формул к коду требует учета множества нюансов: от управления памятью GPU до численной стабильности. Ниже представлен подробный разбор реализации Scaled Dot-Product Attention на PyTorch.

### **4.1 Подготовка тензоров и линейные проекции**

В стандартной реализации входной тензор $X$ имеет размерность (Batch_Size, Seq_Len, D_Model). Первым шагом является проекция $X$ в пространства $Q, K, V$ с помощью линейных слоев. В реализации Multi-Head Attention размерность модели $D\_{model}$ разбивается на $h$ голов (heads), так что $D\_{model} \= h \\times d\_k$.

Python

import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import math

class MultiHeadAttention(nn.Module):  
 def \_\_init\_\_(self, d_model, num_heads):  
 super().\_\_init\_\_()  
 assert d_model % num_heads \== 0, "d_model must be divisible by num_heads"

        self.d\_k \= d\_model // num\_heads
        self.num\_heads \= num\_heads

        \# Линейные проекции. Обратите внимание: мы используем один большой слой
        \# для всех голов сразу, а затем разделяем их. Это вычислительно эффективнее.
        self.w\_q \= nn.Linear(d\_model, d\_model)
        self.w\_k \= nn.Linear(d\_model, d\_model)
        self.w\_v \= nn.Linear(d\_model, d\_model)
        self.w\_o \= nn.Linear(d\_model, d\_model)

    def forward(self, query, key, value, mask=None):
        batch\_size \= query.size(0)

        \# 1\. Проекция и изменение формы (Reshape)
        \# Превращаем (Batch, Seq\_Len, D\_Model) \-\> (Batch, Seq\_Len, Num\_Heads, D\_k)
        Q \= self.w\_q(query).view(batch\_size, \-1, self.num\_heads, self.d\_k)
        K \= self.w\_k(key).view(batch\_size, \-1, self.num\_heads, self.d\_k)
        V \= self.w\_v(value).view(batch\_size, \-1, self.num\_heads, self.d\_k)

        \# 2\. Транспонирование для матричного умножения
        \# Нам нужно, чтобы измерение Heads было перед Seq\_Len, чтобы PyTorch
        \# мог распараллелить вычисления для каждой головы как для отдельного элемента батча.
        \# Результат: (Batch, Num\_Heads, Seq\_Len, D\_k)
        Q \= Q.transpose(1, 2)
        K \= K.transpose(1, 2)
        V \= V.transpose(1, 2)

Важный момент: использование .view требует, чтобы тензор был непрерывным (contiguous) в памяти. Если тензор был получен через сложные операции среза (slicing), может потребоваться вызов .contiguous() перед .view. Транспонирование .transpose(1, 2\) делает тензор "разрывным" в памяти, что важно учитывать при дальнейших операциях изменения формы.19

### **4.2 Вычисление внимания: matmul vs einsum**

Ядро механизма — вычисление $QK^T$. В PyTorch это можно сделать двумя способами.

**Способ А: torch.matmul**

Python

\# K.transpose(-2, \-1) меняет местами последние два измерения (Seq_Len и D_k)  
\# Q: (B, H, L_q, D_k)  
\# K^T: (B, H, D_k, L_k)  
\# Scores: (B, H, L_q, L_k)  
scores \= torch.matmul(Q, K.transpose(-2, \-1)) / math.sqrt(self.d_k)

Это стандартный подход. Он надежен и хорошо оптимизирован.21

Способ Б: torch.einsum  
Эйнштейновское суммирование позволяет записать операцию более декларативно.

Python

scores \= torch.einsum('bhqd, bhkd \-\> bhqk', Q, K) / math.sqrt(self.d_k)

Здесь bhqd означает индексы (Batch, Head, Query_pos, Dimension), а bhkd — (Batch, Head, Key_pos, Dimension). Суммирование происходит по индексу d.  
Хотя einsum читается лучше, на практике с ним связаны проблемы. В некоторых версиях PyTorch и на определенных GPU einsum может вызывать ошибки нехватки памяти (CUDA OOM), так как он может создавать промежуточные копии тензоров или использовать менее оптимизированные ядра по сравнению с прямым вызовом cuBLAS через matmul.23 Для реализации "с нуля" в учебных целях einsum прекрасен, но в продакшене чаще используют matmul или специализированные ядра.

### **4.3 Маскирование (Masking)**

Маски бывают двух типов:

1. **Padding Mask:** Игнорирует токены-заполнители (\<pad\>), добавленные для выравнивания длин предложений в батче.
2. **Look-ahead (Causal) Mask:** Используется в декодерах (например, в GPT) для предотвращения "подглядывания" модели на следующие токены.

Маскирование реализуется через добавление "минус бесконечности" к логитам перед Softmax.

Python

if mask is not None:  
 \# mask имеет форму (Batch, 1, 1, Seq_Len) или (Seq_Len, Seq_Len)  
 \# 0 обозначает позицию, которую нужно замаскировать  
 scores \= scores.masked_fill(mask \== 0, float('-inf'))

Почему \-inf? Потому что $e^{-\\infty} \= 0$. После прохождения через Softmax вероятность внимания к этим позициям станет строгим нулем.21

### **4.4 Численная стабильность Softmax: Трюк Log-Sum-Exp**

При реализации Softmax "руками" (если мы не используем F.softmax) необходимо учитывать риск переполнения (overflow). Если $z\_i \= 1000$, то $e^{1000}$ вызовет переполнение числа с плавающей точкой (inf).  
Стандартный трюк для численной стабильности (реализованный внутри F.softmax) — вычитание максимума:

$$\\text{softmax}(z\_i) \= \\frac{e^{z\_i \- M}}{\\sum e^{z\_j \- M}}, \\quad \\text{где } M \= \\max(z)$$

Вычитание константы из экспоненты не меняет итоговое соотношение вероятностей, но гарантирует, что максимальная степень экспоненты равна 0 ($e^0 \= 1$), что исключает переполнение.25

### **4.5 Агрегация и финализация**

Python

\# Применяем Softmax. dim=-1 означает применение вдоль последнего измерения (по ключам)  
attn_weights \= F.softmax(scores, dim=-1)

\# Агрегация значений  
\# (B, H, L_q, L_k) x (B, H, L_k, D_k) \-\> (B, H, L_q, D_k)  
output \= torch.matmul(attn_weights, V)

\# 3\. Объединение голов (Concat)  
\# Сначала транспонируем обратно: (B, L_q, H, D_k)  
\# Затем выпрямляем в (B, L_q, H \* D_k)  
output \= output.transpose(1, 2).contiguous().view(batch_size, \-1, self.d_model)

\# Финальная линейная проекция  
output \= self.w_o(output)  
return output, attn_weights

## ---

**5\. Multi-Head Attention: Ансамбль фокусов**

Почему недостаточно одной головы внимания? Концепция Multi-Head Attention (MHA) является развитием идеи сверточных каналов в CNN.

### **5.1 Подпространства представлений**

В естественном языке слова имеют множество аспектов отношений: синтаксические (подлежащее-сказуемое), семантические (синонимы, антонимы), референциальные (местоимения). Одна голова внимания может усреднять эти отношения, теряя нюансы. MHA позволяет модели формировать несколько независимых "взглядов" на последовательность.7

**Таблица 2\. Примеры специализации голов внимания (на основе анализа BERT/GPT)**

| Тип головы          | Функция                            | Пример в предложении "The animal didn't cross it..." |
| :------------------ | :--------------------------------- | :--------------------------------------------------- |
| **Синтаксическая**  | Отслеживание грамматических связей | Связывает "**animal**" (сущ) с "didn't cross" (глаг) |
| **Позиционная**     | Фокус на соседних токенах          | Связывает "**it**" с предыдущим словом "cross"       |
| **Референциальная** | Разрешение анафоры (coreference)   | Связывает "**it**" с "**animal**" (а не с "street")  |

### **5.2 Экономика параметров**

Распространенное заблуждение: "Больше голов — больше параметров". Это не так. Размерность проекций для каждой головы $d\_k$ уменьшается пропорционально количеству голов $h$.

$$d\_{model} \= 512, \\quad h \= 8 \\implies d\_k \= 64$$

Матрицы весов для одной головы имеют размер $512 \\times 64$. Для 8 голов: $8 \\times (512 \\times 64\) \= 512 \\times 512$.  
Таким образом, общее количество параметров в слоях $W^Q, W^K, W^V$ остается неизменным, независимо от количества голов (при условии $h \\times d\_k \= d\_{model}$). Мы получаем богатство представлений без увеличения вычислительной стоимости.20

## ---

**6\. Визуализация и интерпретируемость: Читаем мысли модели?**

Матрица весов внимания (attn_weights), возвращаемая нашей функцией, представляет собой карту "мыслей" модели. Однако интерпретация этих карт вызывает жаркие споры в научном сообществе.

### **6.1 Инструменты визуализации: Тепловые карты и BertViz**

Простейший способ визуализации — построение тепловой карты (heatmap) с использованием библиотек matplotlib или seaborn. По оси X откладываются токены-ключи, по оси Y — токены-запросы. Яркость ячейки $(i, j)$ соответствует силе внимания.30  
При построении таких карт важно:

- Использовать правильную цветовую схему (sequential colormap), так как веса всегда положительны.30
- Добавлять аннотации значений в ячейки для детального анализа, если матрица небольшая.31

Для более сложного анализа используется **BertViz** — специализированный инструмент, позволяющий интерактивно исследовать внимание на трех уровнях:

1. **Model View:** Обзор всех слоев и голов. Позволяет увидеть глобальные паттерны.
2. **Head View:** Детальный вид связей для конкретной головы.
3. **Neuron View:** Визуализация того, как отдельные нейроны в векторах Query и Key формируют итоговый вес внимания.33

### **6.2 Полемика: "Attention is (not) Explanation"**

Является ли внимание объяснением причинно-следственной связи в решениях модели?

Аргумент "ПРОТИВ" (Jain & Wallace, 2019):  
В статье "Attention is not Explanation" 35 авторы показали, что:

1. **Отсутствие корреляции:** Веса внимания часто слабо коррелируют с важностью признаков, вычисленной через градиентные методы (например, LIME).
2. **Контрфактуальность:** Можно искусственно создать альтернативную матрицу внимания (adversarial attention), которая радикально отличается от исходной, но приводит к _тому же самому_ предсказанию модели. Если внимание можно изменить без изменения результата, значит, оно не является причиной результата.35

Аргумент "ЗА" (Wiegreffe & Pinter, 2019):  
В ответной статье "Attention is not not Explanation" 37 были выдвинуты контраргументы:

1. **Определение объяснения:** Jain & Wallace требовали "верности" (faithfulness) механизму, игнорируя "правдоподобие" (plausibility) для человека.
2. **Тест на необходимость:** Хотя можно найти адверсальные веса, они часто вырождены. В нормальных условиях обучения модель действительно использует внимание как канал передачи информации. Внимание — это _необходимое_ условие для передачи информации от токена А к токену Б.
3. **Диагностическая ценность:** Внимание показывает _поток информации_. Если вес высок, информация была передана. Использовала ли модель эту информацию для вывода — другой вопрос, но канал связи неоспорим.37

**Вывод:** Внимание следует интерпретировать не как "причину решения" (reasoning), а как "маршрутизацию данных" (routing). Это карта дорог, по которым ехала информация, но не обязательно карта намерений водителя.

## ---

**7\. Вычислительная реальность и оптимизация**

Несмотря на элегантность, "заклинание" имеет высокую цену. Сложность стандартного алгоритма внимания составляет $O(L^2)$ как по времени, так и по памяти.

### **7.1 Квадратичное проклятие**

Матрица внимания имеет размер $L \\times L$. Для последовательности длиной 1024 токена это $10^6$ элементов. Для 32k токенов (как в GPT-4) это $10^9$ элементов. В формате FP16 (2 байта) одна такая матрица занимает 2 ГБ. Учитывая количество слоев и голов, батчинг становится невозможным, а память GPU исчерпывается мгновенно.40

Более того, проблема не только в объеме памяти, но и в пропускной способности (Memory Bandwidth). Чтение и запись огромных матриц внимания в HBM (High Bandwidth Memory) видеокарты занимает больше времени, чем само вычисление.41

### **7.2 FlashAttention: Революция IO**

В 2022 году Tri Dao представил FlashAttention — алгоритм, который изменил правила игры. Основная идея: не материализовывать полную матрицу внимания в HBM.  
Вместо этого FlashAttention:

1. Разбивает входные данные на блоки (тайлинг).
2. Загружает блоки в сверхбыструю память SRAM (кэш) графического процессора.
3. Вычисляет внимание и Softmax внутри SRAM.
4. Записывает в HBM только результат.
5. В обратном проходе (backward pass) он _пересчитывает_ внимание заново, вместо того чтобы хранить его (recomputation).

Это кажется парадоксальным (делать больше вычислений, чтобы работать быстрее), но поскольку доступ к памяти на порядки медленнее вычислений, FlashAttention ускоряет обучение в 2-4 раза и позволяет линейно масштабировать длину контекста.41

### **7.3 PyTorch 2.0 и scaled_dot_product_attention**

Начиная с версии 2.0, PyTorch внедрил функцию torch.nn.functional.scaled_dot_product_attention (SDPA). Это "фьюзнутое" (fused) ядро, которое автоматически выбирает наилучшую реализацию в зависимости от оборудования и параметров:

- FlashAttention (если доступно GPU Ampere/Hopper и FP16/BF16).
- Memory-Efficient Attention (от xFormers).
- Стандартная C++ реализация (Math).

Использование этой функции является "золотым стандартом" современной разработки, так как она снимает с разработчика необходимость ручной оптимизации памяти.43

**Пример использования SDPA:**

Python

\# Вместо ручного matmul \+ softmax  
output \= F.scaled_dot_product_attention(  
 query=Q,  
 key=K,  
 value=V,  
 attn_mask=mask,  
 dropout_p=0.1,  
 is_causal=False \# Автоматическая каузальная маска при True  
)

Важно отметить ограничение: SDPA в PyTorch 2.0 не поддерживает произвольные маски внимания при использовании FlashAttention (только каузальные или отсутствие маски), хотя это ограничение постепенно снимается в новых версиях (FlexAttention).41

## ---

**8\. Заключение**

Реализация "Заклинания Фокуса" — Scaled Dot-Product Attention — требует баланса между глубоким пониманием линейной алгебры и навыками системного программирования. Мы прошли путь от геометрической интуиции скалярного произведения, через статистическое обоснование масштабирующего множителя $\\sqrt{d\_k}$, к деталям реализации на PyTorch и вопросам современной оптимизации.

Ключевые выводы:

1. **Скалярное произведение** — это эффективный метод поиска подобия, но он требует контроля дисперсии.
2. **Масштабирование** критически важно для предотвращения исчезновения градиентов в Softmax.
3. **Multi-Head Attention** позволяет модели учить разнообразные типы отношений без увеличения числа параметров.
4. **Визуализация** внимания полезна для диагностики, но не является исчерпывающим объяснением логики модели.
5. **Оптимизация памяти** (FlashAttention) — ключ к работе с длинными последовательностями, превращающий квадратичную сложность по памяти в линейную.

Владение этим механизмом открывает двери к пониманию и созданию архитектур уровня GPT, BERT и новейших мультимодальных моделей.

### ---

**Приложение: Сравнительная таблица метрик подобия**

| Метрика                | Формула                | Преимущества                                   | Недостатки                                |
| :--------------------- | :--------------------- | :--------------------------------------------- | :---------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------- | ------------------------------------------------------- |
| **Dot Product**        | $A \\cdot B$           | Высокая скорость (MatMul), оптимизация на GPU. | Неограниченная магнитуда, рост дисперсии. |
| **Cosine Similarity**  | $\\frac{A \\cdot B}{\\ | A\\                                            | \\                                        | $                                                                  | Нормализация (-1 до 1), инвариантность к масштабу. | Требует вычисления норм (медленнее), сложнее градиенты. |
| **Euclidean Distance** | $\\                    | A \- B$                                        | Геометрическая интерпретация расстояния.  | Вычислительно дороже ($x^2$), хуже работает в высокой размерности. |

### **Приложение: Влияние масштабирования ($\\sqrt{d\_k}$)**

| Характеристика        | Без масштабирования            | С масштабированием (dk​​1​) |
| :-------------------- | :----------------------------- | :-------------------------- |
| **Дисперсия $QK^T$**  | $d\_k$ (растет с размерностью) | $\\approx 1$ (стабильна)    |
| **Градиенты Softmax** | Исчезающие (Vanishing)         | Стабильные                  |
| **Обучение**          | Нестабильное / Расхождение     | Быстрая сходимость          |

#### **Источники**

1. Self-Attention & Multi-Head Attention Made Simple | by Hugo Le Picard, PhD | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@lepicardhugo/attention-from-first-principles-to-production-the-fundamentals-def39bee8f46](https://medium.com/@lepicardhugo/attention-from-first-principles-to-production-the-fundamentals-def39bee8f46)
2. Attention is All you Need \- NIPS papers, дата последнего обращения: декабря 20, 2025, [https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
3. Measuring Similarity and Distance between Embeddings \- Dataquest, дата последнего обращения: декабря 20, 2025, [https://www.dataquest.io/blog/measuring-similarity-and-distance-between-embeddings/](https://www.dataquest.io/blog/measuring-similarity-and-distance-between-embeddings/)
4. how does the dot product determine similarity? \- Mathematics Stack Exchange, дата последнего обращения: декабря 20, 2025, [https://math.stackexchange.com/questions/689022/how-does-the-dot-product-determine-similarity](https://math.stackexchange.com/questions/689022/how-does-the-dot-product-determine-similarity)
5. Vector Similarity Explained \- Pinecone, дата последнего обращения: декабря 20, 2025, [https://www.pinecone.io/learn/vector-similarity/](https://www.pinecone.io/learn/vector-similarity/)
6. Purpose of sqrt(dim(k)) in Scaled dot product attention \- DeepLearning.AI Community, дата последнего обращения: декабря 20, 2025, [https://community.deeplearning.ai/t/purpose-of-sqrt-dim-k-in-scaled-dot-product-attention/62880](https://community.deeplearning.ai/t/purpose-of-sqrt-dim-k-in-scaled-dot-product-attention/62880)
7. I Finally Understood “Attention is All You Need” After So Long. Here's How I Did It., дата последнего обращения: декабря 20, 2025, [https://ai.plainenglish.io/i-finally-understood-attention-is-all-you-need-after-so-long-heres-how-i-did-it-263b46273f9f](https://ai.plainenglish.io/i-finally-understood-attention-is-all-you-need-after-so-long-heres-how-i-did-it-263b46273f9f)
8. Understanding Query, Key, Value in Transformers and LLMs | by Charles Chi | AI \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/ai-assimilating-intelligence/understanding-query-key-value-in-transformers-c579b93054cc](https://medium.com/ai-assimilating-intelligence/understanding-query-key-value-in-transformers-c579b93054cc)
9. \[D\] What is the rationale behind self-attention equation and how did they came up with the concept query, key and value? : r/MachineLearning \- Reddit, дата последнего обращения: декабря 20, 2025, [https://www.reddit.com/r/MachineLearning/comments/bkw2xp/d_what_is_the_rationale_behind_selfattention/](https://www.reddit.com/r/MachineLearning/comments/bkw2xp/d_what_is_the_rationale_behind_selfattention/)
10. The Comprehensive Guide to Dating App Development \- Fulminous Software, дата последнего обращения: декабря 20, 2025, [https://fulminoussoftware.com/the-comprehensive-guide-to-dating-app-development](https://fulminoussoftware.com/the-comprehensive-guide-to-dating-app-development)
11. The Unexpected Love Affair: How AI Transforms Tinder's Dating Experience?, дата последнего обращения: декабря 20, 2025, [https://www.analyticsvidhya.com/blog/2023/05/the-unexpected-love-affair-how-ai-transforms-tinders-dating-experience/](https://www.analyticsvidhya.com/blog/2023/05/the-unexpected-love-affair-how-ai-transforms-tinders-dating-experience/)
12. \[D\] How to truly understand attention mechanism in transformers? : r/MachineLearning \- Reddit, дата последнего обращения: декабря 20, 2025, [https://www.reddit.com/r/MachineLearning/comments/qidpqx/d_how_to_truly_understand_attention_mechanism_in/](https://www.reddit.com/r/MachineLearning/comments/qidpqx/d_how_to_truly_understand_attention_mechanism_in/)
13. What is an attention mechanism? | IBM, дата последнего обращения: декабря 20, 2025, [https://www.ibm.com/think/topics/attention-mechanism](https://www.ibm.com/think/topics/attention-mechanism)
14. дата последнего обращения: декабря 20, 2025, [https://kargarisaac.medium.com/inside-transformers-an-in-depth-look-at-the-game-changing-machine-learning-architecture-part-3-429858be2f6f\#:\~:text=In%20the%20case%20of%20the,input%20to%20the%20softmax%20function.](https://kargarisaac.medium.com/inside-transformers-an-in-depth-look-at-the-game-changing-machine-learning-architecture-part-3-429858be2f6f#:~:text=In%20the%20case%20of%20the,input%20to%20the%20softmax%20function.)
15. Why does this multiplication of $Q$ and $K$ have a variance of $d\_k$, in scaled dot product attention? \- Artificial Intelligence Stack Exchange, дата последнего обращения: декабря 20, 2025, [https://ai.stackexchange.com/questions/21237/why-does-this-multiplication-of-q-and-k-have-a-variance-of-d-k-in-scaled](https://ai.stackexchange.com/questions/21237/why-does-this-multiplication-of-q-and-k-have-a-variance-of-d-k-in-scaled)
16. What is the rationale behind square root scaling in attention \- DeepLearning.AI Community, дата последнего обращения: декабря 20, 2025, [https://community.deeplearning.ai/t/what-is-the-rationale-behind-square-root-scaling-in-attention/441193](https://community.deeplearning.ai/t/what-is-the-rationale-behind-square-root-scaling-in-attention/441193)
17. Understanding Scaling in Scaled Dot-Product Attention: Simple Math Insights \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@jennifer.zzz/why-scaling-is-important-in-scaled-dot-product-attention-dca8d8cb9504](https://medium.com/@jennifer.zzz/why-scaling-is-important-in-scaled-dot-product-attention-dca8d8cb9504)
18. Why use a "square root" in the scaled dot product \- Artificial Intelligence Stack Exchange, дата последнего обращения: декабря 20, 2025, [https://ai.stackexchange.com/questions/41861/why-use-a-square-root-in-the-scaled-dot-product](https://ai.stackexchange.com/questions/41861/why-use-a-square-root-in-the-scaled-dot-product)
19. Attention is all you need: Discovering the Transformer paper | Towards Data Science, дата последнего обращения: декабря 20, 2025, [https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634/](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634/)
20. Transformers From Scratch: Part 3 — Multi-Head Attention | by Kari Vierimaa | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@kavierim/transformers-from-scratch-part-3-multi-head-attention-d1a3a061ba89](https://medium.com/@kavierim/transformers-from-scratch-part-3-multi-head-attention-d1a3a061ba89)
21. Implement Self-Attention and Cross-Attention in Pytorch | by Hey Amit \- Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@heyamit10/implement-self-attention-and-cross-attention-in-pytorch-cfe17ab0b3ee](https://medium.com/@heyamit10/implement-self-attention-and-cross-attention-in-pytorch-cfe17ab0b3ee)
22. How do I implement this attention layer in PyTorch? \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/76648620/how-do-i-implement-this-attention-layer-in-pytorch](https://stackoverflow.com/questions/76648620/how-do-i-implement-this-attention-layer-in-pytorch)
23. Regarding Scaled Dot Product Attention \- nlp \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/regarding-scaled-dot-product-attention/211744](https://discuss.pytorch.org/t/regarding-scaled-dot-product-attention/211744)
24. Scaled Dot-Product Attention and Masking in Transformers | CodeSignal Learn, дата последнего обращения: декабря 20, 2025, [https://codesignal.com/learn/courses/sequence-models-the-dawn-of-attention-1/lessons/scaled-dot-product-attention-and-masking-in-transformers-1](https://codesignal.com/learn/courses/sequence-models-the-dawn-of-attention-1/lessons/scaled-dot-product-attention-and-masking-in-transformers-1)
25. Softmax Activation Function in Python: A Complete Guide | DataCamp, дата последнего обращения: декабря 20, 2025, [https://www.datacamp.com/tutorial/softmax-activation-function-in-python](https://www.datacamp.com/tutorial/softmax-activation-function-in-python)
26. Numerically Stable Softmax \- Brian Lester, дата последнего обращения: декабря 20, 2025, [https://blester125.com/blog/softmax.html](https://blester125.com/blog/softmax.html)
27. Numerically stable softmax \- python \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/42599498/numerically-stable-softmax](https://stackoverflow.com/questions/42599498/numerically-stable-softmax)
28. MultiheadAttention — PyTorch 2.9 documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
29. \[D\] Multihead attention dimensionalities : r/MachineLearning \- Reddit, дата последнего обращения: декабря 20, 2025, [https://www.reddit.com/r/MachineLearning/comments/q5rhu8/d_multihead_attention_dimensionalities/](https://www.reddit.com/r/MachineLearning/comments/q5rhu8/d_multihead_attention_dimensionalities/)
30. Heatmap: A Complete Guide | Atlassian, дата последнего обращения: декабря 20, 2025, [https://www.atlassian.com/data/charts/heatmap-complete-guide](https://www.atlassian.com/data/charts/heatmap-complete-guide)
31. Creating and Customizing Heatmaps in Matplotlib \- Python-Fiddle, дата последнего обращения: декабря 20, 2025, [https://python-fiddle.com/tutorials/matplotlib-heatmap](https://python-fiddle.com/tutorials/matplotlib-heatmap)
32. how to annotate heatmap with text in matplotlib \- Stack Overflow, дата последнего обращения: декабря 20, 2025, [https://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib](https://stackoverflow.com/questions/11917547/how-to-annotate-heatmap-with-text-in-matplotlib)
33. BertViz: Visualize Attention in NLP Models (BERT, GPT2, BART, etc.) \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)
34. Explainable AI: Visualizing Attention in Transformers \- MLOps Community, дата последнего обращения: декабря 20, 2025, [https://mlops.community/explainable-ai-visualizing-attention-in-transformers/](https://mlops.community/explainable-ai-visualizing-attention-in-transformers/)
35. Attention is not not Explanation \- ACL Anthology, дата последнего обращения: декабря 20, 2025, [https://aclanthology.org/D19-1002.pdf](https://aclanthology.org/D19-1002.pdf)
36. A Song of (Dis)agreement: Evaluating the Evaluation of Explainable Artificial Intelligence in Natural Language Processing \- HHAI Conferences, дата последнего обращения: декабря 20, 2025, [https://www.hhai-conference.org/wp-content/uploads/2022/06/hhai-2022_paper_21.pdf](https://www.hhai-conference.org/wp-content/uploads/2022/06/hhai-2022_paper_21.pdf)
37. Attention is not not explanation \- Ben-Gurion University Research Portal, дата последнего обращения: декабря 20, 2025, [https://cris.bgu.ac.il/en/publications/attention-is-not-not-explanation-2/](https://cris.bgu.ac.il/en/publications/attention-is-not-not-explanation-2/)
38. \[1908.04626\] Attention is not not Explanation \- arXiv, дата последнего обращения: декабря 20, 2025, [https://arxiv.org/abs/1908.04626](https://arxiv.org/abs/1908.04626)
39. Attention is not not Explanation. \[This post is intended for an NLP… | by Yuval Pinter | Medium, дата последнего обращения: декабря 20, 2025, [https://medium.com/@yuvalpinter/attention-is-not-not-explanation-dbc25b534017](https://medium.com/@yuvalpinter/attention-is-not-not-explanation-dbc25b534017)
40. lucidrains/memory-efficient-attention-pytorch: Implementation of a memory efficient multi-head attention as proposed in the paper, "Self-attention Does Not Need O(n²) Memory" \- GitHub, дата последнего обращения: декабря 20, 2025, [https://github.com/lucidrains/memory-efficient-attention-pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch)
41. Out of the box acceleration and memory savings of decoder models with PyTorch 2.0, дата последнего обращения: декабря 20, 2025, [https://pytorch.org/blog/out-of-the-box-acceleration/](https://pytorch.org/blog/out-of-the-box-acceleration/)
42. Softmax function \- Wikipedia, дата последнего обращения: декабря 20, 2025, [https://en.wikipedia.org/wiki/Softmax_function](https://en.wikipedia.org/wiki/Softmax_function)
43. torch.nn.functional.scaled_dot_product_attention — PyTorch 2.9 documentation, дата последнего обращения: декабря 20, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
44. Scaled_dot_product_attention is not numerically stable \- PyTorch Forums, дата последнего обращения: декабря 20, 2025, [https://discuss.pytorch.org/t/scaled-dot-product-attention-is-not-numerically-stable/182582](https://discuss.pytorch.org/t/scaled-dot-product-attention-is-not-numerically-stable/182582)
45. FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention, дата последнего обращения: декабря 20, 2025, [https://pytorch.org/blog/flexattention/](https://pytorch.org/blog/flexattention/)
