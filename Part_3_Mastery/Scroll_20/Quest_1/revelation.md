# **Всесторонний аналитический отчет: Методологии, алгоритмы и практическая реализация генерации синтетических временных рядов**

## **Введение**

В современной экосистеме данных, характеризующейся экспоненциальным ростом объемов информации и усложнением аналитических моделей, проблема доступности качественных данных для обучения и тестирования алгоритмов становится критической. В рамках задачи «Квест 20.1: Сбор данных о времени» мы сталкиваемся с фундаментальной потребностью в создании контролируемых экспериментальных сред. Синтетические данные временных рядов (Synthetic Time Series Data) представляют собой искусственно сгенерированные последовательности данных, которые воспроизводят статистические свойства, временные зависимости и структурные компоненты реальных наблюдений, но при этом не содержат чувствительной информации и полностью подконтрольны исследователю.1

Актуальность создания таких датасетов обусловлена несколькими факторами. Во-первых, это дефицит реальных данных, особенно содержащих редкие аномалии или специфические паттерны «черных лебедей», необходимых для стресс-тестирования систем обнаружения сбоев.3 Во-вторых, вопросы конфиденциальности (GDPR, CCPA) часто делают невозможным использование производственных данных для разработки и отладки моделей.1 В-третьих, синтетические данные позволяют проводить «what-if» анализ, моделируя сценарии, которые еще не происходили в реальности, например, поведение финансовых рынков в условиях беспрецедентной волатильности или реакцию IoT-сенсоров на экстремальные климатические условия.4

Данный отчет представляет собой исчерпывающее руководство по методам создания синтетических временных рядов. Мы детально рассмотрим математическую декомпозицию рядов на тренд, сезонность и шум, проанализируем алгоритмические подходы к их генерации с использованием библиотек Python (NumPy, Pandas, SciPy), изучим методы инъекции аномалий для проверки надежности моделей и обсудим лучшие практики визуализации и валидации полученных данных. Особое внимание будет уделено нюансам программной реализации, таким как векторизация вычислений, управление генераторами случайных чисел и обработка граничных условий в мультипликативных моделях.

## **Теоретические основы декомпозиции временных рядов**

Создание синтетического временного ряда базируется на обратном процессе анализа временных рядов — декомпозиции. Предполагается, что любой сложный сигнал $Y(t)$ можно разложить на более простые, интерпретируемые компоненты. Понимание природы взаимодействия этих компонентов является ключом к выбору правильной архитектуры генератора.

### **Аддитивная модель: Линейная суперпозиция**

Аддитивная модель является наиболее интуитивно понятным подходом к формированию временного ряда. Она постулирует, что наблюдаемое значение является арифметической суммой независимых компонентов.

Математическая формулировка:

$$Y(t) \= T(t) \+ S(t) \+ E(t)$$

где:

- $Y(t)$ — значение ряда в момент времени $t$.
- $T(t)$ — трендовая составляющая (Trend).
- $S(t)$ — сезонная составляющая (Seasonality).
- $E(t)$ — случайная ошибка или шум (Error/Noise).

Характеристики и применение:  
Ключевая особенность аддитивной модели заключается в том, что амплитуда сезонных колебаний и дисперсия шума остаются постоянными, независимо от уровня тренда.5 Например, если мы моделируем потребление воды в городе, и оно растет линейно из\-за прироста населения, но суточные колебания (утро/вечер) остаются в пределах одних и тех же абсолютных значений (литров), то аддитивная модель будет корректной аппроксимацией.7  
С точки зрения генерации кода, аддитивная модель реализуется через поэлементное сложение векторов. Это делает её вычислительно эффективной и устойчивой к ошибкам, так как сложение нечувствительно к нулям или отрицательным значениям в отдельных компонентах.8 Если $T(t)$ становится отрицательным, это просто смещает ряд вниз, не искажая форму сезонности.

### **Мультипликативная модель: Нелинейное взаимодействие**

Мультипликативная модель предполагает, что компоненты взаимодействуют друг с другом пропорционально. Это означает, что изменения в одном компоненте масштабируют остальные.

Математическая формулировка:

$$Y(t) \= T(t) \\times S(t) \\times E(t)$$  
Характеристики и применение:  
В данной модели амплитуда сезонности и волатильность шума изменяются вместе с уровнем тренда.7 Это типично для экономических и финансовых показателей. Рассмотрим продажи растущей компании электронной коммерции: если ежедневные продажи вырастают с 100 до 10 000 единиц, сезонный всплеск перед праздниками будет составлять не фиксированные \+50 единиц, а, например, \+20% от базового уровня.10 Мультипликативная модель естественным образом захватывает эту динамику.  
Сложности реализации:  
Генерация мультипликативных рядов требует особого внимания к области значений компонентов.

1. **Проблема нуля:** Если в какой-либо момент времени $T(t)$ или $S(t)$ обращаются в ноль, весь ряд «схлопывается» в ноль, теряя информацию о других компонентах.
2. **Проблема знака:** Отрицательные значения в тренде или сезонности могут инвертировать ряд, превращая пики во впадины, что часто лишено физического смысла (например, отрицательная цена акции или отрицательный объем продаж).11
3. Логарифмическая трансформация: Для безопасной генерации мультипликативных рядов часто используется переход в логарифмическое пространство, где умножение превращается в сложение:

   $$\\log(Y(t)) \= \\log(T(t)) \+ \\log(S(t)) \+ \\log(E(t))$$

   После генерации в логарифмическом масштабе применяется экспонирование для получения финального ряда. Это гарантирует положительность значений (так как $e^x \> 0$).13

### **Сравнительная таблица моделей**

Ниже приведена таблица, систематизирующая различия между моделями для выбора правильной стратегии генерации.5

| Характеристика             | Аддитивная модель                                              | Мультипликативная модель                                  |
| :------------------------- | :------------------------------------------------------------- | :-------------------------------------------------------- |
| **Формула**                | $Y \= T \+ S \+ E$                                             | $Y \= T \\times S \\times E$                              |
| **Зависимость сезонности** | Амплитуда постоянна (абсолютные значения)                      | Амплитуда пропорциональна тренду (относительные значения) |
| **Характер шума**          | Гомоскедастичность (постоянная дисперсия)                      | Гетероскедастичность (дисперсия растет с уровнем)         |
| **Примеры использования**  | Температура, физические измерения с фиксированной погрешностью | Продажи, пассажиропоток, цены акций, вирусная нагрузка    |
| **Требования к данным**    | Допускает отрицательные значения                               | Обычно требует положительных значений                     |
| **Метод декомпозиции**     | Вычитание (observed \- trend)                                  | Деление (observed / trend)                                |

## **Методология генерации детерминированных компонентов**

Детерминированная часть временного ряда — это сигнал, который мы можем описать аналитической функцией. В контексте Python и библиотеки NumPy, генерация этих компонентов базируется на векторизованных операциях над временным индексом.

### **Формирование временного индекса**

Основой любого синтетического ряда является временная ось. В Python для этого используются pandas.date_range или numpy.arange.

- pd.date_range(start, periods, freq) позволяет создавать сложные календари (рабочие дни, исключая выходные), что критично для реалистичности бизнес-данных.14
- Для чисто математического моделирования часто достаточно числового индекса $t \= 0, 1,..., N$, создаваемого через np.arange(N).

### **Генерация Трендов (Trend)**

Тренд отражает долгосрочную эволюцию процесса. Выбор функции тренда определяет «историю», которую рассказывает синтетический датасет.

#### **Линейные и полиномиальные тренды**

Самый простой вариант — линейный тренд, описываемый уравнением прямой $y \= mx \+ c$. В коде это реализуется через np.linspace(start_val, end_val, n), что создает равномерно распределенные значения между начальной и конечной точкой.4  
Для моделирования ускорения или замедления используются полиномы более высоких степеней (квадратичные, кубические). Например, квадратичный тренд $y \= t^2$ моделирует процесс с постоянным ускорением.8

#### **Логистические (S-образные) тренды**

В реальности бесконечный линейный или экспоненциальный рост невозможен из\-за физических ограничений (насыщение рынка, емкость сервера). Для таких случаев используется логистическая функция:

$$f(x) \= \\frac{L}{1 \+ e^{-k(x-x\_0)}}$$

Где $L$ — предел насыщения, $k$ — крутизна роста. Реализация этой функции позволяет создавать ряды, которые начинаются с медленного роста, ускоряются, а затем выходят на плато.15

#### **Стохастические тренды (Случайные блуждания)**

Для финансового моделирования детерминированные тренды часто заменяются стохастическими. Классическим примером является «Случайное блуждание» (Random Walk), где значение в момент $t$ зависит от значения в момент $t-1$ плюс случайный шаг:

$$y\_t \= y\_{t-1} \+ \\epsilon\_t$$

В NumPy это элегантно реализуется через кумулятивную сумму шума: trend \= np.cumsum(np.random.normal(0, 1, n)). Добавление константы к шуму («дрейф») превращает это в «Случайное блуждание с дрейфом» (Random Walk with Drift), что имитирует общий тренд движения рынка при сохранении локальной непредсказуемости.4

### **Генерация Сезонности (Seasonality)**

Сезонность описывает периодические, повторяющиеся паттерны.

#### **Синусоидальная сезонность**

Базовый метод моделирования сезонности — использование тригонометрических функций. Гармоническое колебание задается как $A \\sin(2\\pi ft \+ \\phi)$.4

- **Множественная сезонность:** Реальные данные часто содержат наложения циклов (например, суточный и годовой). Принцип суперпозиции позволяет просто складывать синусоиды с разными частотами: seasonality \= sin(daily) \+ sin(yearly).15
- **Преимущества:** Математическая чистота, дифференцируемость.
- **Недостатки:** Слишком «гладкая» форма, редко встречающаяся в данных, порожденных деятельностью человека (продажи, трафик).

#### **Календарная и дискретная сезонность**

Человеческая активность часто имеет резкие перепады, например, падение трафика в выходные дни («эффект уикенда»). Для моделирования таких паттернов синусоида не подходит. Вместо этого используются методы отображения (mapping).  
Мы можем использовать оператор по модулю % для определения дня недели или часа дня.

- _Алгоритм:_ Создать массив, где для каждого $t$, если t % 7 (день недели) равен 5 или 6 (суббота, воскресенье), значение сезонности низкое, иначе — высокое.16
- Это создает «блочную» или ступенчатую структуру сезонности, которая гораздо реалистичнее для моделирования бизнес-процессов, чем плавная синусоида.

## **Стохастическое моделирование: Шум и Вариативность**

Именно наличие стохастического компонента (шума) превращает набор функций в имитацию реальных данных. Шум скрывает сигнал, затрудняя работу алгоритмов прогнозирования и создавая необходимые условия для тестирования их робастности.

### **Гауссовский (Белый) Шум**

Стандартный подход предполагает использование Гауссовского белого шума, где значения независимы и нормально распределены с нулевым средним: $E \\sim N(0, \\sigma^2)$.17  
В коде: noise \= np.random.normal(loc=0, scale=std_dev, size=n). Параметр scale (стандартное отклонение) является рычагом управления уровнем «зашумленности» или отношением сигнал/шум (SNR).

### **Коррелированный (Красный) Шум**

Белый шум слишком «резкий» для многих физических процессов (инерция температуры) или экономических систем (рыночный импульс). В таких системах ошибка в момент $t$ коррелирует с ошибкой в момент $t-1$. Такой шум называется «Красным» или Броуновским.  
Он может быть сгенерирован через авторегрессионный процесс AR(1):

$$e\_t \= \\rho e\_{t-1} \+ \\xi\_t$$

Где $\\rho$ — коэффициент автокорреляции, а $\\xi\_t$ — белый шум. Использование красного шума делает синтетические данные более плавными и сложными для простых статистических тестов на стационарность.19

### **Гетероскедастичность (Непостоянная дисперсия)**

Реальные временные ряды часто демонстрируют кластеризацию волатильности — периоды спокойствия сменяются периодами высокой турбулентности. Моделирование этого эффекта (гетероскедастичности) критично для финансового риск-менеджмента.  
Вместо константы в параметре scale функции np.random.normal, можно передать массив той же длины, что и генерируемый ряд. Этот массив может содержать участки с высокими и низкими значениями, модулируя локальную дисперсию шума во времени.9

## **Методы инъекции аномалий и стресс-тестирование**

Одной из главных целей создания синтетических данных является тестирование систем обнаружения аномалий (Anomaly Detection). Поскольку в реальных данных аномалии редки и часто не размечены, синтетический подход позволяет создать «Золотой стандарт» (Ground Truth) с точно известными типами и координатами аномалий.21

### **Классификация и генерация аномалий**

1. Точечные аномалии (Spikes/Outliers):  
   Резкие, кратковременные выбросы.
   - _Механизм:_ Выбор случайных индексов и добавление к ним значения, многократно превышающего стандартное отклонение шума (например, $6\\sigma$).
   - _Код:_ series\[indices\] \+= magnitude \* sign. Важно учитывать знак, чтобы создавать выбросы как вверх, так и вниз.23
2. Сдвиг уровня (Level Shift):  
   Внезапное изменение базового уровня ряда (скачок среднего значения). Это имитирует сбой оборудования или структурное изменение рынка.
   - _Механизм:_ Добавление константы ко всем значениям ряда, начиная с момента времени $t\_{change}$.
   - _Код:_ series\[t_change:\] \+= shift_value.15
3. Изменение тренда (Trend Change):  
   Изменение скорости роста или падения.
   - _Механизм:_ Изменение коэффициента наклона линейного тренда в определенной точке. Это требует сегментированной генерации тренда или использования кусочно-линейных функций.25
4. Аномалии дисперсии (Variance Change):  
   Внезапное увеличение «шумности» сигнала без изменения среднего значения.
   - _Механизм:_ Умножение компонента шума на коэффициент $\>1$ на определенном интервале. Это сложный тип аномалии для обнаружения, так как простые методы сглаживания могут его пропустить.27

### **Алгоритмическая реализация инъекций**

Для создания качественного датасета рекомендуется инкапсулировать логику инъекций в класс или набор функций, которые не только модифицируют ряд, но и возвращают маску аномалий (вектор из 0 и 1), где 1 отмечает аномальное наблюдение. Это необходимо для автоматического расчета метрик качества (Precision/Recall) алгоритмов детекции.25

## **Анализ программного кода и лучшие практики реализации**

Реализация генератора временных рядов требует понимания принципов эффективного программирования на Python, особенно в контексте обработки больших массивов данных.

### **Векторизация против Итерации**

Ключевой принцип при работе с временными рядами в Python — избегать циклов for при математических операциях. Библиотека NumPy позволяет выполнять операции над целыми массивами (векторами) на порядки быстрее благодаря реализации на языке C.

- _Плохой подход:_ Цикл по каждому дню для вычисления синуса.
- _Хороший подход (Векторизация):_ time \= np.arange(N); values \= np.sin(time). Это не только быстрее, но и делает код более читаемым и близким к математической нотации.14

### **Управление воспроизводимостью (Seeding)**

Синтетические данные должны быть случайными, но воспроизводимыми. Это парадокс решается использованием семян генератора случайных чисел (Random Seeds).

- _Практика:_ Обязательная инициализация np.random.seed(42) в начале скрипта. Это гарантирует, что каждый запуск скрипта генерирует идентичный набор «случайных» чисел, идентичное расположение шума и аномалий. Без этого невозможно сравнивать работу моделей между итерациями, так как изменения в метриках могут быть вызваны просто другой реализацией шума.28

### **Обработка граничных условий в мультипликативных моделях**

При реализации кода для мультипликативных моделей ($Y \= T \\times S \\times E$) критически важно следить за тем, чтобы базовые компоненты не принимали значений, нарушающих логику модели.

- Если $T(t)$ близко к нулю, умножение на шум может создать артефакты.
- Отрицательные значения в данных, которые физически должны быть положительными (например, концентрация вещества), требуют применения функций ограничения (clipping) np.clip(data, 0, None) или использования логарифмически-нормальных распределений для генерации шума, которые гарантируют положительность.11

### **Модульность и архитектура**

Рекомендуется использовать объектно-ориентированный подход, создавая классы для каждого компонента (например, LinearTrend, SinusoidalSeasonality), которые наследуются от базового класса TimeSeriesComponent. Это позволяет легко комбинировать различные типы трендов и сезонностей, собирая сложный ряд как конструктор. Такой подход используется в специализированных библиотеках, но его легко реализовать и самостоятельно для кастомных нужд.19

## **Визуализация и Валидация синтетических данных**

Создание данных — это лишь половина задачи. Вторая половина — доказать, что они качественные и подходят для решения поставленной задачи.

### **Визуальный анализ**

1. Графики временного ряда (Line Plots):  
   Основной инструмент первичной оценки. Важно использовать библиотеки типа Matplotlib или Plotly для построения графиков, где по оси X отложено время.
   - _Best Practice:_ Всегда подписывать оси и добавлять легенду. Для длинных рядов имеет смысл отображать только репрезентативный срез (зумирование), чтобы не потерять детализацию структуры за плотностью точек.29
2. Декомпозиционные графики:  
   Функция statsmodels.tsa.seasonal.seasonal_decompose позволяет «разобрать» сгенерированный ряд обратно на компоненты. Визуальное сравнение сгенерированного тренда и извлеченного тренда является отличным способом отладки генератора. Если они сильно расходятся, это может указывать на то, что шум слишком силен или модель сезонности выбрана неверно.31
3. Визуализация аномалий:  
   Для отчетов и презентаций критично выделять инъецированные аномалии. Использование функции axvspan (вертикальная заливка) в Matplotlib позволяет закрасить фоновым цветом временные интервалы, где присутствуют аномалии. Это дает мгновенное визуальное понимание того, как аномалия выглядит на фоне нормальных данных.32

### **Метрики валидации**

Как убедиться, что синтетический ряд «похож» на реальный или соответствует ожиданиям?

| Тип метрики                                 | Описание                                                                  | Инструмент/Метод                                                                                                                                              |
| :------------------------------------------ | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Fidelity (Статистическая точность)**      | Насколько распределение значений синтетического ряда совпадает с целевым. | Тест Колмогорова-Смирнова (KS-test), сравнение гистограмм, QQ-плоты.34                                                                                        |
| **Temporal Dynamics (Временная структура)** | Сохранение автокорреляционных свойств.                                    | Графики ACF (автокорреляционная функция) и PACF. Если синтетический шум белый, ACF должна быстро падать до нуля.18                                            |
| **Utility (Полезность)**                    | TSTR (Train on Synthetic, Test on Real).                                  | Обучение модели прогнозирования на синтетических данных и проверка её точности на реальных. Если модель работает, значит синтетика захватила суть процесса.36 |
| **Spectral Consistency**                    | Совпадение частотных характеристик.                                       | Сравнение спектральной плотности мощности (PSD) через Преобразование Фурье (FFT).38                                                                           |

### **Практическая значимость спектрального анализа**

Использование Быстрого преобразования Фурье (FFT) позволяет увидеть скрытые цикличности. Если мы генерируем сложную сезонность (неделя \+ год), на спектрограмме должны быть четкие пики на частотах $1/7$ и $1/365$. Наличие паразитных пиков может указывать на алиасинг (наложение частот) или ошибки в формулах генерации.38

## **Заключение**

Разработка синтетических временных рядов — это дисциплина, находящаяся на стыке статистики, математического моделирования и программной инженерии. Успех «Квеста 20.1» зависит не просто от вызова функции random, а от глубокого понимания природы моделируемого процесса.

Выбор между аддитивной и мультипликативной моделями диктуется физикой процесса (зависит ли вариативность от масштаба?). Выбор функции тренда определяет долгосрочную динамику сценария. Тонкая настройка параметров шума (цвет, распределение, гетероскедастичность) превращает стерильную математическую кривую в реалистичный поток данных, пригодный для тренировки нейросетей. А компетентная инъекция аномалий превращает датасет из простого хранилища чисел в мощный полигон для стресс-тестирования аналитических систем.

Приведенные в отчете методологии и аналитические выкладки по коду обеспечивают надежный фундамент для создания гибких, масштабируемых и статистически достоверных синтетических наборов данных, способных удовлетворить самые взыскательные требования современных задач Data Science.

#### **Источники**

1. What is Synthetic Data? Examples, Use Cases and Benefits \- HabileData, дата последнего обращения: декабря 21, 2025, [https://www.habiledata.com/blog/what-is-synthetic-data-examples-use-cases-benefits/](https://www.habiledata.com/blog/what-is-synthetic-data-examples-use-cases-benefits/)
2. What is synthetic data? \- MOSTLY AI, дата последнего обращения: декабря 21, 2025, [https://mostly.ai/synthetic-data-basics](https://mostly.ai/synthetic-data-basics)
3. дата последнего обращения: декабря 21, 2025, [https://syntheticus.ai/guide-everything-you-need-to-know-about-synthetic-data\#:\~:text=Stress%20testing%20and%20scenario%20analysis\&text=Synthetic%20data%20enables%20institutions%20to,range%20of%20potential%20market%20conditions.](https://syntheticus.ai/guide-everything-you-need-to-know-about-synthetic-data#:~:text=Stress%20testing%20and%20scenario%20analysis&text=Synthetic%20data%20enables%20institutions%20to,range%20of%20potential%20market%20conditions.)
4. Generating Time Series Data for Python Analysis \- Index.dev, дата последнего обращения: декабря 21, 2025, [https://www.index.dev/blog/generate-time-series-data-python](https://www.index.dev/blog/generate-time-series-data-python)
5. дата последнего обращения: декабря 21, 2025, [https://milvus.io/ai-quick-reference/what-is-the-difference-between-additive-and-multiplicative-time-series-models\#:\~:text=Additive%20and%20multiplicative%20time%20series%20models%20are%20two%20approaches%20to,multiplicative%20models%20assume%20they%20multiply.](https://milvus.io/ai-quick-reference/what-is-the-difference-between-additive-and-multiplicative-time-series-models#:~:text=Additive%20and%20multiplicative%20time%20series%20models%20are%20two%20approaches%20to,multiplicative%20models%20assume%20they%20multiply.)
6. What is the difference between additive and multiplicative time, дата последнего обращения: декабря 21, 2025, [https://milvus.io/ai-quick-reference/what-is-the-difference-between-additive-and-multiplicative-time-series-models](https://milvus.io/ai-quick-reference/what-is-the-difference-between-additive-and-multiplicative-time-series-models)
7. Additive models and multiplicative models \- Support \- Minitab, дата последнего обращения: декабря 21, 2025, [https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/time-series/supporting-topics/time-series-models/additive-and-multiplicative-models/](https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/time-series/supporting-topics/time-series-models/additive-and-multiplicative-models/)
8. How to Decompose Time Series Data into Trend and Seasonality \- MachineLearningMastery.com, дата последнего обращения: декабря 21, 2025, [https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)
9. Time Series Analysis, Concept, Additive and Multiplicative Models \- the intact one, дата последнего обращения: декабря 21, 2025, [https://theintactone.com/2019/08/28/bsa-u2-topic-1-time-series-analysis-concept-additive-and-multiplicative-models/](https://theintactone.com/2019/08/28/bsa-u2-topic-1-time-series-analysis-concept-additive-and-multiplicative-models/)
10. What is the difference between additive and multiplicative time series models? \- Zilliz, дата последнего обращения: декабря 21, 2025, [https://zilliz.com/ai-faq/what-is-the-difference-between-additive-and-multiplicative-time-series-models](https://zilliz.com/ai-faq/what-is-the-difference-between-additive-and-multiplicative-time-series-models)
11. Negative values in time series forecast \- python \- Stack Overflow, дата последнего обращения: декабря 21, 2025, [https://stackoverflow.com/questions/55079109/negative-values-in-time-series-forecast](https://stackoverflow.com/questions/55079109/negative-values-in-time-series-forecast)
12. ThetaForecaster does not work with time series containig negative values \#940 \- GitHub, дата последнего обращения: декабря 21, 2025, [https://github.com/alan-turing-institute/sktime/issues/940](https://github.com/alan-turing-institute/sktime/issues/940)
13. Time Series Decomposition: Understand Trends, Seasonality, and Noise \- DataCamp, дата последнего обращения: декабря 21, 2025, [https://www.datacamp.com/tutorial/time-series-decomposition](https://www.datacamp.com/tutorial/time-series-decomposition)
14. 10 Useful NumPy One-Liners for Time Series Analysis \- MachineLearningMastery.com, дата последнего обращения: декабря 21, 2025, [https://machinelearningmastery.com/10-useful-numpy-one-liners-for-time-series-analysis/](https://machinelearningmastery.com/10-useful-numpy-one-liners-for-time-series-analysis/)
15. Synthetic Time Series Data Generation: ThirdEye Data, дата последнего обращения: декабря 21, 2025, [https://thirdeyedata.ai/synthetic-time-series-data-generation/](https://thirdeyedata.ai/synthetic-time-series-data-generation/)
16. Simple Synthetic Time Series Data \- YData, дата последнего обращения: декабря 21, 2025, [https://ydata.ai/resources/simple-synthetic-time-series-data.html](https://ydata.ai/resources/simple-synthetic-time-series-data.html)
17. numpy.random.normal — NumPy v2.5.dev0 Manual, дата последнего обращения: декабря 21, 2025, [https://numpy.org/devdocs/reference/random/generated/numpy.random.normal.html](https://numpy.org/devdocs/reference/random/generated/numpy.random.normal.html)
18. White Noise | Python, дата последнего обращения: декабря 21, 2025, [https://campus.datacamp.com/courses/time-series-analysis-in-python/some-simple-time-series?ex=4](https://campus.datacamp.com/courses/time-series-analysis-in-python/some-simple-time-series?ex=4)
19. Simple timeseries generation in Python with mockseries | by Cyril de Catheu | Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@cdecatheu/simple-timeseries-generation-in-python-with-mockseries-d6b214111814](https://medium.com/@cdecatheu/simple-timeseries-generation-in-python-with-mockseries-d6b214111814)
20. Time Series Data Transformation using Python \- GeeksforGeeks, дата последнего обращения: декабря 21, 2025, [https://www.geeksforgeeks.org/machine-learning/time-series-data-transformation-using-python/](https://www.geeksforgeeks.org/machine-learning/time-series-data-transformation-using-python/)
21. Implementing Time Series Anomaly Detection in Python: Catching the Outliers That Matter, дата последнего обращения: декабря 21, 2025, [https://medium.com/@deolesopan/implementing-time-series-anomaly-detection-in-python-catching-the-outliers-that-matter-96eefc04493b](https://medium.com/@deolesopan/implementing-time-series-anomaly-detection-in-python-catching-the-outliers-that-matter-96eefc04493b)
22. Create Synthetic Time-series with Anomaly Signatures in Python \- KDnuggets, дата последнего обращения: декабря 21, 2025, [https://www.kdnuggets.com/2021/10/synthetic-time-series-anomaly-signatures-python.html](https://www.kdnuggets.com/2021/10/synthetic-time-series-anomaly-signatures-python.html)
23. How to Detect Anomalies in Time Series Data in Python \- Statology, дата последнего обращения: декабря 21, 2025, [https://www.statology.org/how-to-detect-anomalies-in-time-series-data-in-python/](https://www.statology.org/how-to-detect-anomalies-in-time-series-data-in-python/)
24. How to perform anomaly detection in time series data with python? Methods, Code, Example\! \- Medium, дата последнего обращения: декабря 21, 2025, [https://medium.com/@goldengoat/how-to-perform-anomaly-detection-in-time-series-data-with-python-methods-code-example-e83b9c951a37](https://medium.com/@goldengoat/how-to-perform-anomaly-detection-in-time-series-data-with-python-methods-code-example-e83b9c951a37)
25. A Practical Toolkit for Time Series Anomaly Detection, Using Python | Towards Data Science, дата последнего обращения: декабря 21, 2025, [https://towardsdatascience.com/a-practical-toolkit-for-time-series-anomaly-detection-using-python/](https://towardsdatascience.com/a-practical-toolkit-for-time-series-anomaly-detection-using-python/)
26. Maximize Your Time Series Analysis with Python's Change Point Detection Tools, дата последнего обращения: декабря 21, 2025, [https://medium.datadriveninvestor.com/maximize-your-time-series-analysis-with-pythons-change-point-detection-tools-39ce2bc63be](https://medium.datadriveninvestor.com/maximize-your-time-series-analysis-with-pythons-change-point-detection-tools-39ce2bc63be)
27. Intervention Detection in Python Time Series (Pulse, Trend, Shift) \- Stack Overflow, дата последнего обращения: декабря 21, 2025, [https://stackoverflow.com/questions/17242836/intervention-detection-in-python-time-series-pulse-trend-shift](https://stackoverflow.com/questions/17242836/intervention-detection-in-python-time-series-pulse-trend-shift)
28. Time Series Decomposition Techniques \- GeeksforGeeks, дата последнего обращения: декабря 21, 2025, [https://www.geeksforgeeks.org/python/time-series-decomposition-techniques/](https://www.geeksforgeeks.org/python/time-series-decomposition-techniques/)
29. Plotting Time-Series Data: A Practical Guide for Data Scientists | by Damilare Daramola, дата последнего обращения: декабря 21, 2025, [https://medium.com/@iamdamilare13/plotting-time-series-data-a-practical-guide-for-data-scientists-95e149db3d87](https://medium.com/@iamdamilare13/plotting-time-series-data-a-practical-guide-for-data-scientists-95e149db3d87)
30. Quick start guide — Matplotlib 3.10.8 documentation, дата последнего обращения: декабря 21, 2025, [https://matplotlib.org/stable/users/explain/quick_start.html](https://matplotlib.org/stable/users/explain/quick_start.html)
31. Seasonality, Trend and Noise | Chan\`s Jupyter, дата последнего обращения: декабря 21, 2025, [https://goodboychan.github.io/python/datacamp/time_series_analysis/visualization/2020/06/13/01-Seasonality-Trend-and-Noise.html](https://goodboychan.github.io/python/datacamp/time_series_analysis/visualization/2020/06/13/01-Seasonality-Trend-and-Noise.html)
32. How To Highlight a Time Range in Time Series Plot in Python with Matplotlib?, дата последнего обращения: декабря 21, 2025, [https://www.geeksforgeeks.org/python/how-to-highlight-a-time-range-in-time-series-plot-in-python-with-matplotlib/](https://www.geeksforgeeks.org/python/how-to-highlight-a-time-range-in-time-series-plot-in-python-with-matplotlib/)
33. Highlight sequence of points in matplotlib \[duplicate\] \- Stack Overflow, дата последнего обращения: декабря 21, 2025, [https://stackoverflow.com/questions/69537672/highlight-sequence-of-points-in-matplotlib](https://stackoverflow.com/questions/69537672/highlight-sequence-of-points-in-matplotlib)
34. Metrics for Evaluating Synthetic Time-Series Data of Battery \- MDPI, дата последнего обращения: декабря 21, 2025, [https://www.mdpi.com/2076-3417/14/14/6088](https://www.mdpi.com/2076-3417/14/14/6088)
35. How to Visually Evaluate Your Synthetic Data Quality? \- YData, дата последнего обращения: декабря 21, 2025, [https://ydata.ai/resources/how-to-visually-evaluate-your-synthetic-data-quality.html](https://ydata.ai/resources/how-to-visually-evaluate-your-synthetic-data-quality.html)
36. A Comparative Study of Open-Source Libraries for Synthetic Tabular Data Generation: SDV vs. SynthCity \- arXiv, дата последнего обращения: декабря 21, 2025, [https://arxiv.org/html/2506.17847v1](https://arxiv.org/html/2506.17847v1)
37. How to evaluate the quality of the synthetic data – measuring from the perspective of fidelity, utility, and privacy | Artificial Intelligence \- AWS, дата последнего обращения: декабря 21, 2025, [https://aws.amazon.com/blogs/machine-learning/how-to-evaluate-the-quality-of-the-synthetic-data-measuring-from-the-perspective-of-fidelity-utility-and-privacy/](https://aws.amazon.com/blogs/machine-learning/how-to-evaluate-the-quality-of-the-synthetic-data-measuring-from-the-perspective-of-fidelity-utility-and-privacy/)
38. Analyzing seasonality of Google trend time series using FFT \- Stack Overflow, дата последнего обращения: декабря 21, 2025, [https://stackoverflow.com/questions/52690632/analyzing-seasonality-of-google-trend-time-series-using-fft](https://stackoverflow.com/questions/52690632/analyzing-seasonality-of-google-trend-time-series-using-fft)
