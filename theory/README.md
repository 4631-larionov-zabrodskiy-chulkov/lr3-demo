# Теоретический материал

## Очновные понятия регрессионого анализа

Регрессио́нный анализ — статистический метод исследования влияния одной или нескольких независимых переменных X1, X2, ... , Xn на зависимую переменную Y. Независимые переменные иначе называют регрессорами или предикторами, а зависимые переменные — критериальными.

Уравнение регрессии — это математическая формула, применяемая к независимым переменным, чтобы лучше спрогнозировать зависимую переменную, которую необходимо смоделировать. 

Уравнение регрессии может выглядеть следующим образом: ![equation](https://latex.codecogs.com/gif.latex?y&space;=&space;\beta_0&space;&plus;&space;\beta_1*x_1&space;&plus;&space;\beta_2*x_2&space;&plus;...&space;&plus;&space;\beta_n*x_n&space;&plus;&space;\varepsilon)

Зависимая переменная (Y) — это переменная, описывающая процесс, который мы пытаемся предсказать или понять. 

Независимые переменные (X) — это переменные, используемые для моделирования или прогнозирования значений зависимых переменных. В уравнении регрессии они располагаются справа от знака равенства и часто называются объяснительными переменными. Зависимая переменная - это функция независимых переменных. 

Коэффициенты регрессии (β) — это коэффициенты, которые рассчитываются в результате выполнения регрессионного анализа. Вычисляются величины для каждой независимой переменной, которые представляют силу и тип взаимосвязи независимой переменной по отношению к зависимой. 

Невязки. Существует необъяснимое количество зависимых величин, представленных в уравнении регрессии как случайные ошибки ε.

## Нахождение оценок коэффициентов регрессии, решением системы нормальных уравнении
Пусть даны два ряда наблюдений ![equation](https://latex.codecogs.com/gif.latex?x_i) (независимая переменная) и ![equation](https://latex.codecogs.com/gif.latex?y_i) (зависимая переменная), ![equation](https://latex.codecogs.com/gif.latex?i=\overline{1,n}). Уравнение полинома имеет вид

![equation](https://latex.codecogs.com/gif.latex?y=\sum\limits_{j=0}^k&space;b_jx^j,\&space;\&space;\&space;\&space;\&space;(1))

,где ![equation](https://latex.codecogs.com/gif.latex?b_j) — параметры данного полинома, ![equation](https://latex.codecogs.com/gif.latex?j=\overline{0,k}). Среди них ![equation](https://latex.codecogs.com/gif.latex?b_0) — свободный член. Найдём по методу наименьших квадратов (МНК - метод наименьших квадратов) параметры ![equation](https://latex.codecogs.com/gif.latex?b_j) данной регрессии.
По аналогии с линейной регрессией, МНК также основан на минимизации следующего выражения:

![equation](https://latex.codecogs.com/gif.latex?S=\sum\limits_{i=1}^n\left(\hat&space;y_i-y_i\right)^2\to\min\&space;\&space;\&space;\&space;\&space;(2))

Здесь ![equation](https://latex.codecogs.com/gif.latex?\hat&space;y_i) — теоретические значения, являющиеся значениями полинома (1) в точках ![equation](https://latex.codecogs.com/gif.latex?x_i). Подставляя (1) в (2), получаем

![equation](https://latex.codecogs.com/gif.latex?S=\sum\limits_{i=1}^n\left(\sum_{j=0}^kb_jx_i^j-y_i\right)^2\to\min.)

На основании необходимого условия экстремума функции (k + 1) переменных ![equation](https://latex.codecogs.com/gif.latex?S=S(b_0,&space;b_1,\dots,b_k)) приравняем к нулю её частные производные, т.е. 

![equation](https://latex.codecogs.com/gif.latex?S'_{b_p}=2\sum\limits_{i=1}^nx_i^p\left(\sum\limits_{j=0}^kb_jx_i^j-y_i\right)=0,\&space;\&space;\&space;p=\overline{0,k}.)

Поделив левую и правую часть каждого равенства на 2, раскроем вторую сумму:
![equation](https://latex.codecogs.com/gif.latex?\sum\limits_{i=1}^nx_i^p\left(b_0&plus;b_1x_i&plus;b_2x_i^2&plus;\dots&plus;b_kx_i^k\right)-\sum\limits_{i=1}^nx_i^py_i=0,\&space;\&space;\&space;p=\overline{0,k}.)

Раскрывая скобки, перенесём в каждом ![equation](https://latex.codecogs.com/gif.latex?p)-ом выражении последнее слагаемое с ![equation](https://latex.codecogs.com/gif.latex?y_i) вправо и поделим обе части на ![equation](https://latex.codecogs.com/gif.latex?n). В результате у нас получилось (k + 1) выражений, образующие систему линейных нормальных уравнений относительно ![equation](https://latex.codecogs.com/gif.latex?b_p). Она имеет следующий вид:

<img src="https://latex.codecogs.com/gif.latex?\left\{&space;\begin{array}{l}&space;b_0&plus;b_1\overline&space;x&plus;b_2\overline{x^2}&plus;\dots&plus;b_k\overline{x^k}=\overline&space;y\\&space;b_0\overline&space;x&plus;b_1\overline{x^2}&plus;b_2\overline{x^3}&plus;\dots&plus;b_k\overline{x^{k&plus;1}}=\overline{xy}\\&space;b_0\overline{x^2}&plus;b_1\overline{x^3}&plus;b_2\overline{x^4}&plus;\dots&plus;b_k\overline{x^{k&plus;2}}=\overline{x^2y}\\&space;\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\\&space;b_0\overline{x^k}&plus;b_1\overline{x^{k&plus;1}}&plus;b_2\overline{x^{k&plus;2}}&plus;\dots&plus;b_k\overline{x^{2k}}=\overline{x^ky}&space;\end{array}&space;\right.\&space;\&space;\&space;\&space;\&space;(3)" title="\left\{ \begin{array}{l} b_0+b_1\overline x+b_2\overline{x^2}+\dots+b_k\overline{x^k}=\overline y\\ b_0\overline x+b_1\overline{x^2}+b_2\overline{x^3}+\dots+b_k\overline{x^{k+1}}=\overline{xy}\\ b_0\overline{x^2}+b_1\overline{x^3}+b_2\overline{x^4}+\dots+b_k\overline{x^{k+2}}=\overline{x^2y}\\ \ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\\ b_0\overline{x^k}+b_1\overline{x^{k+1}}+b_2\overline{x^{k+2}}+\dots+b_k\overline{x^{2k}}=\overline{x^ky} \end{array} \right.\ \ \ \ \ (3)" />

Полученную систему можно записать в матричном виде: AB = C, где 

<img src="https://latex.codecogs.com/gif.latex?A=\left(&space;\begin{array}{ccccc}&space;1&space;&&space;\overline&space;x&space;&&space;\overline{x^2}&space;&&space;\ldots&space;&&space;\overline{x^k}\\&space;\overline&space;x&space;&&space;\overline{x^2}&space;&&space;\overline{x^3}&space;&&space;\ldots&space;&&space;\overline{x^{k&plus;1}}\\&space;\overline{x^2}&space;&&space;\overline{x^3}&space;&&space;\overline{x^4}&space;&&space;\ldots&space;&&space;\overline{x^{k&plus;2}}\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots\\&space;\overline{x^k}&space;&&space;\overline{x^{k&plus;1}}&space;&&space;\overline{x^{k&plus;2}}&space;&&space;\ldots&space;&&space;\overline{x^{2k}}&space;\end{array}&space;\right),\&space;\&space;B=\left(\begin{array}{c}&space;b_0\\b_1\\b_2\\\vdots\\b_k&space;\end{array}&space;\right),\&space;\&space;C=\left(\begin{array}{c}&space;\overline&space;y\\\overline{xy}\\\overline{x^2y}\\\vdots\\\overline{x^ky}&space;\end{array}&space;\right)." title="A=\left( \begin{array}{ccccc} 1 & \overline x & \overline{x^2} & \ldots & \overline{x^k}\\ \overline x & \overline{x^2} & \overline{x^3} & \ldots & \overline{x^{k+1}}\\ \overline{x^2} & \overline{x^3} & \overline{x^4} & \ldots & \overline{x^{k+2}}\\ \vdots & \vdots & \vdots & \ddots & \vdots\\ \overline{x^k} & \overline{x^{k+1}} & \overline{x^{k+2}} & \ldots & \overline{x^{2k}} \end{array} \right),\ \ B=\left(\begin{array}{c} b_0\\b_1\\b_2\\\vdots\\b_k \end{array} \right),\ \ C=\left(\begin{array}{c} \overline y\\\overline{xy}\\\overline{x^2y}\\\vdots\\\overline{x^ky} \end{array} \right)." />

Если вывести В, то мы получим вектор-столбец элементы которого, будут являтся оценками коэффициентов уравнения регрессии.  

## Критерий Фишера. Адекватность уравнения регрессии.

Значимость уравнения множественной регрессии в целом, так же как и в парной регрессии, оценивается с помощью F-критерия Фишера. Данный критерий очень важен в регрессионном анализе и по существу является частным случаем проверки ограничений. В данном случае нулевая гипотеза — об одновременном равенстве нулю всех коэффициентов при факторах регрессионной модели (то есть всего ограничений k-1). В данном случае короткая модель — это просто константа в качестве фактора, то есть коэффициент детерминации короткой модели равен нулю. Статистика теста равна:

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/8305967e1dc48a283ff5b866ea379be37f747916), ![alt_text](https://students-library.com/files/5/126/mnozhestvennaja-regressija-i-korreljacija_8.gif), ![alt_text](https://students-library.com/files/5/126/mnozhestvennaja-regressija-i-korreljacija_9.gif)

## Пример:

Пусть оценивается линейная регрессия доли расходов на питание в общей сумме расходов на константу, логарифм совокупных расходов, количество взрослых членов семьи и количество детей до 11 лет. То есть всего в модели 4 оцениваемых параметра (k=4). Пусть по результатам оценки регрессии получен коэффициент детерминации ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/ad9960cd1be8d25072c4aa6e628a7ce66467dd39)  По вышеприведенной формуле рассчитаем значение F-статистики в случае, если регрессия оценена по данным 34 наблюдений и по данным 64 наблюдений:![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/d3209f2a2882d9de5f18fd905f3ecef96e1b5c2f) 

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/450b736e0dde3b51d330db1b6825fccdf4a3f4c1)

Критическое значение статистики при 1 % уровне значимости в первом случае равно ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/3e806d8c52f96f6a5ddb4d629135eb58f738792d)  а во втором случае ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/f59588943799176e7c7400b9ad3844aff0ead97b) В обоих случаях регрессия признается значимой при заданном уровне значимости. В первом случае P-значение равно 0,1 %, а во втором — 0,00005 %. Таким образом, во втором случае уверенность в значимости регрессии существенно выше (существенно меньше вероятность ошибки в случае признания модели значимой).

## Критерий Стьюдента

Проверка статистической значимости параметров регрессионного уравнения (коэффициентов регрессии) выполняется по t-критерию Стьюдента, который рассчитывается по формуле:

![alt text](https://www.chem-astu.ru/science/reference/formula_t.gif)

где P - значение параметра;
      Sp - стандартное отклонение параметра.

Рассчитанное значение критерия Стьюдента сравнивают с его табличным значением при выбранной доверительной вероятности (как правило, 0.95) и числе степеней свободы N-k-1, где N-число точек, k-число переменных в регрессионном уравнении (например, для линейной модели Y=A*X+B подставляем k=1).

Если вычисленное значение tp выше, чем табличное, то коэффициент регрессии является значимым с данной доверительной вероятностью. В противном случае есть основания для исключения соответствующей переменной из регрессионной модели.

Величины параметров и их стандартные отклонения обычно рассчитываются в алгоритмах, реализующих метод наименьших квадратов.

## Пример:

Для изучения эффективности нового препарата железа были выбраны две группы пациентов с анемией. В первой группе пациенты в течение двух недель получали новый препарат, а во второй группе - получали плацебо. После этого было проведено измерение уровня гемоглобина в периферической крови. В первой группе средний уровень гемоглобина составил 115,4±1,2 г/л, а во второй - 103,7±2,3 г/л (данные представлены в формате M±m), сравниваемые совокупности имеют нормальное распределение. При этом численность первой группы составила 34, а второй - 40 пациентов. Необходимо сделать вывод о статистической значимости полученных различий и эффективности нового препарата железа.
Для оценки значимости различий используем t-критерий Стьюдента, рассчитываемый как разность средних значений, поделенная на сумму квадратов ошибок:

![alt_text](http://medstatistic.ru/theory/example_student.png)

После выполнения расчетов, значение t-критерия оказалось равным 4,51. Находим число степеней свободы как (34 + 40) - 2 = 72. Сравниваем полученное значение t-критерия Стьюдента 4,51 с критическим при р=0,05 значением, указанным в таблице: 1,993. Так как рассчитанное значение критерия больше критического, делаем вывод о том, что наблюдаемые различия статистически значимы (уровень значимости р<0,05).

## Справка по работе с библиотекой Scikit-learn 

Для того, чтобы понять необходимость полиномиальной регрессии, сперва сгенерируем случайный набор данных.

```
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)
plt.scatter(x,y, s=10)
plt.show()
```
Эти данные на графике будут выглядеть так:
![alt_text](https://cdn-images-1.medium.com/max/1600/1*dJhMB97nyUB6_OgSECKxEQ.png)

Теперь применим модель линейной регрессии к этому набору данных.

```
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

# transforming the data to include another axis
x = x[:, np.newaxis]
y = y[:, np.newaxis]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

plt.scatter(x, y, s=10)
plt.plot(x, y_pred, color='r')
plt.show()
```
Получили график самой подходящей линии
![alt_text](https://cdn-images-1.medium.com/max/1600/1*yim5OMiku3dNMXEv3GiItg.png)

По графику можем заметить, что прямая не может захватить все экспериментальные данные. Вычисление RMSE(среднеквадратичной ошибки) и R²-показателя(коэффициента детерминации) линейной регрессии дает: 
```
RMSE of linear regression is 15.908242501429998.
R2 score of linear regression is 0.6386750054827146
```
Чтобы преодолеть несостыковки, нам нужно повысить сложность модели. 
Чтобы сгенерировать уравнение более высокого порядка, мы можем добавить возможности оригинальных функций в качестве новых функций. 
Линейная модель, 

![alt_text](https://cdn-images-1.medium.com/max/1600/1*adrhNj5POluyuFCa9WfBIg.png)

может быть преобразована в 

![alt_text](https://cdn-images-1.medium.com/max/1600/1*rL76rQ1hhrvPjAQFwvpN4w.png)



