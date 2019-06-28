# Теоретический материал

# Очновные понятия регрессионого анализа

Регрессио́нный анализ — статистический метод исследования влияния одной или нескольких независимых переменных X1, X2, ... , Xn на зависимую переменную Y. Независимые переменные иначе называют регрессорами или предикторами, а зависимые переменные — критериальными.

Уравнение регрессии — это математическая формула, применяемая к независимым переменным, чтобы лучше спрогнозировать зависимую переменную, которую необходимо смоделировать. 

Уравнение регрессии может выглядеть следующим образом: ![equation](https://latex.codecogs.com/gif.latex?y&space;=&space;\beta_0&space;&plus;&space;\beta_1*x_1&space;&plus;&space;\beta_2*x_2&space;&plus;...&space;&plus;&space;\beta_n*x_n&space;&plus;&space;\varepsilon)

Зависимая переменная (Y) — это переменная, описывающая процесс, который мы пытаемся предсказать или понять. 

Независимые переменные (X) — это переменные, используемые для моделирования или прогнозирования значений зависимых переменных. В уравнении регрессии они располагаются справа от знака равенства и часто называются объяснительными переменными. Зависимая переменная - это функция независимых переменных. 

Коэффициенты регрессии (β) — это коэффициенты, которые рассчитываются в результате выполнения регрессионного анализа. Вычисляются величины для каждой независимой переменной, которые представляют силу и тип взаимосвязи независимой переменной по отношению к зависимой. 

Невязки. Существует необъяснимое количество зависимых величин, представленных в уравнении регрессии как случайные ошибки ε.

# Нахождение оценок коэффициентов регрессии, решением системы нормальных уравнении
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

![equation](https://latex.codecogs.com/gif.latex?\left\{&space;\begin{array}{l}&space;b_0&plus;b_1\overline&space;x&plus;b_2\overline{x^2}&plus;\dots&plus;b_k\overline{x^k}=\overline&space;y\\&space;b_0\overline&space;x&plus;b_1\overline{x^2}&plus;b_2\overline{x^3}&plus;\dots&plus;b_k\overline{x^{k&plus;1}}=\overline{xy}\\&space;b_0\overline{x^2}&plus;b_1\overline{x^3}&plus;b_2\overline{x^4}&plus;\dots&plus;b_k\overline{x^{k&plus;2}}=\overline{x^2y}\\&space;\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\ldots\\&space;b_0\overline{x^k}&plus;b_1\overline{x^{k&plus;1}}&plus;b_2\overline{x^{k&plus;2}}&plus;\dots&plus;b_k\overline{x^{2k}}=\overline{x^ky}&space;\end{array}&space;\right.\&space;\&space;\&space;\&space;\&space;(3))

Полученную систему можно записать в матричном виде: AB = C, где 

![equation](https://latex.codecogs.com/gif.latex?A=\left(&space;\begin{array}{ccccc}&space;1&space;&&space;\overline&space;x&space;&&space;\overline{x^2}&space;&&space;\ldots&space;&&space;\overline{x^k}\\&space;\overline&space;x&space;&&space;\overline{x^2}&space;&&space;\overline{x^3}&space;&&space;\ldots&space;&&space;\overline{x^{k&plus;1}}\\&space;\overline{x^2}&space;&&space;\overline{x^3}&space;&&space;\overline{x^4}&space;&&space;\ldots&space;&&space;\overline{x^{k&plus;2}}\\&space;\vdots&space;&&space;\vdots&space;&&space;\vdots&space;&&space;\ddots&space;&&space;\vdots\\&space;\overline{x^k}&space;&&space;\overline{x^{k&plus;1}}&space;&&space;\overline{x^{k&plus;2}}&space;&&space;\ldots&space;&&space;\overline{x^{2k}}&space;\end{array}&space;\right),\&space;\&space;B=\left(\begin{array}{c}&space;b_0\\b_1\\b_2\\\vdots\\b_k&space;\end{array}&space;\right),\&space;\&space;C=\left(\begin{array}{c}&space;\overline&space;y\\\overline{xy}\\\overline{x^2y}\\\vdots\\\overline{x^ky}&space;\end{array}&space;\right).)

Если вывести В, то мы получим вектор-столбец элементы которого, будут являтся оценками коэффициентов уравнения регрессии.  
