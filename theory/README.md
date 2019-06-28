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
