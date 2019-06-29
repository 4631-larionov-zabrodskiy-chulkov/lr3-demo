import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.array([-4, -3, -2, 0, -1, 2, 3, 4])
y = np.array([-109, -54, -18, 3, -2, -3, 2, 19])
n = len(x)

x_ = np.sum(x)/n
y_ = np.sum(y)/n


A = np.array([
    [1, x_, np.sum(x**2)/n, np.sum(x**3)/n],
    [np.sum(x)/n, np.sum(x**2)/n, np.sum(x**3)/n, np.sum(x**4)/n],
    [np.sum(x**2)/n, np.sum(x**3)/n, np.sum(x**4)/n, np.sum(x**5)/n],
    [np.sum(x**3)/n, np.sum(x**4)/n, np.sum(x**5)/n, np.sum(x**6)/n]
])
C = np.array([y_, np.sum(x*y)/n, np.sum(x*x*y)/n, np.sum(x*x*x*y)/n])
B = np.linalg.solve(A,C)

print("Вариант 22")
print("B[0] = %.3f\nB[1] = %.3f\nB[2] = %.3f\nB[3] = %.3f"%(B[0], B[1], B[2], B[3]))
print("\ty = %.2f x^3 + %.2f x^2 + %.2f x + %.2f " % (B[3], B[2], B[1], B[0]))

xplot=np.arange(np.min(x) - 1,np.max(x) + 1,0.01)
plt.plot(xplot,B[3] * xplot**3 + B[2] * xplot**2 + B[1] * xplot + B[0]) 

y_emp = B[3] * x**3 + B[2] * x**2 + B[1] * x + B[0] 
Rxy = np.sum((y_emp - y_)**2)/np.sum((y - y_)**2) 

F = (Rxy * (n-2))/(1 - Rxy) 
print("\nF = %.2f" % F)

F_k = 5.99 
print(" F = %.2f" % F_k)
if F > F_k:
    print("  F > F=>")
else:
    print("  F < F=>")

S2 = np.dot((y - y_emp).T ,(y - y_emp))/(n-3-1) 
X = np.array([x**0, x, x**2, x**3])
covMatr = S2*np.linalg.inv(np.dot(X, X.T)) 

Sb = np.ones(4) 
for i in range(4):
    Sb[i] = np.sqrt(covMatr[i][i])

tB =np.array([abs(B[0])/Sb[0], abs(B[1])/Sb[1], abs(B[2])/Sb[2], abs(B[3])/Sb[3]]) 
t_k = stats.t.ppf(1 - 0.05/2, 6)
print("\nt[0] = %.2f\tt[1] = %.2f\tt[2] = %.2f\tt[3] = %.2f\n" %(tB[0], tB[1], tB[2], tB[3]))
print("t_k = %.2f" %t_k)

for i in range(4):
    if tB[i] < t_k:
        B[i] = 0
print("\nb[0] = %.2f\tb[1] = %.2f\tb[2] = %.2f\tb[3] = %.2f"%(B[0], B[1], B[2], B[3]))


y_emp_new = B[3] * x**3 + B[2] * x**2 + B[1] * x + B[0] 
Rxy_new = np.sum((y_emp_new - y_)**2)/np.sum((y - y_)**2) 

F_new = (Rxy_new * (n-2))/(1 - Rxy_new) 
print("\n F = %.2f" % F_new)
print(" F = %.2f" % F_k)
if F_new > F_k:
    print("  F > F")
else:
    print("  F < F")

xplot=np.arange(np.min(x) - 1,np.max(x) + 1,0.01)
plt.plot(xplot,B[3] * xplot**3 + B[2] * xplot**2 + B[1] * xplot + B[0]) 
plt.plot(x, y , 'go')
plt.xlabel(r'$x$') 
plt.ylabel(r'$f(x)$')
plt.title(r'$y={}x^3   {}x^2$'.format(round(B[3], 2), round(B[2], 2)))
plt.grid(True) 
plt.show() 
