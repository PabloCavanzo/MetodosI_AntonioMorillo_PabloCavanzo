import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sympy as sp
import math

def sgn(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

X = np.linspace(-1,1,10)
sgn_vectorized = np.vectorize(sgn)
Y = sgn_vectorized(X)

fig,ax=plt.subplots()
ax.scatter(X,Y)
plt.show()

roots = np.polynomial.legendre.leggauss(15)

def polinomio_n(n):
    x = sp.Symbol("x")
    f1 = (x**2 - 1) **n
    c = 1/(2**n * math.factorial(n))
    df1 = sp.diff(f1,x,n)
    f = sp.expand(c*df1)
    
    def evaluar(xi):
        return f.subs("x",xi).evalf()
    
    return evaluar, f

def primeros_n_polinomios(n):
    poly = []
    for i in range(n):
        print("a")
        poly.append(polinomio_n(i)[1])
    
    return poly    

print(primeros_n_polinomios(20))