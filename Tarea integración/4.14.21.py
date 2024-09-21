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

def polinomio_n(n):
    x = sp.Symbol("x")
    f1 = (x ** 2 - 1) ** n
    c = 1 / (2 ** n * math.factorial(n))
    df1 = sp.diff(f1, x, n)
    f = sp.simplify(c * df1)
    
    def evaluar(xi):
        return float(f.subs("x", xi).evalf())
    
    return evaluar, f

def primeros_n_polinomios(n):
    poly = []
    for i in range(n):
        poly.append(polinomio_n(i)[1])
    return poly

def coeficiente(f, Weights, Roots, n):
    p_eval = polinomio_n(n)[0]
    
    t1 = 0.5*(Roots + 1)
    t2 = 0.5*(Roots - 1)
    
    a1 = np.array([f(x) for x in t1])
    b1 = np.array([p_eval(x) for x in t1])
    
    a2 = np.array([f(x) for x in t2])
    b2 = np.array([p_eval(x) for x in t2])
    
    integral0_1 = np.sum(Weights * a1 * b1)
    integral1_0 = np.sum(Weights * a2 * b2)
    
    return (n+0.5)*(integral0_1 + integral1_0)/2

def comb_lineal(f,n):
    p = 0
    roots, weights = np.polynomial.legendre.leggauss(15)
    
    for i in range(n):
       p += coeficiente(f,weights,roots,i) * polinomio_n(i)[1]
       
    return p


sgn_vectorized = np.vectorize(sgn)
X = np.linspace(-1, 1, 100)
Y = sgn_vectorized(X)

fig, ax = plt.subplots()
f_num = sp.lambdify(sp.Symbol("x"),comb_lineal(sgn,20),"numpy")
Y2 = f_num(X)
ax.scatter(X, Y,label="sgn(x)")
ax.plot(X,Y2,color="r")
plt.legend()
plt.show()