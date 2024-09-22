import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

def derivada(f,x,h=0.01):
    return (f(x+h)-f(x-h))/(2*h)

def polinomio_n(n):
    x = sp.Symbol("x")
    f1 = (x**n * sp.exp(-x))
    c = (sp.exp(x) / math.factorial(n))
    df1 = sp.diff(f1,x,n)
    f = sp.expand(c*df1)
    def evaluar(xi):
        return f.subs("x",xi).evalf()
    
    return evaluar, f
        
def newton_raphson(f, df, xn, itmax=1000, precision=1e-10):
    error = 1
    it = 1
    while error > precision and it < itmax:
        try:
            xn1 = xn - (f(xn)/df(f,xn))
        except ZeroDivisionError:
            return False
        error = np.abs(xn1 - xn)
        xn = xn1
        it += 1
    if it == itmax:
        return False
    else:
        return xn

def get_roots(f,df,X,tol=7):
    roots=np.array([])
    for i in X:
        root = newton_raphson(f,df,i)
        if root != False:
            root = round(root,tol)
            if root not in roots:
                roots = np.append(roots,root)          
    return np.sort(roots)
     
def GetWeights(n):
    X = np.linspace(0,50,200)
    Weights = np.array([])
    poly = polinomio_n(n)[0]   
    Roots = get_roots(poly,derivada,X)
    poly_cons = polinomio_n(n+1)[0]
    
    for r in Roots:
        Weights = np.append(Weights, r / ((n+1)**2 * poly_cons(r)**2))

    return Weights

def funcion(x):
    return np.exp(x) * (x**3 / (np.exp(x)-1))

def integral_n(n):
    roots, weights = np.polynomial.laguerre.laggauss(n)
    integral = np.sum(weights * funcion(roots))
    return integral

def error(estimado,real):
    return estimado/real

X = np.array([x for x in range(2,11)])
I = np.array([integral_n(x) for x in X])
Y = error(I, np.pi**4/15)
print(I)
plt.scatter(X,Y)
plt.show()
