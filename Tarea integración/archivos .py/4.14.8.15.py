import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math

def derivada(f,x,h=0.01):
    return (f(x+h)-f(x-h))/(2*h)

def polinomio_n(n):
    x = sp.Symbol("x")
    f1 = (x**2 - 1) **n
    c = 1/(2**n * math.factorial(n))
    df1 = sp.diff(f1,x,n)
    f = sp.expand(c*df1)
    def evaluar(xi):
        return f.subs("x",xi).evalf()
    
    return evaluar, f
        
def newton_raphson(f, df, xn, itmax=1000, precision=1e-6):
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

def get_roots(f,df,X,tol=5):
    roots=np.array([])
    for i in X:
        root = newton_raphson(f,df,i)
        if root != False:
            root = round(root,tol)
            if root not in roots:
                roots = np.append(roots,root)          
    return np.sort(roots)

def n_raices(n):
    X=np.linspace(-1,1,200)
    for i in range(1,n+1):
        print("Polinomio #" + str(i))
        x = get_roots(polinomio_n(i)[0],derivada,X)
        print(list(x))
        print("")

def GetWeights(Roots,Dpoly):
  Weights = np.array([])
  x = sp.Symbol('x',real=True)
  dpn = sp.lambdify([x],Dpoly,'numpy')
  for r in Roots:
    Weights = np.append(Weights,2/((1-r**2)*dpn(r)**2))

  return Weights

def n_pesos(n):
    X=np.linspace(-1,1,200)
    for i in range(1,n+1):
        Dpoly = sp.diff(polinomio_n(i)[1])
        Roots = get_roots(polinomio_n(i)[0],derivada,X)
        print("Polinomio #" + str(i))
        print(list(GetWeights(Roots,Dpoly)))
        print("")

# (a) Halle los ceros de los primeros 20 polinomios de Legendre.
n_raices(20)
# (b) Halle los pesos de ponderaci´on para los primeros 20 polinomios de Legendre.
n_pesos(20)