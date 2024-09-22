import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def Lagrange(X,i):
    x = sp.Symbol("x")
    L = 1
    for j in range(len(X)):
        if j != i:
            L *= (x-X[j])/(X[i]-X[j])
    return L 

def polinomio(X,Y):
    pol = 0
    for i in range(0,len(X)):
        pol += Y[i]*Lagrange(X,i)
    return pol

def integral_a_b():
    a = sp.Symbol("a")
    b = sp.Symbol("b")
    f = sp.Function("f")
    h = (b-a)/3
    
    X = [a,a+h,a+2*h,a+3*h]
    Y = [f(a),f(a+h),f(a+2*h),f(a+3*h)]
    
    for i in range(len(X)):
        print("Función cardinal #" + str(i+1) + ":")
        print(sp.factor(Lagrange(X,i)))
    
    p = polinomio(X,Y)
    p = sp.integrate(p,sp.Symbol("x"))
    int = p.subs(sp.Symbol("x"),b) - p.subs(sp.Symbol("x"),a) 
    print("")
    print("Regla de Simpson 3/8: ", sp.factor(int))
    
    
integral_a_b()
