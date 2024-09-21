import numpy as np
import sympy as sp

def funcion(x):
    a = 0.01
    R = 0.5
    return np.sqrt(a**2 - x**2)/(R+x)

def Lagrange(x,X,i):
    L = 1
    for j in range(len(X)):
        if i != j:
            L *= (x-X[j])/(X[i]-X[j])
    return L

def Interpolate(x,X,Y):
    Poly = 0
    for i in range(len(X)):
        Poly += Y[i]*Lagrange(x,X,i)
    return Poly

def Simpson(a,b,f,h):
    x = sp.Symbol("x")
    xm = (a+b)/2
    soporteX = np.array([a,xm,b])
    soporteY = f(soporteX)
    p = Interpolate(x,soporteX,soporteY)
    print(sp.expand(p))
    ip = sp.integrate(p,x)
    print(ip)
    print(ip.subs(x,b).evalf()-ip.subs(x,a).evalf())
    
Simpson(-0.01,0.01,funcion,0.1)
