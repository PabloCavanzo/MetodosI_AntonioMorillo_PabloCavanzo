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

def Simpson(a,b,f):
    x = sp.Symbol("x")
    xm = (a+b)/2
    soporteX = np.array([a,xm,b])
    soporteY = f(soporteX)
    p = Interpolate(x,soporteX,soporteY)
    p = sp.integrate(p,x)
    
    return p.subs(x,b).evalf()-p.subs(x,a).evalf()

def Trapecio_compuesto(a,b,n,f):
    X = np.linspace(a,b,n)
    h = X[1]-X[0]
    s = 0
    for i in X[1:-1]:
        s += h*f(i)
    
    return (f(a)+f(b))*h/2 + s

a=0.01
simp = Simpson(-a,a,funcion)
tc = Trapecio_compuesto(-a,a,100,funcion)
real = np.pi*(0.5-np.sqrt(0.5**2 - 0.01**2))

print("Cálculo con regla de Simpson 1/3: ",simp)
print("Cálculo con Trapecio compuesto (100 puntos): ",tc)
print("Valor real: ", real)
print("Error simpson:", np.abs(simp-real)/real *100, "%")
print("Error Trapecio:", np.abs(tc-real)/real *100, "%")
