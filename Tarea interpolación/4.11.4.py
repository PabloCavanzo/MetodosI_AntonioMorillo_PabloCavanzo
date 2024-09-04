import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

def Lagrange(x,X,i):

  L = 1
  for j in range(len(X)):
    if i != j:
      L *= (x-X[j])/(X[i]-X[j])
  return L

def funcion(x):
  return np.sqrt(x)

def Interpolate(x,X,Y):

  Poly = 0
  for i in range(len(X)):
    Poly += Y[i]*Lagrange(x,X,i)
  return Poly

x = sym.Symbol("x",real=True)

X2 = np.array([1.4,3.5,5.6])
Y2 = np.array([0.4007954931819738,0.594128102489774,0.29802795523938164])
x11 = np.linspace(0,6,100)
y11 = Interpolate(x11,X2,Y2)
y12 = x11*0.363970234266202
poly = Interpolate(x,X2,Y2)

print(sym.expand(poly))

print(np.sqrt(-9.8 / (2 * -9.8 * np.cos(20)**2)), np.cos(np.pi))

plt.scatter(X2,Y2)
plt.plot(x11,y11, color = "red")
plt.plot(x11,y12, color = "green")
plt.show()