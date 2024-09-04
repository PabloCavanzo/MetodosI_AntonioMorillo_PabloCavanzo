import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

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

def funcion(x):
  return np.sqrt(x)

x = sym.Symbol("x",real=True)
X = np.array([100,121,144])
Y = np.array([10,11,12])
coords_x = np.linspace(0,X[-1],100)
coords_y = Interpolate(coords_x,X,Y)
y_ev = funcion(coords_x)

fig, ax = plt.subplots()
plt.plot(coords_x,y_ev,color="red",label="√x")
plt.plot(coords_x,coords_y,label="Polinomio")
plt.legend()
plt.show()

error = np.abs(funcion(114)-Interpolate(114,X,Y))
print("El error al calcular √114 es:",error)