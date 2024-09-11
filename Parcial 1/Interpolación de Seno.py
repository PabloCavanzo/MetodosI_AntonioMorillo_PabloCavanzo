import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math

#Interpolación
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
  return np.sin(x)
          
#Conjunto de soporte
X = np.array([0,np.pi/3,np.pi/2])
Y = funcion(X)
coords_x = np.linspace(0,X[-1],100)
coords_y = Interpolate(coords_x,X,Y)
y_real = funcion(coords_x)
error_y = np.abs(coords_y-y_real)

#Plot
fig, ax = plt.subplots()
ax.scatter(X,Y,color="#FF3A20",label="Puntos del soporte")
ax.plot(coords_x,y_real,color="red",label="sin(x)")
ax.plot(coords_x,coords_y,label="Polinomio")
ax.plot(coords_x,error_y,color="red",linestyle="--",label="Error de estimación")
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True)
ax.set_title("Gráfica ente 0 y π/2")
ax.legend()
plt.show()

#Cálculos error
def sin(x):
    return sp.sin(x)

def funcion_error(f,X,e):
    x = sp.Symbol("x")
    P = 1
    for i in range(0,len(X)):
        P *= (x-X[i])
        
    fun = f(x)
    dfun = sp.diff(fun,x,len(X)) / math.factorial(len(X))
    fun = dfun * P
    return np.abs(fun.subs("x",e).evalf())

x = sp.Symbol("x")
error = np.abs(funcion(np.pi/8)-Interpolate(np.pi/8,X,Y))
print("Polinomio calculado: ", sp.expand(Interpolate(x,X,Y)))
print("El error en la estimación al calcular sin(π/8) es:",error)
print("El error teórico al calcular sin(pi/8) es",funcion_error(sin,X,sp.pi/8),"\n")