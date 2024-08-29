import matplotlib.pyplot as plt
import numpy as np


def funcion(x):
    return 1/np.sqrt(1+(np.exp(-(x**2))))

def derivada_exacta(x):
    return (x*np.exp(-(x**2)))/(1+np.exp(-(x**2)))**(3/2)

def derivada(f,x,h):
    return (f(x+h)-f(x))/h

def error(x):
    e = np.abs(derivada(funcion,x,0.05)-derivada_exacta(x))
    return e


fig, ax = plt.subplots(nrows=3)
fx = np.linspace(-10,10,100)

ax[0].scatter(fx,funcion(fx))
ax[0].set_title("Función f(x)")
ax[0].grid(True)
ax[1].scatter(fx,derivada(funcion,fx,0.05))
ax[1].set_title("Derivada numérica de f(x)")
ax[1].grid(True)
ax[2].scatter(fx,error(fx),color='r',marker="X")
ax[2].set_title("Error")
ax[2].grid(True)

plt.tight_layout()
plt.show()