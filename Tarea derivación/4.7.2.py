import matplotlib.pyplot as plt
import numpy as np


def funcion(x):
    return 1/np.sqrt(1+(np.exp(-(x**2))))

def derivada_exacta(x):
    return (x*np.exp(-(x**2)))/(1+np.exp(-(x**2)))**(3/2)

def derivada(f,x,h):
    return (f(x+h)-f(x))/h

def error(x):
    e = np.abs(funcion(x)-derivada_exacta(x))
    return e


fig, ax = plt.subplots(nrows=3)

fx = np.linspace(-10,10,100)

ax[0].scatter(fx,funcion(fx))
ax[1].scatter(fx,derivada(funcion,fx,0.05))
ax[2].scatter(fx,error(fx),color='r',marker="X")
print(error(fx))

plt.show()