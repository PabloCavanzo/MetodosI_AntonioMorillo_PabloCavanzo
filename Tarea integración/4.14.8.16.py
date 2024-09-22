import numpy as np
import matplotlib.pyplot as plt

roots, weights = np.polynomial.legendre.leggauss(15)

def funcion(x):
    return 1 / (1 + x**4)
def funcion2(x):
    return (x**2) / (1 + x**4)

t = 0.5 * (roots + 1)
weights *= 0.5

integral1 = np.sum(weights * funcion(t))
integral2 = np.sum(weights * funcion2(t))

print(integral1 + integral2)
