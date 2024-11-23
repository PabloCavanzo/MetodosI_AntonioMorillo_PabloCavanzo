import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math

def random_sequence(size, inf_lim, sup_lim, x0, a=25214903917, m=2**48, c=11):
    sequence = np.zeros(size, dtype=float)
    for i in range(size):
        x0 = (a * x0 + c) % m
        sequence[i] = x0
    
    sequence /= float(m)
    return inf_lim + (sup_lim - inf_lim) * sequence

def correlation(k, size):
    sequence = random_sequence(size, 0, 1, rnd.random())
    sum = 0
    for i in range(size-k):  
        sum += sequence[i] * sequence[i + k]
    
    sum /= size  
    return sum

def function(x,alpha=2,beta=4):
    num = math.factorial(alpha + beta - 1) / (math.factorial(alpha-1) * math.factorial(beta-1))
    num *= x ** (alpha-1)
    num *= (1-x) ** (beta-1)
    return num

def integral(a, b, fun, n_points):
    x = random_sequence(n_points, a, b, rnd.random())
    y = random_sequence(n_points, 0, np.max(fun(x)), rnd.random())
    mask = y < fun(x)
    x = x[mask]
    y = y[mask]
    f_values = fun(x)
    
    integral = np.sum(f_values) / len(f_values)
    integral *= (b - a)
    
    return integral,x,y

inte,x,y = integral(0,1,function,10000)
x0 = np.linspace(0,1,50)
plt.plot(x0,function(x0))
plt.scatter(x,y,color="r")
plt.show()
print("Integral aproximada: ",inte)