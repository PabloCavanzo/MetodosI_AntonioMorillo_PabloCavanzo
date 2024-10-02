#5.7.12
import numpy as np

def GetJacobian(f, x, h=1e-3):
    m = len(f)
    n = x.shape[0]
    J = np.zeros((m, n))
  
    for i in range(m):
        for j in range(n):
            rf = x.copy()
            rb = x.copy()
      
            rf[j] += h
            rb[j] -= h
            
            J[i, j] = (f[i](*rf) - f[i](*rb)) / (2 * h)
      
    return J

def evalf(f,x):
    n = len(f)
    vector = np.array([])
    
    for i in range(n):
        vector = np.append(vector,f[i](*x))
        
    return vector

def norma(f,x):
  return 0.5*np.linalg.norm(evalf(f,x))**2

def newton_generalizado(f,x,itmax=1000,error=1e-16):
    it = 0
    metric = 1
    
    while it < itmax and metric > error:        
        it += 1
        J_inv = np.linalg.inv(GetJacobian(f,x))
        F_eval = evalf(f,x)
        constant = np.dot(J_inv,F_eval)
        x = x - constant
        metric = norma(f,x)
        
    return x

Sistema_1=(lambda x1,x2: np.log(x1**2 + x2**2) - np.sin(x1*x2) - np.log(2) - np.log(np.pi),\
            lambda x1,x2: np.exp(x1-x2)+np.cos(x1*x2))
x0_1 = np.array([2.,2.])

Sistema_2=(lambda x1,x2,x3: 6*x1 - 2*np.cos(x2*x3) - 1,\
            lambda x1,x2,x3: 9*x2 + np.sqrt(x1**2 + np.sin(x3)+1.06) + 0.9,\
            lambda x1,x2,x3: 60*x3 + 3*np.exp(-x1*x2) + 10*np.pi - 3)
x0_2 = np.array([0.,0.,0.])

print("Vector solución sistema 1: ", newton_generalizado(Sistema_1,x0_1))
print("Vector solución sistema 2: ", newton_generalizado(Sistema_2,x0_2))