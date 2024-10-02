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

Sistema=(lambda x1,x2,x3: 6*x1 - 2*np.cos(x2*x3) - 1,\
            lambda x1,x2,x3: 9*x2 + np.sqrt(x1**2 + np.sin(x3)+1.06) + 0.9,\
            lambda x1,x2,x3: 60*x3 + 3*np.exp(-x1*x2) + 10*np.pi - 3)
x0 = np.array([0.5,0.5,0.5])

print("Jacobiano de segundo orden h = 0.01: ", GetJacobian(Sistema,x0))
print("Jacobiano de cuarto orden h = 0.01", GetJacobian(Sistema,x0))
print("Jacobiano de segundo orden h = 0.0001: ", GetJacobian(Sistema,x0))