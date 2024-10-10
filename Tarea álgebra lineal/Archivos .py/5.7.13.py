import numpy as np

def GetJacobian2(f, x, h):
    m = len(f)
    n = x.shape[0]
    J = np.zeros((m, n))
  
    for i in range(m):
        for j in range(n):
            rf = x.copy()
            rb = x.copy()
      
            rf[j] += h
            rb[j] -= h
            
            J[i][j] = (f[i](*rf) - f[i](*rb)) / (2 * h)
      
    return J

def GetJacobian4(f, x, h):
    m = len(f)
    n = x.shape[0]
    J = np.zeros((m, n))
  
    for i in range(m):
        for j in range(n):
            rf2 = x.copy()
            rf = x.copy()
            rb = x.copy()
            rb2 = x.copy()
      
            rf2[j] += 2 * h
            rf[j] += h
            rb[j] -= h
            rb2[j] -= 2 * h
            
            f_rf2 = f[i](*rf2)
            f_rf = f[i](*rf)
            f_rb = f[i](*rb)
            f_rb2 = f[i](*rb2)
            
            J[i, j] = (-f_rf2 + 8 * f_rf - 8 * f_rb + f_rb2) / (12 * h)
      
    return J

Sistema=(lambda x1,x2,x3: 6*x1 - 2*np.cos(x2*x3) - 1,\
            lambda x1,x2,x3: 9*x2 + np.sqrt(x1**2 + np.sin(x3)+1.06) + 0.9,\
            lambda x1,x2,x3: 60*x3 + 3*np.exp(-x1*x2) + 10*np.pi - 3)
x0 = np.array([0.5,0.5,0.5])

print("Jacobiano de segundo orden h = 0.01:\n", GetJacobian2(Sistema,x0,0.01))
print("\nJacobiano de cuarto orden h = 0.01:\n", GetJacobian4(Sistema,x0,0.01))
print("\nJacobiano de segundo orden igualado con h = 0.00001:\n", GetJacobian2(Sistema,x0,0.00001))