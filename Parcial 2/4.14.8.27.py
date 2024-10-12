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

def new_x(f, x, lr):
    n = x.size
    J = GetJacobian(f,x)
    G = evalf(f,x)
    x = x - lr * np.dot(J,G)
        
    return x,G

def descenso(f, x, lr=0.01,itmax=10000, error=1e-16):
    it = 0
    for i in range(itmax):
        it += 1
        x, G = new_x(f, x, lr)
        if np.linalg.norm(0.5*np.dot(G.T,G)) < 0.005:
            lr = 0.001
        if np.linalg.norm(0.5*np.dot(G.T,G)) < error:
            break
            
    return x, it
  
Sistema = (
    lambda w0, w1, w2, w3, x0, x1, x2, x3: w0 + w1 + w2 + w3 - 2,
    lambda w0, w1, w2, w3, x0, x1, x2, x3: w0*x0 + w1*x1 + w2*x2 + w3*x3,
    lambda w0, w1, w2, w3, x0, x1, x2, x3: w0*x0**2 + w1*x1**2 + w2*x2**2 + w3*x3**2 - 2/3,
    lambda w0, w1, w2, w3, x0, x1, x2, x3: w0*x0**3 + w1*x1**3 + w2*x2**3 + w3*x3**3,
    lambda w0, w1, w2, w3, x0, x1, x2, x3: w0*x0**4 + w1*x1**4 + w2*x2**4 + w3*x3**4 - 2/5,
    lambda w0, w1, w2, w3, x0, x1, x2, x3: w0*x0**5 + w1*x1**5 + w2*x2**5 + w3*x3**5,
    lambda w0, w1, w2, w3, x0, x1, x2, x3: w0*x0**6 + w1*x1**6 + w2*x2**6 + w3*x3**6 - 2/7,
    lambda w0, w1, w2, w3, x0, x1, x2, x3: w0*x0**7 + w1*x1**7 + w2*x2**7 + w3*x3**7
)

r0 = np.random.uniform(-1.,1.,size=8)
s = descenso(Sistema,r0)
pesos = s[0][:4]
raices = s[0][4:]
integral = sum(p * np.cos(x) for p, x in zip(pesos, raices))
print("Pesos:", pesos)
print("Raíces:", raices)
print(f"Estimación de la integral: {integral}")
