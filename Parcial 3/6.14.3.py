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

def descenso(f, x, lr,itmax=10000, error=1e-16):
    it = 0
    for i in range(itmax):
        it += 1
        x, G = new_x(f, x, lr)
        if np.linalg.norm(0.5*np.dot(G.T,G)) < error:
            break
    return x, it


T2 = np.array([[0.1, 0.  , 0.02],
             [0.9, 0.05, 0.],
             [0. , 0.95, 0.98]])

Sistema2  = (lambda x1,x2,x3: T2[0,0]*x1 + T2[0,1]*x2 + T2[0,2]*x3 - x1,\
            lambda x1,x2,x3: T2[1,0]*x1 + T2[1,1]*x2 + T2[1,2]*x3 - x2,\
            lambda x1,x2,x3: x1 + x2 + x3 - 1)

x0 = np.array([1.,1.,1.])

estable_2 = descenso(Sistema2,x0,0.01)[0]
print("Estado estable: ", np.round(estable_2,4))

s1_y_s2 = estable_2[0] * estable_2[1]
s1_o_s2 = estable_2[0] + estable_2[1]
no_s1 = 1 - estable_2[0]

print(f"Probabilidad de ambas estaciones ocupadas: {round(s1_y_s2,5)*100}%.")
print(f"Probabilidad de una de las dos estaciones ocupadas: {round(s1_o_s2,4)*100}%.")
print(f"Probabilidad de que no haya nada en producción: {round(no_s1,5)*100}%.")

T2 = np.array([[0.1, 0.  , 0.02],
             [0.9, 0.05, 0.],
             [0. , 0.95, 0.98]])

