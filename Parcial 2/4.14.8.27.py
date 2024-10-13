import numpy as np

def GetJacobian(f, x, h=0.001):
    m = len(f)
    n = x.shape[0]
    J = np.zeros((m, n))
  
    for i in range(m):
        for j in range(n):
            rf2 = x.copy()
            rf = x.copy()
            rb = x.copy()
            rb2 = x.copy()

            [1/(12*h), -2/(3*h), 0, 2/(3*h), -1/(12*h)]
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

def descenso(f, x, lr=0.01,itmax=1000, error=1e-6):
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

integrar = True
while integrar:
    try:
        r0 = np.random.uniform(-1., 1., size=8)
        s = descenso(Sistema, r0)
        pesos = s[0][:4]
        raices = s[0][4:]

        integral = sum(p * np.cos(x) for p, x in zip(pesos, raices))

        if 1.55 <= integral < 1.75:
            print("Pesos encontrados:", pesos)
            print("Raíces encontradas:", raices)
            print(f"Estimación de la integral: {integral} ≈ {round(integral,2)}")
            integrar = False
        else:
            print(f"Integral fuera de rango: {integral}. Intentando de nuevo...")

    except Exception as e:
        print(f"Ocurrió un error: {e}. Intentando de nuevo...")
