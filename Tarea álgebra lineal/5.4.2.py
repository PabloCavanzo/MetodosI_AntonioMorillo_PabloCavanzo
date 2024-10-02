import numpy as np

def GaussSeidel(A,b,x, itmax = 100, tolerancia = 1e-16):
    for it in range(itmax):
        x_new = np.copy(x)
              
        for i in range(A.shape[0]):
            suma = 0
            for j in range(A.shape[1]):
                if i != j:
                    suma += A[i][j]*x_new[j]
            
            x_new[i] = (b[i] - suma)/A[i][i]
                    
        if np.linalg.norm(np.dot(A,x_new)-b) < tolerancia:
            break

        x = np.copy(x_new)
        
    return x,it

A = np.array([[3,-1,-1],[-1,3,1],[2,1,4]])
b = np.array([1.,3.,7.])
x0 = np.array([0.,0.,0.])

xg,itg = GaussSeidel(A,b,x0)