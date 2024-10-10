import numpy as np

class SistemaLineal:
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def Jacobi(self, x, itmax=1000, tolerancia=1e-16):
        for it in range(itmax):
            x_new = np.copy(x)     
            for i in range(self.A.shape[0]):
                suma = 0
                for j in range(self.A.shape[1]):
                    if i != j:
                        suma += self.A[i][j] * x[j]
                
                x_new[i] = (self.b[i] - suma) / self.A[i][i]
                        
            if np.linalg.norm(np.dot(self.A, x_new) - self.b) < tolerancia:
                break

            x = np.copy(x_new)
            
        return x, it

    def GaussSeidel(self, x, itmax=1000, tolerancia=1e-16):
        for it in range(itmax):
            x_new = np.copy(x)
            for i in range(self.A.shape[0]):
                suma = 0
                for j in range(self.A.shape[1]):
                    if i != j:
                        suma += self.A[i][j] * x_new[j]
                
                x_new[i] = (self.b[i] - suma) / self.A[i][i]
                        
            if np.linalg.norm(np.dot(self.A, x_new) - self.b) < tolerancia:
                break

            x = np.copy(x_new)
            
        return x, it

A = np.array([[3, -1, -1],
              [-1, 3, 1],
              [2, 1, 4]], dtype=float)
b = np.array([1., 3., 7.])
x0 = np.array([0., 0., 0.])

sistema = SistemaLineal(A, b)
xj, itj = sistema.Jacobi(x0)
xg, itg = sistema.GaussSeidel(x0)
print("Solución Jacobi:       " + str(xj) + " convirgió en " + str(itj) + " iteraciones.")
print("Solución Gauss-Seidel: " + str(xg) + " convirgió en " + str(itg) + " iteraciones.")
