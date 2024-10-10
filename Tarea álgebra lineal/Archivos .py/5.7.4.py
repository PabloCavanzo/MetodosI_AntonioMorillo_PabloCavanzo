import numpy as np

def mult_matrices(M1,M2):
    m = M1.shape[0]
    p = M2.shape[1]
    n = M1.shape[1]
    
    M = np.zeros((m,p))
    M = M.astype(float)
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                M[i][j] += M1[i][k] * M2[k][j]
            
    return M

def generar_matrices(m, n, p, min_val=-10, max_val=10):
    A = np.random.randint(min_val, max_val+1, size=(m, n))
    B = np.random.randint(min_val, max_val+1, size=(n, p))
    return A, B

A, B = np.array([[1,0,0],[5,1,0],[-2,3,1]]), np.array([[4,-2,1],[0,3,7],[0,0,2]])

C = mult_matrices(A,B)
print("Matriz A:\n",A,"\n\nMatriz B:\n",B,"\n\nMatriz A*B:\n",C,"\n\nResultado numpy:\n", A @ B)
