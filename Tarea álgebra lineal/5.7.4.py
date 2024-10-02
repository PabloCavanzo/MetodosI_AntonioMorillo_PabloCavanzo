import numpy as np

def mult_matrices(M1,M2):
    m = M1.shape[0]
    p = M2.shape[1]
    n = M1.shape[0]
    M = np.zeros((m,p))
    M = M.astype(float)
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                M[i][j] += M1[i][k] * M2[k][j] 
            
    return M

A = np.array([[1,0,0],
              [5,1,0],
              [-2,3,1]])

B = np.array([[4,-2,1],
              [0,3,7],
              [0,0,2]])

C = mult_matrices(A,B)
print("Matriz A:\n",A,"\n\nMatriz B:\n",B,"\n\nMatriz A*B:\n",C)