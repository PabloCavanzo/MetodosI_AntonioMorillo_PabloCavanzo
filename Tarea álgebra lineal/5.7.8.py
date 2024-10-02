import numpy as np

def factor_LU(M):
    M = M.astype(float)
    n = M.shape[0]
    L = np.eye(n)
    
    for j in range(0,n-1):
        for i in range(j+1,n):
            k = M[i][j]/M[j][j]
            M[i] = M[i] - k*M[j]
            
            L[i][j] = k 
        
    return M,L

A = np.array([[4.,-2.,1.],[20.,-7.,12.],[-8.,13., 21.]])
print("Matriz A:\n",A,"\n\nMatriz L:\n",factor_LU(A)[1],"\n\nMatriz U:\n",factor_LU(A)[0])