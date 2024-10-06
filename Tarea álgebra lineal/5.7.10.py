import numpy as np

def diagnoalizacion_Jacobi(A,itmax=1000,tol=1e-16):
    n = A.shape[0]
    
    for it in range(itmax):
        max = 0
        i_max = 0
        j_max = 0
    
        for i in range(n):
            for j in range(n):
                
                if i != j:
                    if np.abs(A[i][j]) > max:
                        max = np.abs(A[i][j])
                        i_max = i
                        j_max = j
    
        if max < tol:
            return A
        
        if A[j_max][j_max] == A[i_max][i_max]:
            theta = np.pi/4
        else:
            theta = 0.5 * np.arctan( 2*A[i_max][j_max] / (A[i_max][i_max]-A[j_max][j_max]) )
        
        J = np.eye(n)
        J[i_max][i_max] = np.cos(theta)
        J[j_max][j_max] = np.cos(theta)
        J[i_max][j_max] = -np.sin(theta)
        J[j_max][i_max] = np.sin(theta)    
        J_inv = np.linalg.inv(J)
    
        A = J_inv @ A @ J
                   
    return A

A = np.array([[4.,1.,1.],[1.,3.,2.],[1.,2.,5.]])
D = diagnoalizacion_Jacobi(A)

n = np.shape(A)[0]
valores = np.array([])
vectores = np.array([])

for i in range(n):
    for j in range(n):
        
        if np.abs(D[i][j]) < 1e-10:
            D[i][j] = 0
        if i == j:
            valores = np.sort(np.append(valores,D[i][j]))
            
print("Matriz A:\n",A,
      "\n\nMatriz Diagonal:\n",D,
      "\n\nValores propios:\n",valores,
      "\n\nVectores propios:\n",vectores,
      "\n\nValores propios numpy:\n", np.sort(np.linalg.eig(A)[0]),
      "\n\nVectores propios numpy:\n",  np.linalg.eig(A)[1])                  