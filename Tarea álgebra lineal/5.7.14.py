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

def get_matrix(n):
    J1 = np.array([[0,0,0],
              [0,0,-1],
              [0,1,0]])
    J2 = np.array([[0,0,1],
              [0,0,0],
              [-1,0,0]])
    J3 = np.array([[0,-1,0],
              [1,0,0],
              [0,0,0]])

    if n == 1:
        return J1
    elif n == 2:
        return J2
    else:
        return J3
    
def conmutador(i,j):
    A,B =   get_matrix(i),get_matrix(j)
    
    P1 = mult_matrices(A,B)
    P2 = mult_matrices(B,A)
    return P1 - P2

def evaluar_k(Jk):
    if Jk[1][2] == 1 or Jk[1][2] == -1:
        k = 1
    elif Jk[0][2] == 1 or Jk[0][2] == -1:
        k = 2
    else:
        k = 3
        
    return k

def evaluar_epsilon(i,j,k):
    if i == j or i == k or k == j:
        eps = 0
    elif i == 1 and j == 2 and k == 3:
        eps = 1
    elif i == 2 and j == 3 and k == 1:
        eps = 1
    elif i == 3 and j == 1 and k == 2:
        eps = 1
    else:
        eps = -1
        
    return eps

i = int(input("Ingrese la i: "))
j = int(input("Ingrese la j: "))
conm = conmutador(i,j)
k = evaluar_k(conm)
eps = evaluar_epsilon(i,j,k)
print("\nk =", k)
print("\nConmutator:\n",conm)
print("\nResultado ϵ*Jk\n", eps * get_matrix(k))