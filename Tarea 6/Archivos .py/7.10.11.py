import numpy as np
from scipy.stats import gamma, norm, uniform

N = 10**4
a = np.array([1,2,-1])

X1 = gamma.rvs(a=2,scale=3,size=N)
X2 = norm.rvs(loc=5,scale=2,size=N)
X3 = uniform.rvs(loc=0,scale=10,size=N)

X = a[0]*X1 + a[1]*X2 + a[2]*X3
m1 = np.mean(X)
m2 = np.var(X)
print(f"a)\nMomento 1:{m1}\nMomento 2:{m2}")

Xv = [X1,X2,X3]
m11 = a.T @ np.mean(Xv,axis=1).T
m22 = a.T @ np.cov(Xv) @ a
print(f"\nb)\nMomento 1:{m11}\nMomento 2:{m22}")

matriz_correlacion = np.corrcoef([X1,X2,X3])
print(f"\nc)\nCorrelación 1 y 2: {matriz_correlacion[0,1]}\nCorrelación 1 y 3: {matriz_correlacion[0,2]}\nCorrelación 2 y 3:{matriz_correlacion[1,2]}")