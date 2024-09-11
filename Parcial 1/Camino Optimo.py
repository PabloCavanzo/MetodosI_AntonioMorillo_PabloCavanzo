import numpy as np
import matplotlib.pyplot as plt

def ct(x, T=(-3, 2), R=(2, -2), n0=1.00, n1=1.33):
    p1 = n0 * np.sqrt((x - T[0])**2 + T[1]**2)
    p2 = n1 * np.sqrt((x - R[0])**2 + R[1]**2)
    return p1 + p2

x = np.linspace(-5, 5, 100)
plt.plot(x, ct(x))
plt.xlabel('x')
plt.ylabel('ct(x)')
plt.title('Camino Óptico')
plt.grid(True)
plt.show()

print('Cualitativamente, se identifica el lugar donde se minimiza el tiempo de viaje del pulso como x = 0 aproximadamente')

def d1(f, x, h=0.01):
    return (f(x + h) - f(x - h)) / (2 * h)

def d2(df, x, h=0.01):
    return (df(x + h) - df(x - h)) / (2 * h)

def newton_raphson(f, df, xn, itmax=1000, precision=1e-6):
    error = 1
    it = 1
    while error > precision and it < itmax:
        try:
            xn1 = xn - (f(xn) / df(xn))
        except ZeroDivisionError:
            return False
        error = np.abs(xn1 - xn)
        xn = xn1
        it += 1
    if it == itmax:
        return False
    else:
        return xn

def get_roots(f, df, X, tol=7):
    roots = np.array([])
    for i in X:
        root = newton_raphson(f, df, i)
        if root is not False:
            root = round(root, tol)
            if root not in roots:
                roots = np.append(roots, root)
    return np.sort(roots)

def ct_func(x):
    return ct(np.array([x]))[0]

def d1_ct(x):
    return d1(ct_func, x)

def d2_ct(x):
    return d2(d1_ct, x)

x = np.linspace(-1, 1, 20)
roots = get_roots(d1_ct, d2_ct, x)
print("Raíces encontradas:", roots)
