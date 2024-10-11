import numpy as np

#a) Cargue N = 50 puntos y pesos de Gauss para calcular el campo gravitacional sobre el disco.
roots, weights = np.polynomial.legendre.leggauss(50)

#b) Defina la función de la Ecuación (4.208).
def campo (x,y,z):
    pass

#c) Defina una función para calcular la integral usando la forma de doble cuadratura (Ecuación (4.170)).
#d) Verifique que el campo gravitaci´on en el punto (0., 0., 0.2) es efectivamente g = −9.813646 m/s^2.
#e) Usando coordenadas polares:
#f) ¿Cómo interpreta que la gravedad no depende del ángulo sobre la tierra?
#g) ¿Qué valores tiene la gravedad en el ecuador R = 0.5 y en el borde R = 1?
#h) ¿Qué podría decirle a un amig@ terraplanista con base a sus resultados teóricos?
