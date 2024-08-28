import matplotlib.pyplot as plt
import numpy as np

def carga_de_datos (ruta: str) -> dict:
    archivo = open(ruta, "r")
    linea = archivo.readline()
    X = []
    Y = [] 
    leer = True
    
    while leer:
        if linea != "":
            lista = linea.split("  ")
            x = float(lista[0])
            y = float(lista[1].replace("\n", ""))
            X.append(x)
            Y.append(y)
            linea = archivo.readline()
        else:
            leer = False
            
    return np.array(X), np.array(Y)

ruta = "/Users/armr1/OneDrive/Documents/Universidad/Tercer Semestre/Métodos Computacionales I/datos tarea 1.txt"

x = carga_de_datos(ruta)[0]
y = carga_de_datos(ruta)[1]
fig, ax = plt.subplots()
ax.plot(x,y)

def encontrar_maximos(x,y):
    px = np.empty(0)
    py = np.empty(0)
    
    for i in range(1,len(x)-1):
        if (y[i] - y[i-1] > 0) and (y[i] - y[i+1] > 0):
            px = np.append(px,x[i])
            py = np.append(py,y[i])
            
    return px, py

px = encontrar_maximos(x,y)[0]
py = encontrar_maximos(x,y)[1]
ax.scatter(px,py,color='red')

plt.show()