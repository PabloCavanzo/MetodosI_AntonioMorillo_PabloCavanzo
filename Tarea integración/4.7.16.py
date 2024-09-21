import matplotlib.pyplot as plt
import numpy as np

# Define el tamaño de la grilla
rows, cols = 10, 10

# Crea una lista de colores
colors = plt.cm.viridis(np.linspace(0, 1, rows * cols))

# Genera las coordenadas de los puntos en la grilla
x = np.repeat(np.arange(cols), rows)
y = np.tile(np.arange(rows), cols)

# Grafica los puntos con colores distintos
plt.scatter(x, y, color=colors)
plt.xlim(-1, cols)
plt.ylim(-1, rows)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
