import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define los cuatro puntos en el espacio 3D
p1 = np.array([0, 0, 0])
p2 = np.array([1, 0, 0])
p3 = np.array([1, 1, 0])
p4 = np.array([0, 1, 0])

# Crear una figura y un eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear una colección de polígonos para el plano
faces = [[p1, p2, p3, p4]]  # Definimos la cara del plano
poly3d = Poly3DCollection(faces, alpha=0.5, facecolors='cyan')

# Añadir la colección al gráfico
ax.add_collection3d(poly3d)

# Configuración de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Establecer límites de los ejes para mejor visualización
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])

# Mostrar el gráfico
plt.show()
