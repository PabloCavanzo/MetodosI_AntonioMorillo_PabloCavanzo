import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 25)
y = np.linspace(-4, 4, 25)
X,Y = np.meshgrid(x, y)

U=Y
V=-X

plt.quiver(X,Y,U,V)
plt.show()