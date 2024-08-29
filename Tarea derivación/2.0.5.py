import matplotlib.pyplot as plt
import numpy as np

def espiral():
    r = np.linspace(0, 2*np.pi,100)
    
    x = r * np.cos(r)
    y = r * np.sin(r)
    
    return x,y

fig, ax = plt.subplots()
r = espiral()[0]
th = espiral()[1]
ax.plot(r,th)
ax.set_title("Espiral de Arquímedes")
ax.grid(True)
plt.show()