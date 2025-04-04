import numpy as np
import matplotlib.pyplot as plt

def cambio_pos_X(x_0, y_0, q1, q2, v_0, c, m, dt):
    r = np.sqrt(x_0**2 + y_0**2)
    F = c * q1 * q2 / r**2
    a_x = F * (x_0 / r) / m
    dv = dt * a_x
    dx = dt * (v_0 + dv)
    return dx, v_0 + dv

def cambio_pos_Y(x_0, y_0, q1, q2, v_0, c, m, dt):
    r = np.sqrt(x_0**2 + y_0**2)
    F = c * q1 * q2 / r**2
    a_y = F * (y_0 / r) / m
    dv = dt * a_y
    dy = dt * (v_0 + dv)
    return dy, v_0 + dv

def b(z, Z, K, theta):
    theta_rad = (theta / 2) * (np.pi / 180)
    K_eV = K / 1.602176634e-19
    b_0 = (z * Z / (2 * K_eV)) * 1.44 * (1 / np.tan(theta_rad)) * (10 ** -9)
    return b_0

def theta(z, Z, K, b):
    K_eV = K / 1.602176634e-19
    b *= (10 ** 9)
    arg = (z * Z * 1.44) / (2 * K_eV * b)
    return round(2 * np.arctan(arg) * 180 / np.pi,2)

z = 2
Z = int(input("Ingrese protones en el núcleo atómico: "))
q1 = z * 1.602176634e-19
q2 = Z * 1.602176634e-19
c = 8.99e9
m = 6.6e-27
b_0 = int(input("Ingrese el parámetro de impacto (fm):"))
x_pos = np.array([-1e-12])
y_pos = np.array([b_0 * (10 ** -15)])
r_coord = np.array([np.sqrt(x_pos[0]**2 + y_pos[0]**2)])
K = 5e6 * 1.602176634e-19
v_x = np.sqrt(2 * K / m)
v_y = 0.0
dt = 1e-22

for i in range(10000):
    dx, v_x = cambio_pos_X(x_pos[-1], y_pos[-1], q1, q2, v_x, c, m, dt)
    dy, v_y = cambio_pos_Y(x_pos[-1], y_pos[-1], q1, q2, v_y, c, m, dt)

    x_new = x_pos[-1] + dx
    y_new = y_pos[-1] + dy

    x_pos = np.append(x_pos, x_new)
    y_pos = np.append(y_pos, y_new)
    r_coord = np.append(r_coord, np.sqrt(x_new**2 + y_new**2))

    if np.sqrt(x_new**2 + y_new**2) > np.sqrt(1.1e-12**2 + 1.1e-12**2):
        break

indice_min = np.argmin(r_coord)
r_min = r_coord[indice_min]
angulo_deg = np.degrees(np.arctan2(v_y, v_x))

print(f"Número de protones en la partícula: {z}")
print(f"Número de protones en el núcleo atómico: {Z}")
print(f"\nÁngulo final (desde velocidades): {theta(z,Z,K,y_pos[0])}°")
print(f"Ángulo final (calculado): {angulo_deg:.2f}°")
print(f"\nPárametro de impacto: {y_pos[0]}")
print(f"Párametro de impacto calculado: {b(z,Z,K,angulo_deg):.4e}")
print(f"\nÍndice de mínimo r: {indice_min}")
print(f"Distancia mínima al núcleo: {r_min:.4e} m")

fig, ax = plt.subplots()
ax.plot(x_pos, y_pos, label='Trayectoria')
ax.plot(0, 0, 'ro', label='Núcleo', markersize=5)
ax.plot(x_pos[0], y_pos[0], 'ko', markersize=5)

l = int(len(x_pos)/10)
for i in range(1,11):
    ax.plot(x_pos[i*l - 5], y_pos[i*l - 5], 'ko', markersize=3)

ax.set_aspect('equal')
ax.set_title('Dispersión de Rutherford')
ax.set_xlabel('x [pm]')
ax.set_ylabel('y [pm]')
ax.grid(True)
ax.legend()
limite = 1.1e-12
ax.set_xlim(-limite, limite)
ax.set_ylim(-0.25e-12, limite)
plt.show()

plt.figure()
plt.plot(np.arange(len(r_coord)), r_coord, label='r vs índice')
plt.plot(indice_min, r_min, 'ro', label='Mínimo r')
plt.text(indice_min, r_min * 1.05, f"min r = {r_min:.2e} m\níndice = {indice_min}", 
         fontsize=8, ha='center', color='red')
plt.title('Distancia al núcleo vs índice de tiempo')
plt.xlabel('Índice')
plt.ylabel('r (m)')
plt.grid(True)
plt.legend()
plt.xlim(0, len(r_coord))
plt.show()
