from formulas_tp import *
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
r = 30  # radio de la curva en metros
v0 = 20  # velocidad lineal inicial en m/s (aprox. 72 km/h)
omega0 = v0 / r  # velocidad angular inicial

theta0 = 0.0  # radianes
t0 = 0.0      # tiempo inicial
tf = 5.0      # tiempo final
h = 0.01      # paso de integración
f_u = ecuacion_curvas  # función que define la curva

t, theta, omega = runge_kutta_4_orden_superior(
    f_u, t0, theta0, omega0, tf, h, r, ACELERACION_MAX
)

# Pasaje a grados 
theta_deg = np.degrees(theta)
omega_deg = np.degrees(omega)

# Graficar
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, theta_deg, label='Ángulo θ (°)')
plt.ylabel('θ (°)')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, omega_deg, label='Velocidad angular θ\' (°/s)', color='orange')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad angular (°/s)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()