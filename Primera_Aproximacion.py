from formulas_tp import *
import numpy as np
import matplotlib.pyplot as plt

F = 1000 # Suponiendo una fuerza constante a lo largo de toda la recta

def ecuacion_recta(tiempo, posicion, velocidad):
    if F > FUERZA_MAX:
        return 0
    return F / M 

y0 = 0.0
u0 = 20.0 # Velocidad inicial en m/s
t0 = 0.0 # Tiempo inicial
tf = 1.0 # Tiempo final (esto implica en cuanto tiempo quiero que recorra la recta)
h = 0.1 # Paso de integración
f_u = ecuacion_recta # Es la ecuacion de F / M
t, y, u = runge_kutta_4_orden_superior(f_u, t0, y0, u0, tf, h)

#Graficar
plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.title("Trayectoria completa - Posición")
plt.plot(t, y, label='Recta')
plt.ylabel("Posición (m)")
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.title("Velocidad en función del tiempo")
plt.plot(t, u, label='Velocidad')
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.grid()

plt.tight_layout()
plt.show()
