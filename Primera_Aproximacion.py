from formulas_tp import *
import matplotlib.pyplot as plt

F = 0 # Suponiendo una fuerza constante a lo largo de toda la recta

def ecuacion_recta(tiempo, posicion, velocidad):
    if F > FUERZA_MAX:
        return 0
    return F / M # Retorna la aceleración (Fuerza / Masa)

y0 = 0.0 # Posición inicial en m
u0x = 40.0 # Velocidad inicial en m/s en el eje X
t0 = 0.0 # Tiempo inicial
tf = 1.6 # Tiempo final (esto implica en cuanto tiempo quiero que recorra la recta)
h = 0.05 # Paso de integración
f_u = ecuacion_recta # Es la ecuacion de F / M
t_x, y_x, u_x = runge_kutta_4_orden_superior(f_u, t0, y0, u0x, tf, h) # En el eje X

u0y = 2.5 # Velocidad inicial en m/s en el eje Y
t_y, y_y, u_y = runge_kutta_4_orden_superior(f_u, t0, y0, u0y, tf, h) # En el eje Y

#Graficar
plt.figure(figsize=(12, 6))

plt.subplot(2,2,1)
plt.title("Trayectoria completa - Posición (eje X)")
plt.plot(t_x, y_x, marker='o', markersize=2, label='Recta')
plt.ylabel("Posición en el eje X (m)")
plt.grid()
plt.legend()

plt.subplot(2,2,2)
plt.title("Trayectoria completa - Posición (eje Y)")
plt.plot(t_y, y_y, marker='o', markersize=2, label='Recta')
plt.ylabel("Posición en el eje Y (m)")
plt.grid()
plt.legend()

plt.subplot(2,2,3)
plt.title("Velocidad en función del tiempo (eje X)")
plt.plot(t_x, u_x, label='Velocidad')
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad en el eje X (m/s)")
plt.grid()

plt.subplot(2,2,4)
plt.title("Velocidad en función del tiempo (eje Y)")
plt.plot(t_y, u_y, label='Velocidad')
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad en el eje Y (m/s)")
plt.grid()

plt.tight_layout()
plt.show()
