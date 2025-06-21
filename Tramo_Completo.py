from formulas_tp import *
import numpy as np
import matplotlib.pyplot as plt
# Constantes físicas
g = 9.81
g_max = 6 * g
M = 800  # masa (kg)

# Paso temporal
dt = 0.01  # segundos

# Funciones de dinámica

def tramo_recto(t, estado, F):
    x, y, v, theta = estado
    a = F / M  # aceleración longitudinal
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dv = a
    dtheta = 0
    return np.array([dx, dy, dv, dtheta])

def tramo_curva(t, estado, radio, v_objetivo):
    x, y, v, theta = estado
    a_lat_max = g_max
    a_lat = v**2 / radio
    if a_lat > a_lat_max:
        v = np.sqrt(a_lat_max * radio)
        a_lat = a_lat_max
    omega = v / radio
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dv = 0
    dtheta = omega
    return np.array([dx, dy, dv, dtheta])

def rk4(f, t, estado, h, *args):
    k1 = f(t, estado, *args)
    k2 = f(t + h/2, estado + h*k1/2, *args)
    k3 = f(t + h/2, estado + h*k2/2, *args)
    k4 = f(t + h, estado + h*k3, *args)
    return estado + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def simular_tramo_recto(estado_inicial, distancia_objetivo, F, dt):
    estado = estado_inicial.copy()
    x_inicio, y_inicio = estado[0], estado[1]
    xs, ys = [], []
    distancia_recorrida = 0.0

    while distancia_recorrida < distancia_objetivo:
        estado = rk4(tramo_recto, 0, estado, dt, F)
        xs.append(estado[0])
        ys.append(estado[1])
        distancia_recorrida = np.sqrt((estado[0]-x_inicio)**2 + (estado[1]-y_inicio)**2)

    return estado, xs, ys

def simular_tramo_curva(estado_inicial, radio, angulo_objetivo, dt):
    estado = estado_inicial.copy()
    theta_inicio = estado[3]
    xs, ys = [], []

    angulo_girado = 0.0
    while angulo_girado < angulo_objetivo:
        estado = rk4(tramo_curva, 0, estado, dt, radio, estado[2])
        xs.append(estado[0])
        ys.append(estado[1])
        angulo_girado = abs(estado[3] - theta_inicio)

    return estado, xs, ys

# Parámetros físicos y fuerzas
F_max = g_max * M

# Datos deseados para la primer recta
x_ini, y_ini = 2, 14
x_fin, y_fin = 66, 10

dx = x_fin - x_ini
dy = y_fin - y_ini
theta_1 = np.arctan2(dy, dx)
dist_1 = np.sqrt(dx**2 + dy**2)

# Distancias y curvas para la trayectoria completa
distancias_rectas = [dist_1, 15, 35]  # el primero calculado para llegar a (66,10)
curvas = [
    (10, np.pi/2),  # radio 10 m, giro 90°
    (15, np.pi/4)    # radio 15 m, giro 45°
]

# Estado inicial con ángulo correcto
estado = np.array([x_ini, y_ini, 20.0, theta_1])

xs_total = []
ys_total = []

# 1) Primera recta
estado, xs, ys = simular_tramo_recto(estado, distancias_rectas[0], F_max, dt)
xs_total += xs
ys_total += ys

# 2) Primera curva
estado, xs, ys = simular_tramo_curva(estado, curvas[0][0], curvas[0][1], dt)
xs_total += xs
ys_total += ys

# 3) Segunda recta
estado, xs, ys = simular_tramo_recto(estado, distancias_rectas[1], F_max, dt)
xs_total += xs
ys_total += ys

# 4) Segunda curva
estado, xs, ys = simular_tramo_curva(estado, curvas[1][0], curvas[1][1], dt)
xs_total += xs
ys_total += ys

# 5) Tercer recta
estado, xs, ys = simular_tramo_recto(estado, distancias_rectas[2], F_max, dt)
xs_total += xs
ys_total += ys

# Graficar trayectoria
plt.figure(figsize=(10,6))
plt.plot(xs_total, ys_total, label="Trayectoria")
plt.axis('equal')
plt.grid(True)
plt.title("Trayectoria: recta - curva - recta - curva - recta")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.show()
