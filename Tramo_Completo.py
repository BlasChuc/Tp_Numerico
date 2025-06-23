import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Cargar imagen de fondo
img = mpimg.imread("Captura de pantalla de 2025-06-2(1).png")

# Constantes físicas
g = 9.81
g_max = 6 * g
M = 800  # masa (kg)
h = 0.01  # paso temporal (s)

# Parámetros
F_max = g_max * M

# Dinámica en tramo recto
def tramo_recto(t, estado, F):
    x, y, v, theta = estado
    a = F / M
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dv = a
    dtheta = 0
    return np.array([dx, dy, dv, dtheta])

# Dinámica en curva
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

# Método de Runge-Kutta de orden 4
def rk4(f, t, estado, h, *args):
    k1 = f(t, estado, *args)
    k2 = f(t + h/2, estado + h*k1/2, *args)
    k3 = f(t + h/2, estado + h*k2/2, *args)
    k4 = f(t + h, estado + h*k3, *args)
    return estado + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# Simulación tramo recto con log de velocidad y aceleración
def simular_tramo_recto(estado_inicial, distancia_objetivo, F, h, t_inicial):
    estado = estado_inicial.copy()
    x_inicio, y_inicio = estado[0], estado[1]
    xs, ys, velocidades, aceleraciones, tiempos = [], [], [], [], []
    distancia_recorrida = 0.0
    t = t_inicial

    while distancia_recorrida < distancia_objetivo:
        estado = rk4(tramo_recto, t, estado, h, F)
        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])
        aceleraciones.append(F / M)
        tiempos.append(t)
        distancia_recorrida = np.sqrt((estado[0]-x_inicio)**2 + (estado[1]-y_inicio)**2)
        t += h

    return estado, xs, ys, velocidades, aceleraciones, tiempos, t

# Simulación curva con aceleración centrípeta
def simular_tramo_curva(estado_inicial, radio, angulo_objetivo, h, t_inicial):
    estado = estado_inicial.copy()
    theta_inicio = estado[3]
    xs, ys, velocidades, aceleraciones, tiempos = [], [], [], [], []
    angulo_girado = 0.0
    t = t_inicial

    while angulo_girado < angulo_objetivo:
        estado = rk4(tramo_curva, t, estado, h, radio, estado[2])
        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])
        aceleraciones.append((estado[2]**2) / radio)
        tiempos.append(t)
        angulo_girado = abs(estado[3] - theta_inicio)
        t += h

    return estado, xs, ys, velocidades, aceleraciones, tiempos, t

# Coordenadas iniciales y finales del primer tramo recto
x_ini, y_ini = 2, 14
x_fin, y_fin = 69, 9

dx = x_fin - x_ini
dy = y_fin - y_ini
theta_1 = np.arctan2(dy, dx)
dist_1 = np.sqrt(dx**2 + dy**2)

# Distancias de tramos rectos y curvas
distancias_rectas = [dist_1, 18, 49]
curvas = [
    (15, (np.pi)*2/5),
    (15, (np.pi)/2)
]

estado = np.array([x_ini, y_ini, 20.0, theta_1])

# Acumuladores
xs_total = []
ys_total = []
vel_total = []
acc_total = []
tiempos_total = []
t_actual = 0.0

# Simulación completa

# 1) Recta 1
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_recto(estado, distancias_rectas[0], F_max, h, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs; tiempos_total += ts

# 2) Curva 1
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_curva(estado, curvas[0][0], curvas[0][1], h, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs; tiempos_total += ts

# 3) Recta 2
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_recto(estado, distancias_rectas[1], F_max, h, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs; tiempos_total += ts

# 4) Curva 2
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_curva(estado, curvas[1][0], curvas[1][1], h, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs; tiempos_total += ts

# 5) Recta 3
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_recto(estado, distancias_rectas[2], F_max, h, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs; tiempos_total += ts

print(f"Tiempo total de la trayectoria: {t_actual:.2f} segundos")

# Gráfico 1: Trayectoria
""" plt.figure(figsize=(10,6))
plt.plot(xs_total, ys_total, label="Trayectoria")
plt.axis('equal')
plt.grid(True)
plt.title("Trayectoria: recta - curva - recta - curva - recta")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.show() """

# Gráficos 2: Velocidad y Aceleración
""" plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(tiempos_total, vel_total, color='blue')
plt.title("Velocidad vs Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (m/s)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tiempos_total, acc_total, color='red')
plt.title("Aceleración vs Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleración (m/s²)")
plt.grid(True) """

plt.figure(figsize=(10, 6))
plt.imshow(img, extent=[0, 100, 0, 80], aspect='auto', zorder=0)
plt.plot(xs_total, ys_total, label="Trayectoria", color='black', linewidth=2, zorder=1)
plt.axis('equal')
plt.grid(True)
plt.title("Trayectoria con fondo personalizado")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()


plt.tight_layout()
plt.show()

