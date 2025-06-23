import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread("pista.png")  # O el nombre del archivo real


# Constantes
g = 9.81
g_max = 6 * g
M = 800
dt = 0.01
F_max = g_max * M

# === Utilidades ===
def vector(p1, p2):
    return np.array([p2[0] - p1[0], p2[1] - p1[1]])

def angulo_entre(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot)

def rk4(f, t, estado, h, *args):
    k1 = f(t, estado, *args)
    k2 = f(t + h/2, estado + h*k1/2, *args)
    k3 = f(t + h/2, estado + h*k2/2, *args)
    k4 = f(t + h, estado + h*k3, *args)
    return estado + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# === Dinámica ===
def tramo_recto(t, estado, F):
    x, y, v, theta = estado
    v_ref = 60
    F_real = F * np.exp(-v / v_ref)
    F_real = np.clip(F_real, -F_max, F_max)
    a = F_real / M
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dv = a
    dtheta = 0
    return np.array([dx, dy, dv, dtheta])

def tramo_curva(t, estado, radio, v_objetivo):
    x, y, v, theta = estado
    a_lat = v**2 / radio
    if a_lat > g_max:
        v = np.sqrt(g_max * radio)
    omega = v / radio
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dv = 0
    dtheta = omega
    return np.array([dx, dy, dv, dtheta])

# === Simulaciones ===
def simular_tramo_recto(estado_inicial, distancia_objetivo, F, dt, t_inicial):
    estado = estado_inicial.copy()
    x0, y0 = estado[0], estado[1]
    xs, ys, velocidades, aceleraciones, tiempos, fuerzas = [], [], [], [], [], []
    distancia = 0
    t = t_inicial

    while distancia < distancia_objetivo:
        v = estado[2]
        F_real = F * np.exp(-v / 60)
        F_real = np.clip(F_real, -F_max, F_max)

        estado = rk4(tramo_recto, t, estado, dt, F)
        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])
        aceleraciones.append(F_real / M)
        tiempos.append(t)
        fuerzas.append(F_real)
        distancia = np.hypot(estado[0]-x0, estado[1]-y0)
        t += dt

    return estado, xs, ys, velocidades, aceleraciones, tiempos, t, fuerzas

def simular_tramo_curva(estado_inicial, radio, angulo_objetivo, dt, t_inicial):
    estado = estado_inicial.copy()
    theta0 = estado[3]
    xs, ys, velocidades, aceleraciones, tiempos = [], [], [], [], []
    angulo = 0
    t = t_inicial

    while angulo < angulo_objetivo:
        estado = rk4(tramo_curva, t, estado, dt, radio, estado[2])
        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])
        aceleraciones.append((estado[2]**2) / radio)
        tiempos.append(t)
        angulo = abs(estado[3] - theta0)
        t += dt

    return estado, xs, ys, velocidades, aceleraciones, tiempos, t

# === Puntos reales desde gráfico ===
x_ini, y_ini = 2, 14
x_fin, y_fin = 79.30, 10.30
x_ini2, y_ini2 = 88, 19
x_fin2, y_fin2 = 90, 48
x_ini3, y_ini3 = 89, 51
x_fin3, y_fin3 = 36, 74

# Vectores entre tramos
v1 = vector((x_ini, y_ini), (x_fin, y_fin))
v2 = vector((x_ini2, y_ini2), (x_fin2, y_fin2))
v3 = vector((x_ini3, y_ini3), (x_fin3, y_fin3))

# Ángulos de giro reales entre tramos
theta1 = angulo_entre(v1, v2)
theta2 = angulo_entre(v2, v3)

# Distancias rectas
dist_1 = np.linalg.norm(v1)
dist_2 = np.linalg.norm(v2)

# === Simulación ===
estado = np.array([x_ini, y_ini, 50.0, np.arctan2(y_fin - y_ini, x_fin - x_ini)])

xs_total, ys_total = [], []
vel_total, acc_total, tiempos_total, fuerzas_total = [], [], [], []
t_actual = 0.0

# Tramo recto 1
estado, xs, ys, vs, accs, ts, t_actual, fuerzas = simular_tramo_recto(estado, dist_1, 15000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs
tiempos_total += ts; fuerzas_total += fuerzas

# Curva 1 (radio 9)
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_curva(estado, 9, theta1, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs
tiempos_total += ts; fuerzas_total += [0]*len(ts)

# Corregir orientación tras curva 1
estado[3] = np.arctan2(y_fin2 - y_ini2, x_fin2 - x_ini2)

# Tramo recto 2
estado, xs, ys, vs, accs, ts, t_actual, fuerzas = simular_tramo_recto(estado, dist_2, 10000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs
tiempos_total += ts; fuerzas_total += fuerzas

# Curva 2 (radio 4)
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_curva(estado, 4, theta2, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs
tiempos_total += ts; fuerzas_total += [0]*len(ts)

# Corregir orientación hacia tramo final
estado[3] = np.arctan2(y_fin3 - estado[1], x_fin3 - estado[0])
distancia_final = np.linalg.norm([x_fin3 - estado[0], y_fin3 - estado[1]])

# Tramo recto final
estado, xs, ys, vs, accs, ts, t_actual, fuerzas = simular_tramo_recto(estado, distancia_final, 18000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs; acc_total += accs
tiempos_total += ts; fuerzas_total += fuerzas

# === Resultados ===
print(f"Tiempo total: {t_actual:.2f} s")
print(f"Punto final alcanzado: ({estado[0]:.2f}, {estado[1]:.2f})")

# === Gráficos ===
plt.figure(figsize=(10,6))
plt.plot(xs_total, ys_total, label="Trayectoria")
plt.scatter([x_ini, x_fin, x_ini2, x_fin2, x_ini3, x_fin3],
            [y_ini, y_fin, y_ini2, y_fin2, y_ini3, y_fin3],
            c='red', marker='x', label="Puntos de control")
plt.axis('equal')
plt.grid(True)
plt.title("Trayectoria: recta - curva - recta - curva - recta")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
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
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(tiempos_total, fuerzas_total, color='purple')
plt.title("Fuerza aplicada vs Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Fuerza (N)")
plt.grid(True)
plt.tight_layout()
plt.show()


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
