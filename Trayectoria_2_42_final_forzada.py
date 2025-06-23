import numpy as np
import matplotlib.pyplot as plt

# Constantes físicas
g = 9.81
g_max = 6 * g
M = 800  # masa (kg)
dt = 0.01  # paso temporal (s)
F_max = g_max * M

# --- RECTAS con modelo más realista ---
def tramo_recto(t, estado, F):
    x, y, v, theta = estado

    # Aplicación más realista de fuerza: disminuye con la velocidad
    v_ref = 60  # velocidad de referencia para caída exponencial
    F_real = F * np.exp(-v / v_ref)
    F_real = np.clip(F_real, -F_max, F_max)

    a = F_real / M
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dv = a
    dtheta = 0
    return np.array([dx, dy, dv, dtheta])

# --- CURVAS ---
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

# --- Runge-Kutta 4 ---
def rk4(f, t, estado, h, *args):
    k1 = f(t, estado, *args)
    k2 = f(t + h/2, estado + h*k1/2, *args)
    k3 = f(t + h/2, estado + h*k2/2, *args)
    k4 = f(t + h, estado + h*k3, *args)
    return estado + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# --- Simulación de tramos rectos con fuerza guardada ---
def simular_tramo_recto(estado_inicial, distancia_objetivo, F, dt, t_inicial):
    estado = estado_inicial.copy()
    x_inicio, y_inicio = estado[0], estado[1]
    xs, ys, velocidades, aceleraciones, tiempos, fuerzas = [], [], [], [], [], []
    distancia_recorrida = 0.0
    t = t_inicial

    if F > F_max:
        F = F_max   

    while distancia_recorrida < distancia_objetivo:
        v = estado[2]
        F_real = F * np.exp(-v / 60)
        F_real = np.clip(F_real, -F_max, F_max)

        estado = rk4(tramo_recto, t, estado, dt, F)
        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])
        aceleraciones.append(F_real / M)
        fuerzas.append(F_real)
        tiempos.append(t)
        distancia_recorrida = np.sqrt((estado[0]-x_inicio)**2 + (estado[1]-y_inicio)**2)
        t += dt

    return estado, xs, ys, velocidades, aceleraciones, tiempos, t, fuerzas

# --- Simulación de curvas (sin fuerza longitudinal) ---
def simular_tramo_curva(estado_inicial, radio, angulo_objetivo, dt, t_inicial):
    estado = estado_inicial.copy()
    theta_inicio = estado[3]
    xs, ys, velocidades, aceleraciones, tiempos = [], [], [], [], []
    angulo_girado = 0.0
    t = t_inicial

    while angulo_girado < angulo_objetivo:
        estado = rk4(tramo_curva, t, estado, dt, radio, estado[2])
        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])
        aceleraciones.append((estado[2]**2) / radio)
        tiempos.append(t)
        angulo_girado = abs(estado[3] - theta_inicio)
        t += dt

    return estado, xs, ys, velocidades, aceleraciones, tiempos, t

# Coordenadas de cada tramo
x_ini, y_ini = 2, 14
x_fin, y_fin = 79, 10
x_ini2, y_ini2 = 88, 19
x_fin2, y_fin2 = 90, 48
x_ini3, y_ini3 = 89, 51
x_fin3, y_fin3 = 36, 74  # PUNTO OBJETIVO

# Cálculos geométricos
dx = x_fin - x_ini
dy = y_fin - y_ini
dx2 = x_fin2 - x_ini2
dy2 = y_fin2 - y_ini2
theta_1 = np.arctan2(dy, dx)
dist_1 = np.sqrt(dx**2 + dy**2)
dist_2 = np.sqrt(dx2**2 + dy2**2)
distancias_rectas = [dist_1, dist_2]
curvas = [(9,  np.pi/2), (4, np.pi/4)]

# Estado inicial: 180 km/h = 50 m/s
estado = np.array([x_ini, y_ini, 50.0, theta_1])

# Acumuladores
xs_total, ys_total = [], []
vel_total, acc_total, tiempos_total, fuerzas_total = [], [], [], []
t_actual = 0.0

# 1) Recta 1
estado, xs, ys, vs, accs, ts, t_actual, fuerzas = simular_tramo_recto(estado, dist_1, 15000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; tiempos_total += ts; fuerzas_total += fuerzas

# 2) Curva 1
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_curva(estado, *curvas[0], dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; tiempos_total += ts
fuerzas_total += [0]*len(ts)  # sin fuerza en curva

# 3) Recta 2
estado, xs, ys, vs, accs, ts, t_actual, fuerzas = simular_tramo_recto(estado, dist_2, 10000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; tiempos_total += ts; fuerzas_total += fuerzas

# 4) Curva 2
estado, xs, ys, vs, accs, ts, t_actual = simular_tramo_curva(estado, *curvas[1], dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; tiempos_total += ts
fuerzas_total += [0]*len(ts)

# 5) Recta final
objetivo = np.array([x_fin3, y_fin3])
estado[3] = np.arctan2(objetivo[1] - estado[1], objetivo[0] - estado[0])
distancia_final = np.linalg.norm(objetivo - estado[:2])
estado, xs, ys, vs, accs, ts, t_actual, fuerzas = simular_tramo_recto(estado, distancia_final, 18000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; tiempos_total += ts; fuerzas_total += fuerzas

# Salida
print(f"Tiempo total: {t_actual:.2f} s, Punto final alcanzado: ({estado[0]:.2f}, {estado[1]:.2f})")

# --- Gráficos ---
plt.figure(figsize=(10,6))
plt.plot(xs_total, ys_total, label="Trayectoria")
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

# --- Fuerza aplicada ---
plt.figure(figsize=(10, 4))
plt.plot(tiempos_total, fuerzas_total, color='purple')
plt.title("Fuerza aplicada vs Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Fuerza (N)")
plt.grid(True)
plt.tight_layout()
plt.show()
