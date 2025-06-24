import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Imagen de pista
img = mpimg.imread("pista.png")

# Constantes físicas
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
    a_real = np.clip(F_real / M, -g_max, g_max)
    F_real = a_real * M
    dx = v * np.cos(theta)
    dy = v * np.sin(theta)
    dv = a_real
    dtheta = 0
    return np.array([dx, dy, dv, dtheta])

def tramo_curva(t, estado, radio, _):
    x, y, v, theta = estado
    v_seguro = min(v, np.sqrt(g_max * radio))
    omega = v_seguro / radio
    dx = v_seguro * np.cos(theta)
    dy = v_seguro * np.sin(theta)
    dv = 0
    dtheta = omega
    return np.array([dx, dy, dv, dtheta])

# === Simulaciones ===
def simular_tramo_recto(estado_inicial, distancia_objetivo, F, dt, t_inicial):
    estado = estado_inicial.copy()
    x0, y0 = estado[0], estado[1]
    xs, ys, velocidades, aceleraciones, fuerzas = [], [], [], [], []
    tiempos, a_tan_total, a_cen_total = [], [], []
    distancia = 0
    t = t_inicial

    while distancia < distancia_objetivo:
        estado_anterior = estado.copy()
        estado = rk4(tramo_recto, t, estado, dt, F)

        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])

        a_tan = (estado[2] - estado_anterior[2]) / dt
        a_tan = np.clip(a_tan, -g_max, g_max)
        a_cen = 0

        aceleraciones.append(a_tan)
        a_tan_total.append(a_tan)
        a_cen_total.append(a_cen)

        f_real = a_tan * M
        f_real = np.clip(f_real, -F_max, F_max)
        fuerzas.append(f_real)
        tiempos.append(t)

        distancia = np.hypot(estado[0] - x0, estado[1] - y0)
        t += dt

    return estado, xs, ys, velocidades, aceleraciones, fuerzas, tiempos, t, a_tan_total, a_cen_total

def simular_tramo_curva(estado_inicial, radio, angulo_objetivo, dt, t_inicial):
    estado = estado_inicial.copy()
    theta0 = estado[3]
    xs, ys, velocidades, aceleraciones, fuerzas = [], [], [], [], []
    tiempos, a_tan_total, a_cen_total = [], [], []
    angulo = 0
    t = t_inicial

    while angulo < angulo_objetivo:
        estado = rk4(tramo_curva, t, estado, dt, radio, estado[2])
        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])

        a_tan = 0
        v_seguro = min(estado[2], np.sqrt(g_max * radio))
        a_cen = v_seguro ** 2 / radio

        aceleraciones.append(a_cen)
        a_tan_total.append(a_tan)
        a_cen_total.append(a_cen)

        fuerzas.append(0)
        tiempos.append(t)
        angulo = abs(estado[3] - theta0)
        t += dt

    return estado, xs, ys, velocidades, aceleraciones, fuerzas, tiempos, t, a_tan_total, a_cen_total

# === Puntos y vectores ===
x_ini, y_ini = 2, 14
x_fin, y_fin = 79.30, 10.30
x_ini2, y_ini2 = 88, 19
x_fin2, y_fin2 = 90, 48
x_ini3, y_ini3 = 89, 51
x_fin3, y_fin3 = 36, 74

v1 = vector((x_ini, y_ini), (x_fin, y_fin))
v2 = vector((x_ini2, y_ini2), (x_fin2, y_fin2))
v3 = vector((x_ini3, y_ini3), (x_fin3, y_fin3))

theta1 = angulo_entre(v1, v2)
theta2 = angulo_entre(v2, v3)

dist_1 = np.linalg.norm(v1)
dist_2 = np.linalg.norm(v2)

# === Inicialización ===
estado = np.array([x_ini, y_ini, 50.0, np.arctan2(y_fin - y_ini, x_fin - x_ini)])
t_actual = 0.0

# Acumuladores globales
xs_total, ys_total = [], []
vel_total, acc_total, fuerzas_total, tiempos_total = [], [], [], []
a_tan_total, a_cen_total = [], []

# Acumuladores separados
acc_rectas, acc_curvas = [], []
tiempos_rectas, tiempos_curvas = [], []

# === Tramos ===
estado, xs, ys, vs, accs, fuerzas, ts, t_actual, a_tan, a_cen = simular_tramo_recto(estado, dist_1, 15000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
a_tan_total += a_tan; a_cen_total += a_cen
acc_rectas += accs; tiempos_rectas += ts

estado, xs, ys, vs, accs, fuerzas, ts, t_actual, a_tan, a_cen = simular_tramo_curva(estado, 9, theta1, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
a_tan_total += a_tan; a_cen_total += a_cen
acc_curvas += accs; tiempos_curvas += ts

estado[3] = np.arctan2(y_fin2 - y_ini2, x_fin2 - x_ini2)

estado, xs, ys, vs, accs, fuerzas, ts, t_actual, a_tan, a_cen = simular_tramo_recto(estado, dist_2, 10000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
a_tan_total += a_tan; a_cen_total += a_cen
acc_rectas += accs; tiempos_rectas += ts

estado, xs, ys, vs, accs, fuerzas, ts, t_actual, a_tan, a_cen = simular_tramo_curva(estado, 4, theta2, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
a_tan_total += a_tan; a_cen_total += a_cen
acc_curvas += accs; tiempos_curvas += ts

estado[3] = np.arctan2(y_fin3 - estado[1], x_fin3 - estado[0])
distancia_final = np.linalg.norm([x_fin3 - estado[0], y_fin3 - estado[1]])
estado, xs, ys, vs, accs, fuerzas, ts, t_actual, a_tan, a_cen = simular_tramo_recto(estado, distancia_final, 18000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
a_tan_total += a_tan; a_cen_total += a_cen
acc_rectas += accs; tiempos_rectas += ts

# === Verificaciones ===
print(f"\n⏱️ Tiempo total: {t_actual:.2f} s")
print(f"📍 Punto final alcanzado: ({estado[0]:.2f}, {estado[1]:.2f})")

print(f"\n📈 Aceleración máxima en rectas: {np.max(np.abs(acc_rectas)):.2f} m/s²")
print(f"📈 Aceleración máxima en curvas: {np.max(np.abs(acc_curvas)):.2f} m/s²")

if np.max(np.abs(acc_rectas)) > g_max:
    print("⚠️ Las rectas superan el límite de aceleración.")
else:
    print("✅ Las aceleraciones en rectas están dentro del límite.")

if np.max(np.abs(acc_curvas)) > g_max:
    print("⚠️ Las curvas superan el límite de aceleración lateral.")
else:
    print("✅ Las aceleraciones en curvas están dentro del límite.")

# === Gráficos ===
# === TODOS LOS GRÁFICOS EN UNA SOLA VENTANA EN HORIZONTAL ===
fig, axs = plt.subplots(2, 4, figsize=(24, 10))  # 2 filas, 4 columnas

# Trayectoria sin fondo
axs[0, 0].plot(xs_total, ys_total, label="Trayectoria", color='black')
axs[0, 0].scatter([x_ini, x_fin, x_ini2, x_fin2, x_ini3, x_fin3],
                 [y_ini, y_fin, y_ini2, y_fin2, y_ini3, y_fin3],
                 c='red', marker='x', label="Puntos de control")
axs[0, 0].set_title("Trayectoria")
axs[0, 0].axis('equal')
axs[0, 0].grid(True)
axs[0, 0].set_xlabel("X (m)")
axs[0, 0].set_ylabel("Y (m)")
axs[0, 0].legend()

# Trayectoria con fondo
axs[0, 1].imshow(img, extent=[0, 100, 0, 80], aspect='auto', zorder=0)
axs[0, 1].plot(xs_total, ys_total, color='black', linewidth=2, zorder=1)
axs[0, 1].set_title("Trayectoria sobre pista")
axs[0, 1].axis('equal')
axs[0, 1].grid(True)
axs[0, 1].set_xlabel("X (m)")
axs[0, 1].set_ylabel("Y (m)")

# Velocidad vs Tiempo
axs[0, 2].plot(tiempos_total, vel_total, color='blue')
axs[0, 2].set_title("Velocidad vs Tiempo")
axs[0, 2].set_xlabel("Tiempo (s)")
axs[0, 2].set_ylabel("Velocidad (m/s)")
axs[0, 2].grid(True)

# Aceleración total
axs[0, 3].plot(tiempos_total, acc_total, color='red')
axs[0, 3].set_title("Aceleración total")
axs[0, 3].set_xlabel("Tiempo (s)")
axs[0, 3].set_ylabel("Aceleración (m/s²)")
axs[0, 3].grid(True)

# Aceleración tangencial
axs[1, 0].plot(tiempos_total, a_tan_total, color='orange', label="Tangencial")
axs[1, 0].axhline(y=g_max, color='red', linestyle='--', label='Límite 6g')
axs[1, 0].axhline(y=-g_max, color='red', linestyle='--')
axs[1, 0].set_title("Aceleración Tangencial")
axs[1, 0].set_xlabel("Tiempo (s)")
axs[1, 0].set_ylabel("aₜ (m/s²)")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Aceleración centrípeta
axs[1, 1].plot(tiempos_total, a_cen_total, color='green', label="Centrípeta")
axs[1, 1].axhline(y=g_max, color='red', linestyle='--', label='Límite 6g')
axs[1, 1].set_title("Aceleración Centrípeta")
axs[1, 1].set_xlabel("Tiempo (s)")
axs[1, 1].set_ylabel("a꜀ (m/s²)")
axs[1, 1].grid(True)
axs[1, 1].legend()

# Aceleraciones por tramo
axs[1, 2].plot(tiempos_rectas, acc_rectas, label="Rectas", color='blue')
axs[1, 2].plot(tiempos_curvas, acc_curvas, label="Curvas", color='green')
axs[1, 2].axhline(y=g_max, color='red', linestyle='--', label='Límite 6g')
axs[1, 2].axhline(y=-g_max, color='red', linestyle='--')
axs[1, 2].set_title("Aceleraciones por tramo")
axs[1, 2].set_xlabel("Tiempo (s)")
axs[1, 2].set_ylabel("Aceleración (m/s²)")
axs[1, 2].grid(True)
axs[1, 2].legend()

# Fuerza aplicada
axs[1, 3].plot(tiempos_total, fuerzas_total, color='purple')
axs[1, 3].set_title("Fuerza aplicada")
axs[1, 3].set_xlabel("Tiempo (s)")
axs[1, 3].set_ylabel("Fuerza (N)")
axs[1, 3].grid(True)

# Ajuste general
plt.tight_layout()
plt.show()
