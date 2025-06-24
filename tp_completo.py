import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Imagen de la pista
img = mpimg.imread("pista.png")

# Constantes físicas
g = 9.81
g_max = 6 * g
M = 800
dt = 0.01
F_max = g_max * M

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
    velocidad_segura = min(v, np.sqrt(g_max * radio))
    omega = velocidad_segura / radio
    dx = velocidad_segura * np.cos(theta)
    dy = velocidad_segura * np.sin(theta)
    dv = 0
    dtheta = omega
    return np.array([dx, dy, dv, dtheta])

def simular_tramo_recto(estado_inicial, distancia_objetivo, F, dt, t_inicial):
    estado = estado_inicial.copy()
    x0, y0 = estado[0], estado[1]
    xs, ys, velocidades, aceleraciones, fuerzas = [], [], [], [], []
    tiempos, acc_tangencial_total, acc_centripeta_total = [], [], []
    distancia = 0
    t = t_inicial

    while distancia < distancia_objetivo:
        estado_anterior = estado.copy()
        estado = rk4(tramo_recto, t, estado, dt, F)

        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])

        acc_tangencial = (estado[2] - estado_anterior[2]) / dt
        acc_tangencial = np.clip(acc_tangencial, -g_max, g_max)
        acc_centripeta = 0

        aceleraciones.append(acc_tangencial)
        acc_tangencial_total.append(acc_tangencial)
        acc_centripeta_total.append(acc_centripeta)

        f_real = acc_tangencial * M
        f_real = np.clip(f_real, -F_max, F_max)
        fuerzas.append(f_real)
        tiempos.append(t)

        distancia = np.hypot(estado[0] - x0, estado[1] - y0)
        t += dt

    return estado, xs, ys, velocidades, aceleraciones, fuerzas, tiempos, t, acc_tangencial_total, acc_centripeta_total

def simular_tramo_curva(estado_inicial, radio, angulo_objetivo, dt, t_inicial):
    estado = estado_inicial.copy()
    theta0 = estado[3]
    xs, ys, velocidades, aceleraciones, fuerzas = [], [], [], [], []
    tiempos, acc_tangencial_total, acc_centripeta_total = [], [], []
    angulo = 0
    t = t_inicial

    while angulo < angulo_objetivo:
        estado = rk4(tramo_curva, t, estado, dt, radio, estado[2])
        xs.append(estado[0])
        ys.append(estado[1])
        velocidades.append(estado[2])

        acc_tangencial = 0
        velocidad_segura = min(estado[2], np.sqrt(g_max * radio))
        acc_centripeta = velocidad_segura ** 2 / radio

        aceleraciones.append(acc_centripeta)
        acc_tangencial_total.append(acc_tangencial)
        acc_centripeta_total.append(acc_centripeta)

        fuerzas.append(0)
        tiempos.append(t)
        angulo = abs(estado[3] - theta0)
        t += dt

    return estado, xs, ys, velocidades, aceleraciones, fuerzas, tiempos, t, acc_tangencial_total, acc_centripeta_total

# Puntos iniciales y finales de cada tramo 
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

# Inicialización
estado = np.array([x_ini, y_ini, 50.0, np.arctan2(y_fin - y_ini, x_fin - x_ini)])
t_actual = 0.0

# Sumadores de diferentes parametros 
xs_total, ys_total = [], []
vel_total, acc_total, fuerzas_total, tiempos_total = [], [], [], []
acc_tangencial_total, acc_centripeta_total = [], []
acc_rectas, acc_curvas = [], []
tiempos_rectas, tiempos_curvas = [], []

# Recta inicial
estado, xs, ys, vs, accs, fuerzas, ts, t_actual, acc_tangencial, acc_centripeta = simular_tramo_recto(estado, dist_1, 15000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
acc_tangencial_total += acc_tangencial; acc_centripeta_total += acc_centripeta
acc_rectas += accs; tiempos_rectas += ts

# Curva 1
estado, xs, ys, vs, accs, fuerzas, ts, t_actual, acc_tangencial, acc_centripeta = simular_tramo_curva(estado, 9, theta1, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
acc_tangencial_total += acc_tangencial; acc_centripeta_total += acc_centripeta
acc_curvas += accs; tiempos_curvas += ts
estado[3] = np.arctan2(y_fin2 - y_ini2, x_fin2 - x_ini2)

# Recta 2
estado, xs, ys, vs, accs, fuerzas, ts, t_actual, acc_tangencial, acc_centripeta = simular_tramo_recto(estado, dist_2, 10000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
acc_tangencial_total += acc_tangencial; acc_centripeta_total += acc_centripeta
acc_rectas += accs; tiempos_rectas += ts

# Curva 2
estado, xs, ys, vs, accs, fuerzas, ts, t_actual, acc_tangencial, acc_centripeta = simular_tramo_curva(estado, 4, theta2, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
acc_tangencial_total += acc_tangencial; acc_centripeta_total += acc_centripeta
acc_curvas += accs; tiempos_curvas += ts
estado[3] = np.arctan2(y_fin3 - estado[1], x_fin3 - estado[0])

distancia_final = np.linalg.norm([x_fin3 - estado[0], y_fin3 - estado[1]])

# Recta final
estado, xs, ys, vs, accs, fuerzas, ts, t_actual, acc_tangencial, acc_centripeta = simular_tramo_recto(estado, distancia_final, 18000, dt, t_actual)
xs_total += xs; ys_total += ys; vel_total += vs
acc_total += accs; fuerzas_total += fuerzas; tiempos_total += ts
acc_tangencial_total += acc_tangencial; acc_centripeta_total += acc_centripeta
acc_rectas += accs; tiempos_rectas += ts

# Imprimir tiempo total de simulación
print(f"\nTiempo total: {t_actual:.2f} s")

# Gráficos
fig, posicion_grafico = plt.subplots(2, 4, figsize=(24, 10)) 

# Trayectoria sin fondo
posicion_grafico[0, 0].plot(xs_total, ys_total, label="Trayectoria", color='black')
posicion_grafico[0, 0].set_title("Trayectoria")
posicion_grafico[0, 0].axis('equal')
posicion_grafico[0, 0].grid(True)
posicion_grafico[0, 0].set_xlabel("X (m)")
posicion_grafico[0, 0].set_ylabel("Y (m)")
posicion_grafico[0, 0].legend()

# Trayectoria con fondo
posicion_grafico[0, 1].imshow(img, extent=[0, 100, 0, 80], aspect='auto', zorder=0)
posicion_grafico[0, 1].plot(xs_total, ys_total, color='black', linewidth=2, zorder=1)
posicion_grafico[0, 1].set_title("Trayectoria sobre pista")
posicion_grafico[0, 1].axis('equal')
posicion_grafico[0, 1].grid(True)
posicion_grafico[0, 1].set_xlabel("X (m)")
posicion_grafico[0, 1].set_ylabel("Y (m)")

# Velocidad vs Tiempo
posicion_grafico[0, 2].plot(tiempos_total, vel_total, color='blue')
posicion_grafico[0, 2].set_title("Velocidad vs Tiempo")
posicion_grafico[0, 2].set_xlabel("Tiempo (s)")
posicion_grafico[0, 2].set_ylabel("Velocidad (m/s)")
posicion_grafico[0, 2].grid(True)

# Aceleración total
posicion_grafico[0, 3].plot(tiempos_total, acc_total, color='red')
posicion_grafico[0, 3].set_title("Aceleración total")
posicion_grafico[0, 3].set_xlabel("Tiempo (s)")
posicion_grafico[0, 3].set_ylabel("Aceleración (m/s²)")
posicion_grafico[0, 3].grid(True)

# Aceleración tangencial
posicion_grafico[1, 0].plot(tiempos_total, acc_tangencial_total, color='orange', label="Tangencial")
posicion_grafico[1, 0].axhline(y=g_max, color='red', linestyle='--', label='Límite 6g')
posicion_grafico[1, 0].axhline(y=-g_max, color='red', linestyle='--')
posicion_grafico[1, 0].set_title("Aceleración Tangencial")
posicion_grafico[1, 0].set_xlabel("Tiempo (s)")
posicion_grafico[1, 0].set_ylabel("aₜ (m/s²)")
posicion_grafico[1, 0].grid(True)
posicion_grafico[1, 0].legend()

# Aceleración centrípeta
posicion_grafico[1, 1].plot(tiempos_total, acc_centripeta_total, color='green', label="Centrípeta")
posicion_grafico[1, 1].axhline(y=g_max, color='red', linestyle='--', label='Límite 6g')
posicion_grafico[1, 1].set_title("Aceleración Centrípeta")
posicion_grafico[1, 1].set_xlabel("Tiempo (s)")
posicion_grafico[1, 1].set_ylabel("a꜀ (m/s²)")
posicion_grafico[1, 1].grid(True)
posicion_grafico[1, 1].legend()

# Aceleraciones por tramo
posicion_grafico[1, 2].plot(tiempos_rectas, acc_rectas, label="Rectas", color='blue')
posicion_grafico[1, 2].plot(tiempos_curvas, acc_curvas, label="Curvas", color='green')
posicion_grafico[1, 2].axhline(y=g_max, color='red', linestyle='--', label='Límite 6g')
posicion_grafico[1, 2].axhline(y=-g_max, color='red', linestyle='--')
posicion_grafico[1, 2].set_title("Aceleraciones por tramo")
posicion_grafico[1, 2].set_xlabel("Tiempo (s)")
posicion_grafico[1, 2].set_ylabel("Aceleración (m/s²)")
posicion_grafico[1, 2].grid(True)
posicion_grafico[1, 2].legend()

# Fuerza
posicion_grafico[1, 3].plot(tiempos_total, fuerzas_total, color='purple')
posicion_grafico[1, 3].set_title("Fuerza aplicada")
posicion_grafico[1, 3].set_xlabel("Tiempo (s)")
posicion_grafico[1, 3].set_ylabel("Fuerza (N)")
posicion_grafico[1, 3].grid(True)

plt.tight_layout()
plt.show()
