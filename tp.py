import numpy as np

# Constantes físicas
G = 9.81
ACELERACION_MAX = 6 * G  # m/s²
M = 800  # kg
FUERZA_MAX = ACELERACION_MAX * M  # N

# ---------- Ecuaciones del modelo ----------

def ecuacion_recta(t, x, v, F):
    a = F / M
    if a > ACELERACION_MAX:
        a = ACELERACION_MAX
    elif a < -ACELERACION_MAX:
        a = -ACELERACION_MAX
    return a


def runge_kutta_cond_parada(f_u, y0, u0, h, condicion_limite, condicion_parada, *args):
    t = [0]
    y = [y0]
    u = [u0]
    i = 0
    while True:
        k1 = u[i]
        m1 = f_u(t[i], y[i], u[i], *args)
        k2 = u[i] + (h * m1) / 2
        m2 = f_u(t[i] + h/2, y[i] + (h * k1)/2, k2, *args)
        k3 = u[i] + (h * m2) / 2
        m3 = f_u(t[i] + h/2, y[i] + (h * k2)/2, k3, *args)
        k4 = u[i] + (h * m3)
        m4 = f_u(t[i] + h, y[i] + (h * k3), k4, *args)
        y_next = y[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        u_next = u[i] + h * (m1 + 2*m2 + 2*m3 + m4) / 6
        t_next = t[i] + h
        t.append(t_next)
        y.append(y_next)
        u.append(u_next)
        i += 1
        if y_next >= condicion_limite or i > 100000:
            break
    return t, y, u

def parar_por_posicion(pos_actual, pos_objetivo):
    return pos_actual >= pos_objetivo

def tiempo_curva(r, angulo_rad):
    if r < 1:
        return 1e6  # Penaliza radios no permitidos
    v_max = np.sqrt(ACELERACION_MAX * r)
    return abs(r * angulo_rad) / v_max

# ---------- Función principal ----------

def calcular_tiempo_total(F1, F2, F3, r1, r2, v_inicial=50):
    h = 0.01
    d1, d2, d3 = 60, 10, 40  # Longitudes de rectas (m)
    ang1, ang2 = np.pi / 2, -np.pi / 4  # Ángulos de giro en radianes

    try:
        # Recta 1
        t1, x1, v1 = runge_kutta_cond_parada(ecuacion_recta, 0.0, v_inicial, h, d1, parar_por_posicion, F1)
        v_antes_curva1 = v1[-1]
        tiempo_curva1 = tiempo_curva(r1, ang1)

        # Recta 2
        t2, x2, v2 = runge_kutta_cond_parada(ecuacion_recta, x1[-1], v_antes_curva1, h, x1[-1] + d2, parar_por_posicion, F2)
        v_antes_curva2 = v2[-1]
        tiempo_curva2 = tiempo_curva(r2, ang2)

        # Recta 3
        t3, x3, v3 = runge_kutta_cond_parada(ecuacion_recta, x2[-1], v_antes_curva2, h, x2[-1] + d3, parar_por_posicion, F3)

        tiempo_total = t1[-1] + tiempo_curva1 + t2[-1] + tiempo_curva2 + t3[-1]
        return round(tiempo_total, 3)

    except Exception as e:
        return f"Error: {e}"

# ---------- Valores a modificar manualmente ----------

F1 = 45000   # N
F2 = 16000  # N
F3 = 40000  # N
r1 = 10    # m
r2 = 20    # m

# ---------- Ejecución ----------
tiempo = calcular_tiempo_total(F1, F2, F3, r1, r2)
print(f"Tiempo total del recorrido: {tiempo} segundos")
def buscar_mejor_combinacion():
    mejores = []
    for F1 in range(0, 45001, 1000):
        for F2 in range(-20000, 20000, 1000):  # Freno/aceleración en la recta 2
            for F3 in range(0, 46000, 5000):
                for r1 in range(10, 41, 5):
                    for r2 in range(20, 41, 5):
                        tiempo = calcular_tiempo_total(F1, F2, F3, r1, r2)
                        if isinstance(tiempo, float) and tiempo < 1000:  # Filtra penalizados
                            mejores.append((tiempo, F1, F2, F3, r1, r2))
    
    mejores.sort()
    print("\nTop 5 combinaciones más rápidas:")
    for i, (t, F1, F2, F3, r1, r2) in enumerate(mejores[:5], 1):
        print(f"{i}) Tiempo: {t}s - F1: {F1}, F2: {F2}, F3: {F3}, r1: {r1}, r2: {r2}")
    
    return mejores[:5]

mejores = buscar_mejor_combinacion()
