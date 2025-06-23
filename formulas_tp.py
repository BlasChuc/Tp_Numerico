import numpy as np

G = 9.81 # m/s²
V_INICIAL_MAX = 180 # km/h
M = 800 # kg
ACELERACION_MAX = 6 * G # m/s²
FUERZA_MAX = ACELERACION_MAX * M # N

def runge_kutta_4_orden_superior(f_u, t0, y0, u0, tf, h, condicion_limite):
    """
    Resuelve una EDO usando el método Runge-Kutta de 4to orden
    
    Parámetros:
    f_u: Función que define u' = d²y/dt² 
    t0: Valor inicial de t
    y0: Valor inicial de y
    u0: Valor inicial de u
    tf: Valor final de t
    h: Tamaño del paso
    
    Retorna:
    t: Arreglo de valores de t (tiempo)
    y: Arreglo de valores de y (posición)
    u: Arreglo de valores de u (velocidad)
    """
    # Calcular número de pasos
    n = int((tf - t0) / h) + 1
    
    t = []
    y = []
    u = []
    
    # Condiciones iniciales
    t.append(t0)
    y.append(y0)
    u.append(u0)
    
    # Método de Runge-Kutta de 4to orden
    for i in range(1, n):
        # 1era iteración
        k1 = u[i-1]
        m1 = f_u(t[i-1], y[i-1], u[i-1])

        # 2da iteración
        k2 = u[i-1] + (h * m1) / 2
        m2 = f_u(t[i-1] + h/2, y[i-1] + (h * k1)/2, k2)
        
        # 3ra iteración
        k3 = u[i-1] + (h * m2) / 2
        m3 = f_u(t[i-1] + h/2, y[i-1] + (h * k2)/2, k3)

        # 4ta iteración
        k4 = u[i-1] + (h * m3) / 2
        m4 = f_u(t[i-1] + h, y[i-1] + (h * k3), k4)

        # Actualizar y, u, t
        y.append(y[i-1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        u.append(u[i-1] + h * (m1 + 2 * m2 + 2 * m3 + m4) / 6)
        t.append(t[i-1] + h)

        if y[i] >= condicion_limite:
            break
    
    return t, y, u

def ecuacion_curvas(theta, r, g):
    if g > ACELERACION_MAX:
        return 0
    
    return (- g/ r) * np.sin(theta)


def runge_kutta_4_orden_superior_2d(f, t0, x0, y0, vx0, vy0, tf, h, dist_lim):
    print(f, t0, x0, y0, vx0, vy0, tf, h)
    t = t0
    xs, ys = [x0], [y0]
    vxs, vys = [vx0], [vy0]

    x, y = x0, y0
    vx, vy = vx0, vy0

    while t < tf and np.sqrt(x**2 + y**2) < dist_lim:
        # k y m para X
        k1x = vx
        m1x = f(t, x, vx)

        k2x = (vx + (h * m1x)/2)
        m2x = f(t + h/2, x + (h * k1x)/2, k2x)

        k3x = (vx + (h * m2x)/2)
        m3x = f(t + h/2, x + (h * k2x)/2, k3x)

        k4x = (vx + (h * m3x)/2)
        m4x = f(t + h, x + (h * k3x), k4x)

        # k y m para Y
        k1y = vy
        m1y = f(t, y, vy)

        k2y = (vy + (h * m1y)/2)
        m2y = f(t + h/2, y + (h * k1y)/2, k2y)

        k3y = (vy + (h * m2y)/2)
        m3y = f(t + h/2, y + (h * k2y)/2, k3y)

        k4y = (vy + (h * m3y)/2)
        m4y = f(t + h, y + (h * k3y), k4y)

        # Actualización
        x = x + h * (k1x + 2*k2x + 2*k3x + k4x) / 6
        vx = vx + h * (m1x + 2*m2x + 2*m3x + m4x) / 6

        y = y + h * (k1y + 2*k2y + 2*k3y + k4y) / 6
        vy = vy + h * (m1y + 2*m2y + 2*m3y + m4y) / 6

        t += h
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)

    return xs, ys, vxs, vys, t