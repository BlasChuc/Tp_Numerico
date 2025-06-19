import numpy as np

G = 9.81 # m/s²
V_INICIAL_MAX = 180 # km/h
M = 800 # kg
ACELERACION_MAX = 6 * G # m/s²
FUERZA_MAX = ACELERACION_MAX * M # N

def runge_kutta_4_orden_superior(f_u, t0, y0, u0, tf, h, *args):
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
    t: Arreglo de valores de t
    y: Arreglo de valores de y
    u: Arreglo de valores de u
    """
    # Calcular número de pasos
    n = int((tf - t0) / h) + 1
    
    t = [0] * n
    y = [0] * n
    u = [0] * n
    
    # Condiciones iniciales
    t[0] = t0
    y[0] = y0
    u[0] = u0
    
    # Método de Runge-Kutta de 4to orden
    for i in range(1, n):
        # 1era iteración
        k1 = u[i-1]
        m1 = f_u(t[i-1], y[i-1], u[i-1], *args)

        # 2da iteración
        k2 = u[i-1] + (h * m1) / 2
        m2 = f_u(t[i-1] + h/2, y[i-1] + (h * k1)/2, k2, *args)
        
        # 3ra iteración
        k3 = u[i-1] + (h * m2) / 2
        m3 = f_u(t[i-1] + h/2, y[i-1] + (h * k2)/2, k3, *args)

        # 4ta iteración
        k4 = u[i-1] + (h * m3) / 2
        m4 = f_u(t[i-1] + h, y[i-1] + (h * k3), k4, *args)

        # Actualizar y, u, t
        y[i] = y[i-1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        u[i] = u[i-1] + h * (m1 + 2 * m2 + 2 * m3 + m4) / 6
        t[i] = t[i-1] + h
    
    return t, y, u

def ecuacion_curvas(t,theta, omega, r, max_G):
    if max_G > ACELERACION_MAX:
        return 0
    
    return (- max_G/ r) * np.sin(theta)
