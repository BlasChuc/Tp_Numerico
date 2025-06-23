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