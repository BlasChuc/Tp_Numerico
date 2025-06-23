from formulas_tp import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Cargar imagen de fondo
img = mpimg.imread("Captura de pantalla de 2025-06-2(1).png")

F = 0 # Suponiendo una fuerza constante a lo largo de toda la recta

def ecuacion_recta(tiempo, posicion, velocidad):
    if F > FUERZA_MAX:
        return 0
    return F / M # Retorna la aceleración (Fuerza / Masa)

def calcular_velocidad_resultante(v_final_x, v_final_y):
     return np.sqrt(v_final_x**2 + v_final_y**2)



def calcular_tiempo_total(r1, r2, v_inicial_x, v_inicial_y, d1, d2, pos_inicial_x, pos_inicial_y, t_lim_1):
    x_total = []
    y_total = []    
    h = 0.1
    ang1, ang2 = np.pi / 2, -np.pi / 4  # Ángulos de giro en radianes


    # Recta 1 (en eje x)
    """ t_x_1, pos_x_1, v_x_1 = runge_kutta_4_orden_superior(ecuacion_recta, 0.0, pos_inicial_x, v_inicial_x, t_lim_1, h, d1)
    v_final_x = v_x_1[-1]
    x_total += pos_x_1 """

    xs, ys, vxs, vys, t = runge_kutta_4_orden_superior_2d(ecuacion_recta, 0.0, pos_inicial_x, pos_inicial_y, v_inicial_x, v_inicial_y, t_lim_1, h, d1)
    v_final_x = vxs[-1]
    x_total += xs
    y_total += ys


    # Recta 1 (en eje y)
    """ t_y_1, pos_y_1, v_y_1 = runge_kutta_4_orden_superior(ecuacion_recta, 0.0, pos_inicial_y, v_inicial_y, t_lim_1, h, d2)
    v_final_y = v_y_1[-1]
    y_total += pos_y_1
 """
    v_antes_curva1 = calcular_velocidad_resultante(vxs[-1], vys[-1])
    print("Tiempo final en el eje X:", xs[-1])
    print("Tiempo final en el eje Y:", ys[-1])

    

    # Curva 1 
    #t_x_2, pos_x_2, v_x_2 = runge_kutta_4_orden_superior(ecuacion_curvas, pos_x_1[-1], v_antes_curva1, h, pos_x_1[-1], r1, ang1)

    #tiempo_curva2 = tiempo_curva(r2, ang2)

    return x_total, y_total


r1 = 20  # Radio de la curva 1
r2 = 30  # Radio de la curva 2
v_inicial_x = 50.0  # Velocidad inicial en el eje X (m/s)
v_inicial_y = -2.5   # Velocidad inicial en el eje Y (m/s)
d2 = -2

x_ini, y_ini = 2, 14
x_fin, y_fin = 69, 9

dx = x_fin - x_ini
dy = y_fin - y_ini
d_lim_recta_1 = np.sqrt(dx**2 + dy**2)


pos_inicial_x = 2.0  # Posición inicial en el eje X (m)
pos_inicial_y = 14.0  # Posición inicial en el eje Y (m)
t_lim_1 = 2

x_total, y_total = calcular_tiempo_total(r1, r2, v_inicial_x, v_inicial_y, d_lim_recta_1, d2, pos_inicial_x, pos_inicial_y, t_lim_1)


plt.figure(figsize=(10, 6))
plt.imshow(img, extent=[0, 100, 0, 80], aspect='auto', zorder=0)
plt.plot(x_total, y_total, label="Trayectoria", color='black', linewidth=2, zorder=1)
plt.axis('equal')
plt.grid(True)
plt.title("Trayectoria con fondo personalizado")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()


plt.tight_layout()
plt.show()

