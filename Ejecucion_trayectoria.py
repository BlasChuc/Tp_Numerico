from formulas_tp import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Cargar imagen de fondo
img = mpimg.imread("Captura de pantalla de 2025-06-2(1).png")

#F = 0 # Suponiendo una fuerza constante a lo largo de toda la recta

X_INI = 2
Y_INI = 14

def ecuacion_recta(tiempo, posicion, velocidad, *args):
    F = args[0] 
    if F > FUERZA_MAX:
        return 0
    return F / M # Retorna la aceleración (Fuerza / Masa)

def calcular_velocidad_resultante(v_final_x, v_final_y):
     return np.sqrt(v_final_x**2 + v_final_y**2)

def calcular_tiempo_total(r1, v_inicial_x, v_inicial_y, d1, d2, t_lim, F1, F2):
    x_total = []
    y_total = []    
    h = 0.1
    t0 = 0.0
    ang1= (np.pi*3)/7

    args_recta_1 = (F1)

    # Recta 1
    #xs1, ys1, vxs1, vys1, t1 = runge_kutta_4_orden_superior_2d(ecuacion_recta, t0, X_INI, Y_INI, v_inicial_x, v_inicial_y, t_lim, h, d1, args_recta_1)
    xs1, ys1, vxs1, vys1, t1 = simular_tramo_recto(t0, X_INI, Y_INI, v_inicial_x, v_inicial_y, t_lim, h, d1, args_recta_1)

    x_total += xs1
    y_total += ys1

    args_curva_1 = (ang1, r1, 1000)

    # Curva 1
    #xs2, ys2, vxs2, vys2, t2 = runge_kutta_4_orden_superior_2d(ecuacion_curvas, t1, xs1[-1], ys1[-1], v_final_x_1, v_final_y_1, t_lim, h, d2, args_curva_1)
    xs2, ys2, vxs2, vys2, t2 = simular_tramo_curvo(t1, xs1[-1], ys1[-1], vxs1[-1], vys1[-1], t_lim, h, args_curva_1)

    x_total += xs2
    y_total += ys2

    args_recta_2 = (F2)

    xs3, ys3, vxs3, vys3, t3 = simular_tramo_recto(t2, xs2[-1], ys2[-1], vxs2[-1], vys2[-1], t_lim, h, d2, args_recta_2)
    x_total += xs3
    y_total += ys3


    return x_total, y_total, t1+t2+t3


r1 = 24  # Radio de la curva 1
v_inicial_x = 40.0  # Velocidad inicial en el eje X (m/s)
v_inicial_y = -5.0 # Velocidad inicial en el eje Y (m/s)

x_fin, y_fin = 60, 10

dx = x_fin - X_INI
dy = y_fin - Y_INI
d_lim_recta_1 = np.sqrt(dx**2 + dy**2)
d_lim_recta_2 = 39


t_lim = 6

f1 = 5000
f2 = 0

x_total, y_total, t_total = calcular_tiempo_total(r1, v_inicial_x, v_inicial_y, d_lim_recta_1, d_lim_recta_2, t_lim, f1, f2)
print("Tiempo total:",t_total)

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

