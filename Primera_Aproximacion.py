from formulas_tp import *
import numpy as np

F = 1000 # Suponiendo una fuerza constante a lo largo de toda la recta

def ecuacion_recta(tiempo, posicion, velocidad):
    if F > FUERZA_MAX:
        return 0
    return F / M 

def main():
    y0 = 0.0
    u0 = 50.0 # Velocidad inicial en m/s
    t0 = 0.0 # Tiempo inicial
    tf = 1.0 # Tiempo final (esto implica en cuanto tiempo quiero que recorra la recta)
    h = 0.1 # Paso de integración
    f_u = ecuacion_recta # Es la ecuacion de F / M
    t, y, u = runge_kutta_4_orden_superior(f_u, t0, y0, u0, tf, h)
    print("Tiempo:\n\n", t)
    print("\n\nPosición:", y)
    print("\n\nVelocidad:", u)  

main()
