# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 18:47:25 2025
@author: sgmrz
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def funcion_objetivo(x):
    # Función con múltiples máximos locales
    return np.sin(3*x) + 0.5*np.cos(5*x) - 0.1*x**2

def recocido_simulado(f_obj, x_inicio=0.0, temp_inicio=10.0, temp_final=0.01,
                      tasa_enfriamiento=0.98, max_iter=500,
                      long_ciclo=20):
    # Valores iniciales
    x_actual  = x_inicio
    f_act     = f_obj(x_actual)
    mejor_x   = x_actual
    mejor_f   = f_act
    T         = temp_inicio
    
    historial = []
    iter_total= 0

    # Mientras T suficientemente alta y no demasiadas iteraciones
    while T > temp_final and iter_total < max_iter:
        
        # Explorar varios vecinos a la misma T
        for _ in range(long_ciclo):
            iter_total += 1
            # Perturbación aleatoria según distribución Uniforme
            delta_x = np.random.uniform(-0.5, 0.5)
            x_nuevo = x_actual + delta_x
            f_nuevo = f_obj(x_nuevo)
            delta   = f_nuevo - f_act

            # Criterio de Metropolis
            if delta >= 0 or np.exp(delta / T) > np.random.rand():
                # Acepto el nuevo estado
                print(f'\nAcepto, paso de ({x_actual}, {f_act})')
                x_actual, f_act = x_nuevo, f_nuevo
                # Actualizo óptimo global si corresponde
                print(f'A  ({x_nuevo}, {f_nuevo})')
                if f_act > mejor_f:
                    print(f'¡{f_act} > {mejor_f}!, mejoro mi (x,f(x)): paso a ({x_actual}, {f_act})')
                    mejor_x, mejor_f = x_actual, f_act
                else:
                    print(f'¡{f_act} < {mejor_f}! No he mejorado mi mejor f')
            time.sleep(1)
            historial.append((x_actual, f_act))
            if iter_total >= max_iter:
                break

        # Enfriamiento geométrico
        T *= tasa_enfriamiento

    return mejor_x, mejor_f, historial

# Ejecutar la optimización
x_opt, f_opt, historial = recocido_simulado(f_obj=funcion_objetivo)

# Visualización de la trayectoria
x_vals = np.linspace(-5, 5, 1000)
y_vals = funcion_objetivo(x_vals)
hist_x, hist_y = zip(*historial)
iters = np.arange(len(hist_x))

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label='Función objetivo', color='blue')
sc = plt.scatter(hist_x, hist_y, c=iters, cmap='viridis', alpha=0.4, label='Camino explorado')
plt.colorbar(sc, label='Iteración')
plt.scatter([x_opt], [f_opt], color='red', s=200, marker='*', label='Solución óptima')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Optimización por Recocido Simulado\nMejor punto: ({x_opt:.4f}, {f_opt:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mejor x: {x_opt:.4f}, Mejor f(x): {f_opt:.4f}")
