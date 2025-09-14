# -*- coding: utf-8 -*-
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import random  # ¡Importar la librería random!

# ------------------- Configuración del problema -------------------
num_clientes = 120
num_vehiculos = 8
capacidad_vehiculo = 200

# Usamos la semilla de NumPy para que la configuración sea siempre la misma
np.random.seed(177)
ubicaciones_clientes = np.random.rand(num_clientes, 2) * 100
ubicacion_deposito = np.array([[50, 50]])
todas_ubicaciones = np.vstack([ubicacion_deposito, ubicaciones_clientes])
demandas_clientes = np.random.randint(5, 15, size=num_clientes)
matriz_distancias = distance_matrix(todas_ubicaciones, todas_ubicaciones)

# ------------------- Coste: depósito -> clientes -> depósito -------------------
def costo_ruta(rutas):
    dist = 0.0
    for ruta in rutas:
        if ruta:
            # ida
            dist += matriz_distancias[0, ruta[0]]
            # entre clientes
            for i in range(len(ruta) - 1):
                dist += matriz_distancias[ruta[i], ruta[i + 1]]
            # regreso
            dist += matriz_distancias[ruta[-1], 0]
    return dist

# ------------------- Generación por muestreo enfatizado -------------------
def generar_rutas_enfatizadas(D, demandas, capacidad, alpha=2.0, eps=1e-9):
    restantes = set(range(1, num_clientes + 1))
    rutas = []

    while restantes:
        carga = 0
        actual = 0
        ruta = []

        while True:
            candidatos = [c for c in restantes if carga + demandas[c - 1] <= capacidad]
            if not candidatos:
                break

            d = np.array([D[actual, c] for c in candidatos])
            inv = 1.0 / np.maximum(d, eps)
            pesos = inv**alpha
            probs = pesos / pesos.sum()

            # Elegir siguiente cliente usando random.choices
            # np.random.choice usa el generador de NumPy. random.choices usa el generador de Python.
            c = random.choices(candidatos, weights=probs)[0]

            ruta.append(c)
            carga += demandas[c - 1]
            restantes.remove(c)
            actual = c
        rutas.append(ruta)
    return rutas

# ------------------- Monte Carlo con rutas enfatizadas -------------------
num_simulaciones = 10000
alpha_enfasis = 2.0

mejor_rutas, mejor_dist = None, float('inf')

for _ in range(num_simulaciones):
    rutas = generar_rutas_enfatizadas(
        D=matriz_distancias,
        demandas=demandas_clientes,
        capacidad=capacidad_vehiculo,
        alpha=alpha_enfasis,
        eps=1e-9
    )

    dist = costo_ruta(rutas)
    if dist < mejor_dist:
        mejor_dist, mejor_rutas = dist, rutas

# ------------------- Visualización -------------------
# plt.figure(figsize=(10, 6))
# ax = plt.gca()

# # Depósito y clientes
# ax.scatter(*ubicacion_deposito[0], c='red', marker='s', s=200, label='Depósito')
# ax.scatter(ubicaciones_clientes[:, 0], ubicaciones_clientes[:, 1],
#            c='mediumblue', s=75, label='Clientes')

# # Etiquetas de clientes con su demanda
# for idx, (x, y) in enumerate(ubicaciones_clientes, start=1):
#     ax.text(x, y + 2, f'{idx} ({demandas_clientes[idx-1]})',
#             fontsize=9, ha='center')

# # Colores de rutas
# colores = ['limegreen', 'orangered', 'darkviolet', 'dodgerblue']

# for idx, ruta in enumerate(mejor_rutas):
#     recorrido = [0] + ruta + [0]
#     col = colores[idx % len(colores)]
#     etiqueta = f'Vehículo {idx + 1}'

#     for i in range(len(recorrido) - 1):
#         inicio = todas_ubicaciones[recorrido[i]]
#         fin = todas_ubicaciones[recorrido[i + 1]]
#         dx, dy = fin[0] - inicio[0], fin[1] - inicio[1]

#         ax.arrow(inicio[0], inicio[1], dx, dy,
#                  # head_width=2, head_length=3,
#                  fc=col, ec=col,
#                  length_includes_head=True, alpha=0.8,
#                  label=etiqueta if i == 0 else None)

# plt.title(f'Monte-Carlo IS para SCVRP \nDistancia total: {mejor_dist:.2f}')
# plt.legend(loc='lower left')
# plt.grid(True, linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# # ------------------- Resultados por consola -------------------
# print('Rutas optimas por vehiculo:')
# for i, ruta in enumerate(mejor_rutas, start=1):
#     carga = sum(demandas_clientes[c - 1] for c in ruta)
#     print(f'Vehículo {i} (carga {carga}/{capacidad_vehiculo}): Depósito -> {ruta} -> Depósito')
# print(f'\nDistancia total optimizada: {mejor_dist:.2f}')