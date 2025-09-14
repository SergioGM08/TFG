# -*- coding: utf-8 -*-
"""
OPTIMIZACIÓN DE RUTAS DE ENTREGA (TSP exacto)

Este script evalúa todas las rutas posibles (10! = 3 628 800) para 10 clientes 
y un depósito central, calcula la distancia total de cada ruta y selecciona 
la más corta. Además mide el tiempo que tarda en generar y evaluar todas las rutas.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import itertools
from math import factorial
import time

# Configuración del problema
num_clientes = 12
np.random.seed(57)  # semilla para reproducibilidad

# Coordenadas aleatorias de los clientes (cuadrícula 100×100)
clientes    = np.random.rand(num_clientes, 2) * 100
deposito    = np.array([[50, 50]])                    # nodo 0
ubicaciones = np.vstack([deposito, clientes])         # nodos 0..10

# Matriz de distancias euclídeas
matriz_dist = cdist(ubicaciones, ubicaciones, metric='euclidean')

# Todas las rutas
total_rutas = factorial(num_clientes)
print(f"Evaluando todas las rutas posibles: {total_rutas}")

mejor_distancia = np.inf
mejor_ruta      = None

# Comenzar medición de tiempo
tiempo_inicio = time.perf_counter()
for ruta in itertools.permutations(range(1, num_clientes + 1)):
    ruta_completa = (0,) + ruta + (0,)  # ida y vuelta al depósito
    distancia = sum(
        matriz_dist[ruta_completa[i], ruta_completa[i + 1]]
        for i in range(len(ruta_completa) - 1)
    )
    if distancia < mejor_distancia:
        mejor_distancia = distancia
        mejor_ruta      = ruta_completa

# Fin de medición de tiempo
tiempo_fin = time.perf_counter()
print(f"Tiempo de generación y evaluación de rutas: {tiempo_fin - tiempo_inicio:.2f} segundos")

# Visualización
plt.figure(figsize=(8, 6))

# Dibujar puntos de clientes y depósito
plt.scatter(clientes[:, 0], clientes[:, 1], c='blue',  s=100, label='Clientes')
plt.scatter(deposito[0, 0],   deposito[0, 1],   c='red',   s=200, marker='s', label='Depósito')

# Etiquetar clientes (1..10)
for i, (x, y) in enumerate(clientes, start=1):
    plt.text(x + 1, y + 1, str(i), color='blue')

# Etiquetar depósito (0)
plt.text(deposito[0, 0] + 1, deposito[0, 1] + 2, '0', color='red', fontweight='bold')

# Dibujar la mejor ruta encontrada
for i in range(len(mejor_ruta) - 1):
    a, b = mejor_ruta[i], mejor_ruta[i + 1]
    plt.plot(
        [ubicaciones[a, 0], ubicaciones[b, 0]],
        [ubicaciones[a, 1], ubicaciones[b, 1]],
        'k-'
    )

# Título con datos de la solución
plt.title(
    f"TSP exacto para {num_clientes} nodos\n"
    f"Distancia mínima = {mejor_distancia:.2f}\n"
    f"Ruta óptima = {mejor_ruta}"
)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
