#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

Ejemplo más avanzado y de uso real de optimización basada en métodos de Monte C
arlo, aplicado a la logística y distribución.

Problema: Optimización de rutas de entrega

Una empresa de logística tiene un conjunto de clientes dispersos en una ciudad 
y un almacén central. La empresa quiere minimizar la distancia total recorrida 
por sus repartidores al visitar a todos los clientes una sola vez y regresar 
al almacén.

Este problema es una variante del Traveling Salesman Problem (TSP).

Solución con Monte Carlo

Generamos muchas rutas aleatorias entre los clientes.

Calculamos la distancia total recorrida para cada ruta.

Seleccionamos la ruta con la menor distancia como la mejor solución encontrada.

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Configuración del problema
num_clients = 10  # Número de clientes
np.random.seed(42)  # Fijamos la semilla para reproducibilidad

# Generamos coordenadas aleatorias para los clientes en un mapa 100x100
clients = np.random.rand(num_clients, 2) * 100
depot = np.array([[50, 50]])  # Almacén central en el centro del mapa
locations = np.vstack([depot, clients])  # Todas las ubicaciones

# Calculamos la matriz de distancias entre todos los puntos
distance_matrix = cdist(locations, locations, metric='euclidean')

# Monte Carlo Simulation: Generamos rutas aleatorias
num_simulations = 10000
best_distance = float('inf')
best_route = None

for _ in range(num_simulations):
    route = np.random.permutation(range(1, num_clients + 1))  # Generamos una ruta aleatoria (excluyendo el almacén)
    full_route = np.concatenate(([0], route, [0]))  # Añadimos el almacén al inicio y al final
    
    # Calculamos la distancia total de la ruta
    total_distance = sum(distance_matrix[full_route[i], full_route[i + 1]] for i in range(len(full_route) - 1))
    
    # Guardamos la mejor ruta encontrada
    if total_distance < best_distance:
        best_distance = total_distance
        best_route = full_route

# Visualización de la mejor ruta
plt.figure(figsize=(8, 6))
plt.scatter(clients[:, 0], clients[:, 1], color='blue', label="Clientes", s=100)
plt.scatter(depot[:, 0], depot[:, 1], color='red', marker="s", s=200, label="Almacén")

# Dibujamos la mejor ruta encontrada
for i in range(len(best_route) - 1):
    start, end = best_route[i], best_route[i + 1]
    plt.plot([locations[start, 0], locations[end, 0]], [locations[start, 1], locations[end, 1]], 'k-')

plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.title(f"Optimización de Rutas con Monte Carlo (Distancia: {best_distance:.2f})")
plt.legend()
plt.grid()
plt.show()

# Imprimir la mejor ruta
print(f"\nMejor ruta encontrada: {best_route}")
print(f"Distancia total mínima encontrada: {best_distance:.2f}")
