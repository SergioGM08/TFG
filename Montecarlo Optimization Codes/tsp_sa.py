# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time

np.random.seed(57)  # Semilla para reproducibilidad

# -----------------------------
# Datos del TSP
# -----------------------------
num_clientes = 10
clientes     = np.random.rand(num_clientes, 2) * 100
deposito     = np.array([[50, 50]])
ubicaciones  = np.vstack([deposito, clientes])   # nodo 0 es el depósito

# Matriz de distancias euclídeas entre todos los nodos (0 = depósito)
matriz_dist = cdist(ubicaciones, ubicaciones, metric='euclidean')

# -----------------------------
# Utilidades para el TSP
# -----------------------------
def distancia_ruta_clientes(orden_clientes, D):
    """
    Coste total incluyendo ida y vuelta al depósito:
    0 -> c1 -> c2 -> ... -> cn -> 0
    """
    if len(orden_clientes) == 0:
        return 0.0
    # del depósito al primero
    dist = D[0, orden_clientes[0]]
    # entre clientes consecutivos
    for i in range(len(orden_clientes)-1):
        dist += D[orden_clientes[i], orden_clientes[i+1]]
    # del último al depósito
    dist += D[orden_clientes[-1], 0]
    return dist

def ruta_completa(orden_clientes):
    """Devuelve la ruta completa con depósito al inicio y al final."""
    return np.array([0] + list(orden_clientes) + [0], dtype=int)

def vecino_2opt(orden):
    """
    Genera un vecino con movimiento 2-opt:
    elige dos índices i<j y revierte el segmento orden[i:j+1].
    """
    n = len(orden)
    if n < 2:
        return orden.copy()
    i, j = np.sort(np.random.choice(n, size=2, replace=False))
    if i == j:
        return orden.copy()
    nuevo = orden.copy()
    nuevo[i:j+1] = nuevo[i:j+1][::-1]
    return nuevo

def vecino_swap(orden):
    """
    Genera un vecino intercambiando dos ciudades (swap).
    """
    n = len(orden)
    if n < 2:
        return orden.copy()
    i, j = np.sort(np.random.choice(n, size=2, replace=False))
    nuevo = orden.copy()
    nuevo[i], nuevo[j] = nuevo[j], nuevo[i]
    return nuevo

def generar_vecino(orden, p_2opt=0.7):
    """
    Mezcla de movimientos (por defecto, 70% 2-opt y 30% swap).
    """
    if np.random.rand() < p_2opt:
        return vecino_2opt(orden)
    else:
        return vecino_swap(orden)

# -----------------------------
# Temple simulado para el TSP
# -----------------------------
def temple_simulado_tsp(D,
                        num_clientes,
                        temp_inicio=10,
                        temp_final=0.01,
                        tasa_enfriamiento=0.98,
                        long_ciclo=2,
                        max_iter=1000):
    """
    Temple simulado clásico (minimización):
      - Estado: orden de visita de los clientes (sin incluir depósito).
      - Vecino: 2-opt o swap.
      - Criterio de Metrópolis: acepta peores con prob. exp(-Δ/T).

    Devuelve: (mejor_orden, mejor_coste, historial_costes)
    """
    # Estado inicial: permutación aleatoria de los clientes 1..num_clientes
    actual = np.random.permutation(np.arange(1, num_clientes+1))
    coste_actual = distancia_ruta_clientes(actual, D)
    mejor = actual.copy()
    mejor_coste = coste_actual

    T = temp_inicio
    historial = [coste_actual]
    it_total = 0

    t0 = time.perf_counter()

    while T > temp_final and it_total < max_iter:
        for _ in range(long_ciclo):
            it_total += 1
            vecino = generar_vecino(actual)
            coste_vecino = distancia_ruta_clientes(vecino, D)
            delta = coste_vecino - coste_actual  # minimización

            # Criterio de Metropolis
            if delta <= 0 or np.exp(-delta / T) > np.random.rand():
                actual, coste_actual = vecino, coste_vecino
                if coste_actual < mejor_coste:
                    mejor, mejor_coste = actual.copy(), coste_actual

            historial.append(coste_actual)
            if it_total >= max_iter:
                break

        # Enfriamiento geométrico
        T *= tasa_enfriamiento

    t1 = time.perf_counter()
    duracion = t1 - t0
    return mejor, mejor_coste, historial, duracion

# -----------------------------
# Ejecutar temple simulado
# -----------------------------
mejor_orden, mejor_distancia, historial_costes, dur = temple_simulado_tsp(
    matriz_dist,
    num_clientes=num_clientes,
    temp_inicio=100.0,
    temp_final=1e-3,
    tasa_enfriamiento=0.995,
    long_ciclo=200,
    max_iter=20000
)

mejor_ruta = ruta_completa(mejor_orden)

print(f"Tiempo de optimización (SA): {dur:.3f} s")
print(f"Mejor distancia: {mejor_distancia:.2f}")
print(f"Mejor ruta: {mejor_ruta}")

# -----------------------------
# Visualización
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(clientes[:, 0], clientes[:, 1], c='blue', s=100, label='Clientes')
plt.scatter(deposito[0, 0], deposito[0, 1], c='red', s=200, marker='s', label='Depósito')

# Etiquetas
for i, (x, y) in enumerate(clientes, start=1):
    plt.text(x + 1, y + 1, str(i), color='blue')
plt.text(deposito[0, 0] + 1, deposito[0, 1] + 2, '0', color='red', fontweight='bold')

# Dibuja la mejor ruta
for i in range(len(mejor_ruta) - 1):
    a, b = mejor_ruta[i], mejor_ruta[i + 1]
    plt.plot([ubicaciones[a, 0], ubicaciones[b, 0]],
             [ubicaciones[a, 1], ubicaciones[b, 1]], 'k-')

plt.title(
    f"TSP por Temple Simulado\n"
    f"Distancia mínima: {mejor_distancia:.2f}\n"
    f"Ruta: {mejor_ruta}"
)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
