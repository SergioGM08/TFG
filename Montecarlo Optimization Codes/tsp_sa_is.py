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
    dist = D[0, orden_clientes[0]]  # del depósito al primero
    for i in range(len(orden_clientes)-1):  # entre clientes consecutivos
        dist += D[orden_clientes[i], orden_clientes[i+1]]
    dist += D[orden_clientes[-1], 0]  # del último al depósito
    return dist

def ruta_completa(orden_clientes):
    """Devuelve la ruta completa con depósito al inicio y al final."""
    return np.array([0] + list(orden_clientes) + [0], dtype=int)

# ---------- Muestreo enfatizado: construcción de orden ----------
def generar_orden_enfatizado(D, num_clientes, alpha=2.0, eps=1e-9):
    """
    Construye un orden [c1, c2, ..., cn] (sin el 0) partiendo del depósito.
    En cada paso: prob ∝ (1 / distancia(Actual,c)^alpha).
    """
    restantes = list(range(1, num_clientes + 1))
    orden = []
    actual = 0  # depósito

    while restantes:
        d     = np.array([D[actual, c] for c in restantes])
        inv   = 1.0 / np.maximum(d, eps)
        pesos = inv ** alpha
        probs = pesos / pesos.sum()
        idx   = np.random.choice(len(restantes), p=probs)
        sig   = restantes.pop(idx)
        orden.append(sig)
        actual = sig

    return np.array(orden, dtype=int)

# ---------- Vecino por muestreo enfatizado (reconstrucción parcial) ----------
def vecino_enfatizado_parcial(orden, D, alpha=2.0, eps=1e-9):
    """
    Toma un bloque aleatorio de la ruta y lo reconstruye con muestreo enfatizado:
    desde el cliente previo al bloque (o el depósito si el bloque arranca al inicio),
    elige el siguiente dentro del bloque con prob ∝ 1/dist^alpha hasta agotar el bloque.
    """
    n = len(orden)
    if n < 2:
        return orden.copy()

    i, j = np.sort(np.random.choice(n, size=2, replace=False))
    if i == j:
        return orden.copy()

    bloque  = list(orden[i:j+1])
    prefijo = list(orden[:i])
    sufijo  = list(orden[j+1:])

    reconstruido = []
    actual = 0 if i == 0 else prefijo[-1]  # contexto de arranque

    candidatos = bloque.copy()
    while candidatos:
        d     = np.array([D[actual, c] for c in candidatos])
        inv   = 1.0 / np.maximum(d, eps)
        pesos = inv ** alpha
        probs = pesos / pesos.sum()
        idx   = np.random.choice(len(candidatos), p=probs)
        sig   = candidatos.pop(idx)
        reconstruido.append(sig)
        actual = sig

    nuevo_orden = np.array(prefijo + reconstruido + sufijo, dtype=int)
    return nuevo_orden

# -----------------------------
# Temple simulado para el TSP (inicial y vecinos con muestreo enfatizado)
# -----------------------------
def temple_simulado_tsp(D,
                        num_clientes,
                        temp_inicio=100.0,
                        temp_final=1e-3,
                        tasa_enfriamiento=0.995,
                        long_ciclo=200,
                        max_iter=20000,
                        alpha_inicial=2.0,
                        alpha_vecino=2.0):
    """
    Temple simulado (minimización) con muestreo enfatizado:
      - Estado: orden de visita de clientes (sin el 0).
      - Inicial: construido con prob ∝ 1/d^alpha_inicial.
      - Vecino: reconstrucción parcial por muestreo enfatizado (alpha_vecino).
      - Aceptación: Metrópolis exp(-Δ/T).
    """
    # Estado inicial (enfatizado)
    actual = generar_orden_enfatizado(D, num_clientes, alpha=alpha_inicial)
    coste_actual = distancia_ruta_clientes(actual, D)
    mejor, mejor_coste = actual.copy(), coste_actual

    T = temp_inicio
    historial = [coste_actual]
    it_total = 0

    t0 = time.perf_counter()

    while T > temp_final and it_total < max_iter:
        for _ in range(long_ciclo):
            it_total += 1
            vecino = vecino_enfatizado_parcial(actual, D, alpha=alpha_vecino)
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
    temp_inicio=10,
    temp_final=1e-3,
    tasa_enfriamiento=0.995,
    long_ciclo=200,
    max_iter=10000,
    alpha_inicial=2.0,     # sesgo en la construcción inicial
    alpha_vecino=2.0       # sesgo en la vecindad (reconstrucción parcial)
)

mejor_ruta = ruta_completa(mejor_orden)

print(f"Tiempo de optimización (SA+IS): {dur:.3f} s")
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
ruta = [0] + list(mejor_orden) + [0]
for i in range(len(ruta) - 1):
    a, b = ruta[i], ruta[i + 1]
    plt.plot([ubicaciones[a, 0], ubicaciones[b, 0]],
             [ubicaciones[a, 1], ubicaciones[b, 1]], 'k-')

plt.title(
    f"TSP híbrido SA + IS\n"
    f"Distancia mínima: {mejor_distancia:.2f}\n"
    f"Ruta: {mejor_ruta}"
)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()