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
ubicaciones  = np.vstack([deposito, clientes])

# Distancias euclídeas entre todos los nodos (0 = depósito)
matriz_dist = cdist(ubicaciones, ubicaciones, metric='euclidean')

# -------------------------------------------------------------
# NUEVO: Generación de rutas por MUestreo ENFATIZADO (IS)
# -------------------------------------------------------------
def generar_ruta_enfatizada(matriz_dist, num_clientes, alpha=2.0, eps=1e-9):
    """
    Construye una ruta 0 -> ... -> 0 eligiendo el siguiente cliente
    con probabilidad proporcional a (1 / distancia^alpha) desde el nodo actual.
    - alpha controla cuánto sesgamos hacia los cercanos (alpha=0 ≈ uniforme).
    - eps evita división por cero.
    Devuelve: ruta completa (incluye 0 al inicio y final).
    """
    # Conjunto de clientes aún no visitados (1..num_clientes)
    restantes = list(range(1, num_clientes + 1))
    ruta = [0]  # empezamos en el depósito

    actual = 0
    while restantes:
        # distancias desde 'actual' a cada candidato no visitado
        d = np.array([matriz_dist[actual, c] for c in restantes])
        # IMPORTANCIA: prob ∝ 1 / d^alpha
        inv = 1.0 / np.maximum(d, eps)
        pesos = inv ** alpha
        probs = pesos / pesos.sum()

        # elegimos el siguiente cliente según probs
        idx = np.random.choice(len(restantes), p=probs)
        siguiente = restantes.pop(idx)
        ruta.append(siguiente)
        actual = siguiente

    ruta.append(0)  # volvemos al depósito
    return np.array(ruta, dtype=int)

# -------------------------------------------------------------
# Función de coste: longitud total de una ruta completa
# -------------------------------------------------------------
def distancia_ruta(ruta, D):
    nodos_inicio = ruta[:-1]
    nodos_fin = ruta[1:]
    
    distancias = D[nodos_inicio, nodos_fin]
    
    return np.sum(distancias)

# -------------------------------------------------------------
# Bucle Monte Carlo con muestreo enfatizado
# -------------------------------------------------------------
num_simulaciones = 10000
mejor_distancia  = np.inf
mejor_ruta       = None

tiempo_inicio = time.perf_counter()

for _ in range(num_simulaciones):
    # --- CAMBIO CLAVE: antes usabas np.random.permutation; ahora IS sesgado ---
    ruta_completa = generar_ruta_enfatizada(matriz_dist, num_clientes, alpha=2.0)
    
    dist = distancia_ruta(ruta_completa, matriz_dist)
    #print(f"Ruta: {ruta_completa}, {dist}")
    if dist < mejor_distancia:
        mejor_distancia = dist
        mejor_ruta      = ruta_completa

tiempo_fin = time.perf_counter()
print(f"Tiempo de generación y evaluación: {tiempo_fin - tiempo_inicio:.2f} s")

# -------------------------------------------------------------
# Visualización
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(clientes[:, 0], clientes[:, 1], c='blue', s=100, label='Clientes')
plt.scatter(deposito[0, 0], deposito[0, 1], c='red', s=200, marker='s', label='Depósito')

for i, (x, y) in enumerate(clientes, start=1):
    plt.text(x + 1, y + 1, str(i), color='blue')
plt.text(deposito[0, 0] + 1, deposito[0, 1] + 2, '0', color='red', fontweight='bold')

for i in range(len(mejor_ruta) - 1):
    a, b = mejor_ruta[i], mejor_ruta[i + 1]
    plt.plot([ubicaciones[a, 0], ubicaciones[b, 0]],
             [ubicaciones[a, 1], ubicaciones[b, 1]], 'k-')

plt.title(
    f"TSP con muestreo enfatizado (IS)\n"
    f"Simulaciones: {num_simulaciones}; Distancia mínima: {mejor_distancia:.2f}\n"
    f"Ruta: {mejor_ruta}"
)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
