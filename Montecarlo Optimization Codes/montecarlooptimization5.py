import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(57)         # Semilla para reproducibilidad
# Numero de clientes, posiciones
num_clientes = 10
clientes  = np.random.rand(num_clientes, 2) * 100
deposito  = np.array([[50, 50]])
ubicaciones = np.vstack([deposito, clientes])

# Matriz con distancias euclidea entre todos los nodos
matriz_dist = cdist(ubicaciones, ubicaciones, metric='euclidean')

# Generacion aleatoria de posibles rutas
num_simulaciones = 10000
mejor_distancia  = np.inf
mejor_ruta       = None

for _ in range(num_simulaciones):
    ruta          = np.random.permutation(range(1, num_clientes + 1))   # Permuta aleatoria
    ruta_completa = np.concatenate(([0], ruta, [0]))

    # Distancia de la ruta
    dist = sum(matriz_dist[ruta_completa[i], ruta_completa[i + 1]]
               for i in range(len(ruta_completa) - 1))

    # Mejor ruta hasta el momento
    if dist < mejor_distancia:
        mejor_distancia = dist
        mejor_ruta      = ruta_completa

# Visualizacion
plt.figure(figsize=(8, 6))

plt.scatter(clientes[:, 0], clientes[:, 1], c='blue', s=100, label='Clientes')
plt.scatter(deposito[0, 0], deposito[0, 1], c='red', s=200, marker='s', label='Depósito')

for i, (x, y) in enumerate(clientes, start=1):
    plt.text(x + 1, y + 1, str(i), color='blue')

plt.text(deposito[0, 0] + 1, deposito[0, 1] + 2, '0', color='red', fontweight='bold')

# Mejor ruta
for i in range(len(mejor_ruta) - 1):
    a, b = mejor_ruta[i], mejor_ruta[i + 1]
    plt.plot([ubicaciones[a, 0], ubicaciones[b, 0]],
             [ubicaciones[a, 1], ubicaciones[b, 1]], 'k-')

plt.title(f"TSP mediante Monte Carlo para {num_clientes} nodos\n"
          f"Simulaciones: {num_simulaciones}; Distancia mínima: {mejor_distancia:.2f};\n"
          f"Ruta óptima: {mejor_ruta}")

plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
