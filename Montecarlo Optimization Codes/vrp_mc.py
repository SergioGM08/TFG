import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Configuración del problema 15 3 50
num_clientes       = 15
num_vehiculos      = 8
capacidad_vehiculo = 200
demandas_clientes = np.random.randint(5, 15, size=num_clientes)

np.random.seed(177)
ubicaciones_clientes = np.random.rand(num_clientes, 2) * 100  # retículo 100×100
ubicacion_deposito   = np.array([[50, 50]])                   # nodo 0
todas_ubicaciones    = np.vstack([ubicacion_deposito, ubicaciones_clientes])

matriz_distancias = distance_matrix(todas_ubicaciones, todas_ubicaciones)

# Función de coste: suma de distancias depósito→clientes→depósito
def costo_ruta(rutas):
    dist = 0.0
    for ruta in rutas:
        if ruta:
            # ida
            dist += matriz_distancias[0, ruta[0]]
            for i in range(len(ruta) - 1):
                dist += matriz_distancias[ruta[i], ruta[i+1]]
            # regreso
            dist += matriz_distancias[ruta[-1], 0]
    return dist

# Parámetros Monte Carlo
num_simulaciones = 10000
mejor_rutas, mejor_dist = None, float('inf')

for _ in range(num_simulaciones):
    ids = list(range(1, num_clientes+1)) # Rutas aleatorias
    random.shuffle(ids)

    # Generar rutas respetando capacidad
    rutas = []
    while ids:
        carga, ruta = 0, []
        i = 0
        while i < len(ids):
            c = ids[i]
            if carga + demandas_clientes[c-1] <= capacidad_vehiculo:
                carga += demandas_clientes[c-1]
                ruta.append(c)
                ids.pop(i)
            else:
                i += 1
        rutas.append(ruta)

    dist = costo_ruta(rutas)
    if dist < mejor_dist:
        mejor_dist, mejor_rutas = dist, rutas

# Visualizacion
plt.figure(figsize=(10,6))
ax = plt.gca()

# Deposito y clientes
ax.scatter(*ubicacion_deposito[0], c='red',  marker='s', s=200, label='Depósito')
ax.scatter(ubicaciones_clientes[:,0], ubicaciones_clientes[:,1],
           c='mediumblue', s=75, label='Clientes')

# Etiquetado
for idx, (x,y) in enumerate(ubicaciones_clientes, start=1):
    ax.text(x, y+2, f'{idx} ({demandas_clientes[idx-1]})',
            fontsize=9, ha='center')

# Colores de rutas
colores = ['limegreen','orangered','darkviolet','dodgerblue']

for idx, ruta in enumerate(mejor_rutas):
    recorrido = [0] + ruta + [0]
    col = colores[idx % len(colores)]
    etiqueta = f'Vehiculo {idx + 1}'
    
    for i in range(len(recorrido) - 1):
        inicio = todas_ubicaciones[recorrido[i]]
        fin    = todas_ubicaciones[recorrido[i + 1]]
        dx, dy = fin[0] - inicio[0], fin[1] - inicio[1]
        
        # Flecha
        ax.arrow(inicio[0], inicio[1], dx, dy,
                 head_width=2, head_length=3,
                 fc=col, ec=col,
                 length_includes_head=True, alpha=0.8,
                 label=etiqueta if i == 0 else None)
        
plt.title(f'Solución MC VRP con demanda\nDistancia total: {mejor_dist:.2f}')
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Resultados por consola
print('Rutas óptimas encontradas:')
for i, ruta in enumerate(mejor_rutas, start=1):
    carga = sum(demandas_clientes[c-1] for c in ruta)
    print(f'Vehículo {i} (carga {carga}/{capacidad_vehiculo}): Depósito -> {ruta} -> Depósito')
print(f'\nDistancia total optimizada: {mejor_dist:.2f}')
