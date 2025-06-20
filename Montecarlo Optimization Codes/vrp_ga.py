import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from deap import base, creator, tools

# Configuracion del problema 15 - 3 - 200
num_clientes       = 15
num_vehiculos      = 8
capacidad_vehiculo = 200

demandas_clientes = np.random.randint(5, 15, size=num_clientes)

np.random.seed(177)
ubicaciones_clientes = np.random.rand(num_clientes, 2) * 100  # reticulo 100x100
ubicacion_deposito   = np.array([[50, 50]])                   # nodo 0
todas_ubicaciones    = np.vstack([ubicacion_deposito, ubicaciones_clientes])

matriz_distancias = distance_matrix(todas_ubicaciones, todas_ubicaciones)

# Rutas iniciales mediante Monte Carlo
def generar_rutas_iniciales():
    clientes = list(range(1, num_clientes + 1))
    random.shuffle(clientes)
    rutas = []
    while clientes:
        ruta, carga = [], 0
        while clientes and carga + demandas_clientes[clientes[0] - 1] <= capacidad_vehiculo:
            c = clientes.pop(0)
            ruta.append(c)
            carga += demandas_clientes[c - 1]
        rutas.append(ruta)
    return rutas

# Funcion de coste
def costo_ruta(rutas):
    dist = 0
    for ruta in rutas:
        if ruta:
            camino = [0] + ruta + [0]
            dist  += sum(matriz_distancias[camino[i], camino[i + 1]]
                         for i in range(len(camino) - 1))
    return dist

# Configuracion del Algoritmo Genetico
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individuo", tools.initIterate, creator.Individuo, generar_rutas_iniciales)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)

def evaluar(ind): return costo_ruta(ind),

# Operador de cruce
def cruce(p1, p2):
    corte1, corte2 = sorted(random.sample(range(1, num_clientes), 2))
    seg1 = sum(p1, [])[corte1:corte2]
    seg2 = sum(p2, [])[corte1:corte2]
    
    def rellenar(padre, segmento):
        resto = [c for c in sum(padre, []) if c not in segmento]
        out   = []
        while resto:
            ruta, carga = [], 0
            while resto and carga + demandas_clientes[resto[0] - 1] <= capacidad_vehiculo:
                ruta.append(resto.pop(0))
                carga += demandas_clientes[ruta[-1] - 1]
            out.append(ruta)
        return out
    
    return (creator.Individuo(rellenar(p1, seg1)),
            creator.Individuo(rellenar(p2, seg2)))

# Operador de mutacion
def mutar(ind):
    if len(ind) > 1:
        r1, r2 = random.sample(range(len(ind)), 2)
        if ind[r1] and ind[r2]:
            c1, c2 = random.choice(ind[r1]), random.choice(ind[r2])
            
            ind[r1].remove(c1); ind[r2].remove(c2)
            ind[r1].append(c2);  ind[r2].append(c1)
    return ind,

toolbox.register("mate", cruce)
toolbox.register("mutate", mutar)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluar)

# GA con los operadores de cruce y mutacion
poblacion = toolbox.poblacion(n=100)
GENERACIONES, PROB_CRUCE, PROB_MUTACION = 50, 0.7, 0.2

for _ in range(GENERACIONES):
    descendencia = toolbox.select(poblacion, len(poblacion))
    descendencia = list(map(toolbox.clone, descendencia))

    for h1, h2 in zip(descendencia[::2], descendencia[1::2]):
        if random.random() < PROB_CRUCE:
            toolbox.mate(h1, h2)
            del h1.fitness.values, h2.fitness.values
            
    for mut in descendencia:
        if random.random() < PROB_MUTACION:
            toolbox.mutate(mut); del mut.fitness.values

    for ind in (i for i in descendencia if not i.fitness.valid):
        ind.fitness.values = toolbox.evaluate(ind)

    poblacion[:] = descendencia

"""
# Mejor solucion
mejor_solucion = tools.selBest(poblacion, 1)[0]
mejor_costo = costo_ruta(mejor_solucion)
"""

# Ordenamos de mejor a peor según fitness
poblacion_ordenada = sorted(poblacion, key=lambda ind: ind.fitness.values)

# Buscamos la primera solución que cumpla la capacidad
for ind in poblacion_ordenada:
    cargas = [sum(demandas_clientes[c-1] for c in ruta) for ruta in ind]
    if all(c <= capacidad_vehiculo for c in cargas):
        # print(f"Aceptada la solución de rutas {ind} con cargas {cargas}")
        mejor_solucion = ind
        mejor_costo    = costo_ruta(ind)
        break
    # else:
    #     # imprimimos descartes mientras recorremos
    #     print(f"Exceso de carga: rutas {ind} con cargas {cargas}\n")
else:
    # SOLO se ejecuta si NINGÚN break fue alcanzado
    # print("Ninguna sollucion con carga admisible. Se toma la de menor distancia.")
    mejor_solucion = poblacion_ordenada[0]
    mejor_costo    = costo_ruta(mejor_solucion)

    
# Visualizacion de la mejor solucion
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Deposito y clientes
plt.scatter(ubicacion_deposito[0, 0], ubicacion_deposito[0, 1],
            color='red', s=200, marker='s', label='Deposito')
plt.scatter(ubicaciones_clientes[:, 0], ubicaciones_clientes[:, 1],
            color='mediumblue', s=75, label='Clientes')

# Colores para cada vehiculo
colores = ['limegreen', 'orangered', 'darkviolet', 'dodgerblue']

for idx, ruta in enumerate(mejor_solucion):
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

# Etiquetado
for i, (x, y) in enumerate(ubicaciones_clientes, start=1):
    plt.text(x, y + 1.5, str(i), fontsize=11,
             color='black', fontweight='bold')

plt.title(f'Solucion optimizada del VRP\nDistancia total: {mejor_costo:.2f}')
plt.legend(loc='lower left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Rutas optimas por vehículo
print('\nRutas optimas por vehiculo:')
for i, ruta in enumerate(mejor_solucion, start=1):
    carga = sum(demandas_clientes[c-1] for c in ruta)
    print(f'Vehiculo {i} (carga {carga}/{capacidad_vehiculo}): Deposito -> {" -> ".join(map(str, ruta))} -> Deposito')
print(f'\nDistancia total optimizada: {mejor_costo:.2f}')
