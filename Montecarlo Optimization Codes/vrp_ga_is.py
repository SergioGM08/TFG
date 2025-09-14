import numpy as np
import random
# import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from deap import base, creator, tools

# ---------------- Configuración del problema ----------------
num_clientes       = 120
num_vehiculos      = 8        # (informativo; el generador crea las rutas que hagan falta)
capacidad_vehiculo = 200

np.random.seed(177)
ubicaciones_clientes = np.random.rand(num_clientes, 2) * 100
ubicacion_deposito   = np.array([[50, 50]])
todas_ubicaciones    = np.vstack([ubicacion_deposito, ubicaciones_clientes])
demandas_clientes    = np.random.randint(5, 15, size=num_clientes)
matriz_distancias    = distance_matrix(todas_ubicaciones, todas_ubicaciones)

# ------------- Rutas iniciales por muestreo enfatizado -------------
def generar_rutas_iniciales(alpha=2.0, eps=1e-9):
    """
    Genera un conjunto de rutas factibles usando muestreo enfatizado:
    en cada paso se elige el siguiente cliente c de los que caben en
    capacidad con prob ∝ (1 / distancia(ultimo,c))**alpha.
    """
    restantes = list(range(1, num_clientes + 1))
    rutas = []
    while restantes:
        ruta, carga = [], 0
        actual = 0  # empezamos cada ruta en el depósito
        while True:
            # candidatos que aún no están servidos y caben por capacidad
            candidatos = [c for c in restantes
                          if carga + demandas_clientes[c-1] <= capacidad_vehiculo]
            if not candidatos:        # no cabe nadie más: cerramos esta ruta
                break

            # distancias desde el último nodo de la ruta al conjunto candidato
            d = np.array([matriz_distancias[actual, c] for c in candidatos])
            inv = 1.0 / np.maximum(d, eps)   # evita división por 0
            pesos = inv ** alpha
            probs = pesos / pesos.sum()

            # elegimos próximo cliente según las probabilidades enfatizadas
            idx = np.random.choice(len(candidatos), p=probs)
            c   = candidatos[idx]

            ruta.append(c)
            carga  += demandas_clientes[c-1]
            restantes.remove(c)
            actual = c                 # avanzamos el "cursor" de la ruta

        rutas.append(ruta)
    return rutas

# ------------------ Coste (distancia total) ------------------
def costo_ruta(rutas):
    dist = 0
    for ruta in rutas:
        if ruta:
            camino = [0] + ruta + [0]
            dist  += sum(matriz_distancias[camino[i], camino[i + 1]]
                         for i in range(len(camino) - 1))
    return dist

# ---------------- Configuración del GA (DEAP) ----------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individuo", tools.initIterate, creator.Individuo, generar_rutas_iniciales)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)
toolbox.register("evaluate", lambda ind: (costo_ruta(ind),))

def cruce(p1, p2):
    corte1, corte2 = sorted(random.sample(range(1, num_clientes), 2))
    seg1 = sum(p1, [])[corte1:corte2]
    seg2 = sum(p2, [])[corte1:corte2]

    def rellenar(padre, segmento):
        resto = [c for c in sum(padre, []) if c not in segmento]
        out = []
        while resto:
            ruta, carga, actual = [], 0, 0
            while resto and any(carga + demandas_clientes[c-1] <= capacidad_vehiculo for c in resto):
                # candidatos que caben
                cand = [c for c in resto if carga + demandas_clientes[c-1] <= capacidad_vehiculo]
                # sesgo también aquí para mantener la idea
                d = np.array([matriz_distancias[actual, c] for c in cand])
                inv = 1.0 / np.maximum(d, 1e-9)
                probs = (inv**2.0) / (inv**2.0).sum()
                elegido = cand[np.random.choice(len(cand), p=probs)]
                ruta.append(elegido)
                carga += demandas_clientes[elegido-1]
                resto.remove(elegido)
                actual = elegido
            out.append(ruta)
        return out

    return (creator.Individuo(rellenar(p1, seg1)),
            creator.Individuo(rellenar(p2, seg2)))

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

# ------------------------- Bucle GA --------------------------
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

# ------------------ Selección de la mejor --------------------
poblacion_ordenada = sorted(poblacion, key=lambda ind: ind.fitness.values)
for ind in poblacion_ordenada:
    cargas = [sum(demandas_clientes[c-1] for c in ruta) for ruta in ind]
    if all(c <= capacidad_vehiculo for c in cargas):
        mejor_solucion = ind
        mejor_costo    = costo_ruta(ind)
        break
else:
    mejor_solucion = poblacion_ordenada[0]
    mejor_costo    = costo_ruta(mejor_solucion)

# ------------------------ Visualización ----------------------
# plt.figure(figsize=(10, 6))
# ax = plt.gca()

# plt.scatter(ubicacion_deposito[0, 0], ubicacion_deposito[0, 1],
#             color='red', s=200, marker='s', label='Deposito')
# plt.scatter(ubicaciones_clientes[:, 0], ubicaciones_clientes[:, 1],
#             color='mediumblue', s=75, label='Clientes')

# colores = ['limegreen', 'orangered', 'darkviolet', 'dodgerblue']

# for idx, ruta in enumerate(mejor_solucion):
#     recorrido = [0] + ruta + [0]
#     col = colores[idx % len(colores)]
#     etiqueta = f'Vehiculo {idx + 1}'
#     for i in range(len(recorrido) - 1):
#         inicio = todas_ubicaciones[recorrido[i]]
#         fin    = todas_ubicaciones[recorrido[i + 1]]
#         dx, dy = fin[0] - inicio[0], fin[1] - inicio[1]
#         ax.arrow(inicio[0], inicio[1], dx, dy,
#                  head_width=2, head_length=3,
#                  fc=col, ec=col,
#                  length_includes_head=True, alpha=0.8,
#                  label=etiqueta if i == 0 else None)

# for idx, (x, y) in enumerate(ubicaciones_clientes, start=1):
#     ax.text(x, y+2, f'{idx} ({demandas_clientes[idx-1]})',
#             fontsize=9, ha='center')

# plt.title(f'Algoritmo Genetico para SCVRP (inicial por muestreo enfatizado)\nDistancia total: {mejor_costo:.2f}')
# plt.legend(loc='lower left')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# print('\nRutas optimas por vehiculo:')
# for i, ruta in enumerate(mejor_solucion, start=1):
#     carga = sum(demandas_clientes[c-1] for c in ruta)
#     print(f'Vehiculo {i} (carga {carga}/{capacidad_vehiculo}): Deposito -> {" -> ".join(map(str, ruta))} -> Deposito')
# print(f'\nDistancia total optimizada: {mejor_costo:.2f}')
