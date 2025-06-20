import time
import numpy as np
import random
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from deap import base, creator, tools

# MC
def run_mc(num_clientes, capacidad_vehiculo=200, num_sim=10000):
    demandas = np.random.randint(5, 15, size=num_clientes)
    rnd = np.random.RandomState(177)
    ubic = rnd.rand(num_clientes, 2) * 100
    deposito = np.array([[50, 50]])
    puntos = np.vstack([deposito, ubic])
    D = distance_matrix(puntos, puntos)

    def costo_ruta(rutas):
        d = 0.0
        for ruta in rutas:
            if ruta:
                d += D[0, ruta[0]]
                for i in range(len(ruta) - 1):
                    d += D[ruta[i], ruta[i+1]]
                d += D[ruta[-1], 0]
        return d

    start = time.perf_counter()
    best = float('inf')
    for _ in range(num_sim):
        ids = list(range(1, num_clientes+1))
        random.shuffle(ids)
        rutas = []
        while ids:
            carga, ruta, i = 0, [], 0
            while i < len(ids):
                c = ids[i]
                if carga + demandas[c-1] <= capacidad_vehiculo:
                    carga += demandas[c-1]
                    ruta.append(c)
                    ids.pop(i)
                else:
                    i += 1
            rutas.append(ruta)
        d = costo_ruta(rutas)
        if d < best:
            best = d
    end = time.perf_counter()
    return end - start

# GA
def run_ga(num_clientes, capacidad_vehiculo=200,
           poblacion_size=100, GENERACIONES=50,
           PROB_CRUCE=0.7, PROB_MUTACION=0.2):
    # Datos compartidos
    demandas_clientes = np.random.randint(5, 15, size=num_clientes)
    rnd = np.random.RandomState(177)
    ubicaciones_clientes = rnd.rand(num_clientes, 2) * 100
    ubicacion_deposito   = np.array([[50, 50]])
    todas_ubicaciones    = np.vstack([ubicacion_deposito, ubicaciones_clientes])
    matriz_distancias    = distance_matrix(todas_ubicaciones,
                                            todas_ubicaciones)

    # Rutas iniciales mediante Monte Carlo
    def generar_rutas_iniciales():
        clientes = list(range(1, num_clientes + 1))
        random.shuffle(clientes)
        rutas = []
        while clientes:
            ruta, carga = [], 0
            while (clientes and
                   carga + demandas_clientes[clientes[0] - 1]
                   <= capacidad_vehiculo):
                c = clientes.pop(0)
                ruta.append(c)
                carga += demandas_clientes[c - 1]
            rutas.append(ruta)
        return rutas

    # Función de coste
    def costo_ruta(rutas):
        dist = 0
        for ruta in rutas:
            if ruta:
                camino = [0] + ruta + [0]
                dist  += sum(
                    matriz_distancias[camino[i], camino[i+1]]
                    for i in range(len(camino)-1)
                )
        return dist

    # Configuración del Algoritmo Genético
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individuo", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individuo",
                     tools.initIterate,
                     creator.Individuo,
                     generar_rutas_iniciales)
    toolbox.register("poblacion",
                     tools.initRepeat,
                     list,
                     toolbox.individuo)

    def evaluar(ind): 
        return (costo_ruta(ind),)
    toolbox.register("evaluate", evaluar)

    # Operador de cruce
    def cruce(p1, p2):
        corte1, corte2 = sorted(random.sample(
            range(1, num_clientes), 2))
        seg1 = sum(p1, [])[corte1:corte2]
        seg2 = sum(p2, [])[corte1:corte2]

        def rellenar(padre, segmento):
            resto = [c for c in sum(padre, [])
                     if c not in segmento]
            out = []
            while resto:
                ruta, carga = [], 0
                while (resto and
                       carga + demandas_clientes[resto[0] - 1]
                       <= capacidad_vehiculo):
                    ruta.append(resto.pop(0))
                    carga += demandas_clientes[ruta[-1] - 1]
                out.append(ruta)
            return out

        return (creator.Individuo(rellenar(p1, seg1)),
                creator.Individuo(rellenar(p2, seg2)))

    # Operador de mutación
    def mutar(ind):
        if len(ind) > 1:
            r1, r2 = random.sample(range(len(ind)), 2)
            if ind[r1] and ind[r2]:
                c1 = random.choice(ind[r1])
                c2 = random.choice(ind[r2])
                ind[r1].remove(c1)
                ind[r2].remove(c2)
                ind[r1].append(c2)
                ind[r2].append(c1)
        return (ind,)

    toolbox.register("mate",   cruce)
    toolbox.register("mutate", mutar)
    toolbox.register("select",
                     tools.selTournament,
                     tournsize=3)

    # GA con los operadores de cruce y mutación
    poblacion = toolbox.poblacion(n=poblacion_size)
    start = time.perf_counter()
    for _ in range(GENERACIONES):
        descendencia = toolbox.select(poblacion,
                                      len(poblacion))
        descendencia = list(map(toolbox.clone,
                                descendencia))

        for h1, h2 in zip(descendencia[::2],
                          descendencia[1::2]):
            if random.random() < PROB_CRUCE:
                toolbox.mate(h1, h2)
                del h1.fitness.values
                del h2.fitness.values

        for mut in descendencia:
            if random.random() < PROB_MUTACION:
                toolbox.mutate(mut)
                del mut.fitness.values

        for ind in (i for i in descendencia
                    if not i.fitness.valid):
            ind.fitness.values = toolbox.evaluate(ind)

        poblacion[:] = descendencia
    
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

    end = time.perf_counter()

    # Limpiamos los tipos para poder llamar run_ga varias veces
    del creator.Individuo
    del creator.FitnessMin

    return end - start

# Main
def main():
    tamaños = [15, 30, 60, 120]
    reps = 10000
    mc_times = []
    ga_times = []

    for n in tamaños:
        tm = [run_mc(n) for _ in range(reps)]
        mc_times.append(np.mean(tm))
        tg = [run_ga(n) for _ in range(reps)]
        ga_times.append(np.mean(tg))
        print(f"Clientes={n}: MC≈{mc_times[-1]:.3f}s, GA≈{ga_times[-1]:.3f}s")

    plt.figure(figsize=(10,6))
    plt.plot(tamaños, mc_times, 'o--',  label='Monte Carlo',
             color='blue')
    plt.plot(tamaños, ga_times, 's--', label='Algoritmo Genético',
             color='red')
    plt.xlabel('Número de clientes')
    # # fijar ticks en el eje X de 15 en 15
    # max_x = max(tamaños)
    # plt.xticks(range(15, max_x+1, 15))
    plt.ylabel('Tiempo medio de ejecución (s)')
    plt.title('Monte Carlo vs Algoritmo Genético')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
