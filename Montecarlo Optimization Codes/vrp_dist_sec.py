#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import runpy
import numpy as np
import matplotlib.pyplot as plt

def medir_mc():
    """Ejecuta el script de Monte Carlo y devuelve la mejor distancia encontrada."""
    resultado = runpy.run_path("VRP mc time.py", run_name="__main__")
    return resultado["mejor_dist"]

def medir_ga():
    """Ejecuta el script de Algoritmo Genético y devuelve la mejor distancia encontrada."""
    resultado = runpy.run_path("VRP GA time.py", run_name="__main__")
    return resultado["mejor_costo"]

def recoger_datos(func_medir, n_iter=10000):
    """
    Ejecuta func_medir() n_iter veces,
    mide el tiempo y recoge la distancia devuelta.
    """
    tiempos = np.empty(n_iter)
    distancias = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter()
        dist = func_medir()
        t1 = time.perf_counter()
        tiempos[i] = t1 - t0
        distancias[i] = dist
    return tiempos, distancias

def main():
    n_iter = 1000
    # Recogemos datos
    mc_tiempos, mc_dists = recoger_datos(medir_mc, n_iter)
    ga_tiempos, ga_dists = recoger_datos(medir_ga, n_iter)

    # Calculamos medias
    mean_mc_tiempo = mc_tiempos.mean()
    mean_ga_tiempo = ga_tiempos.mean()
    mean_mc_dist   = mc_dists.mean()
    mean_ga_dist   = ga_dists.mean()

    # Representación gráfica: Distancia vs Tiempo
    plt.figure(figsize=(10, 6))

    plt.scatter(mc_dists, mc_tiempos, color="blue", s=5, alpha=0.3, label="Monte Carlo")
    plt.scatter(ga_dists, ga_tiempos, color="red",  s=5, alpha=0.3, label="Genético (GA)")


    
    # Líneas horizontales en la media de tiempos
    plt.axhline(mean_mc_tiempo, color="blue", linestyle=":",
                label=f"MC tiempo medio: {mean_mc_tiempo:.3f}s")
    plt.axhline(mean_ga_tiempo, color="red",  linestyle=":",
                label=f"GA tiempo medio: {mean_ga_tiempo:.3f}s")

    # Líneas verticales en la media de distancias
    plt.axvline(mean_mc_dist, color="blue", linestyle=":",
                label=f"MC distancia media: {mean_mc_dist:.2f}")
    plt.axvline(mean_ga_dist, color="red",  linestyle=":",
                label=f"GA distancia media: {mean_ga_dist:.2f}")

    plt.xlabel("Distancia total optimizada")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("Comparación Monte Carlo vs Algoritmo Genético\n\
Clientes: 120; Vehículos: 8; Capacidad: 200")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
