#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import runpy
import numpy as np
import matplotlib.pyplot as plt

# --- Wrappers que ejecutan cada script y devuelven su mejor distancia ---
def medir_mc():
    """Ejecuta vrp_is.py (Monte Carlo) y devuelve la mejor distancia encontrada."""
    res = runpy.run_path("vrp_is.py", run_name="__main__")
    return res["mejor_dist"]                     # el script debe definir esta variable

def medir_ga():
    """Ejecuta vrp_ga.py (Algoritmo Genético) y devuelve la mejor distancia."""
    res = runpy.run_path("vrp_ga.py", run_name="__main__")
    return res["mejor_costo"]                    # el script debe definir esta variable

def medir_ga_is():
    """Ejecuta vrp_ga_is.py (GA con muestreo enfatizado) y devuelve la mejor distancia."""
    res = runpy.run_path("vrp_ga_is.py", run_name="__main__")
    return res["mejor_costo"]                    # el script debe definir esta variable

# --- Utilidad para medir tiempos y recoger distancias n_iter veces ---
def recoger_datos(func_medir, n_iter=1000):
    tiempos = np.empty(n_iter, dtype=float)
    distancias = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        t0 = time.perf_counter()
        dist = func_medir()
        t1 = time.perf_counter()
        tiempos[i] = t1 - t0
        distancias[i] = dist
    return tiempos, distancias

def main():
    n_iter = 1000  # ajusta si quieres más/menos repeticiones

    # Recogida de datos
    mc_tiempos,  mc_dists  = recoger_datos(medir_mc,  n_iter)
    ga_tiempos,  ga_dists  = recoger_datos(medir_ga,  n_iter)
    gai_tiempos, gai_dists = recoger_datos(medir_ga_is, n_iter)

    # Medias
    mean_mc_t,  mean_mc_d  = mc_tiempos.mean(),  mc_dists.mean()
    mean_ga_t,  mean_ga_d  = ga_tiempos.mean(),  ga_dists.mean()
    mean_gai_t, mean_gai_d = gai_tiempos.mean(), gai_dists.mean()
    
    
    print(f"MC IS: {mean_mc_d:.2f};  {mean_mc_t:.3f}s")
    print(f"GA:    {mean_ga_d:.2f};  {mean_ga_t:.3f}s")
    print(f"GA+IS: {mean_gai_d:.2f}; {mean_gai_t:.3f}s")
    # --- Gráfica Distancia vs Tiempo ---
    plt.figure(figsize=(10, 6))

    plt.scatter(mc_dists,  mc_tiempos,  color="blue",  s=5, alpha=0.50, label="Monte-Carlo IS (MC IS)")
    plt.scatter(ga_dists,  ga_tiempos,  color="red",   s=5, alpha=0.50, label="Algoritmo Genético (GA)")
    plt.scatter(gai_dists, gai_tiempos, color="green", s=5, alpha=0.50, label="Genético+IS (GA+IS)")

    # # Líneas horizontales (tiempos medios)
    # plt.axhline(mean_mc_t,  color="blue",  linestyle=":", label=f"MC IS tiempo medio: {mean_mc_t:.3f}s")
    # plt.axhline(mean_ga_t,  color="red",   linestyle=":", label=f"GA tiempo medio: {mean_ga_t:.3f}s")
    # plt.axhline(mean_gai_t, color="green", linestyle=":", label=f"GA+IS tiempo medio: {mean_gai_t:.3f}s")

    # # Líneas verticales (distancias medias)
    # plt.axvline(mean_mc_d,  color="blue",  linestyle=":", label=f"MC IS distancia media: {mean_mc_d:.2f}")
    # plt.axvline(mean_ga_d,  color="red",   linestyle=":", label=f"GA distancia media: {mean_ga_d:.2f}")
    # plt.axvline(mean_gai_d, color="green", linestyle=":", label=f"GA+IS distancia media: {mean_gai_d:.2f}")

    plt.xlabel("Distancia")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("MC IS vs GA vs GA+IS\nClientes: 15; Vehículos: 3; Capacidad: 50")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
