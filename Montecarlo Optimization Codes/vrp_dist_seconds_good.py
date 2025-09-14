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
    return res["mejor_dist"]  # el script debe definir esta variable

def medir_ga():
    """Ejecuta vrp_ga.py (Algoritmo Genético) y devuelve la mejor distancia."""
    res = runpy.run_path("vrp_ga.py", run_name="__main__")
    return res["mejor_costo"]  # el script debe definir esta variable

def medir_ga_is():
    """Ejecuta vrp_ga_is.py (GA con muestreo enfatizado) y devuelve la mejor distancia."""
    res = runpy.run_path("vrp_ga_is.py", run_name="__main__")
    return res["mejor_costo"]  # el script debe definir esta variable

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

    # Medias (se calculan sobre TODOS los puntos, también los que luego se filtrarán)
    mean_mc_t,  mean_mc_d  = mc_tiempos.mean(),  mc_dists.mean()
    mean_ga_t,  mean_ga_d  = ga_tiempos.mean(),  ga_dists.mean()
    mean_gai_t, mean_gai_d = gai_tiempos.mean(), gai_dists.mean()

    print(f"MC IS: {mean_mc_d:.2f};  {mean_mc_t:.3f}s")
    print(f"GA:    {mean_ga_d:.2f};  {mean_ga_t:.3f}s")
    print(f"GA+IS: {mean_gai_d:.2f}; {mean_gai_t:.3f}s")

    # ---------------------- Filtrado por 6× media de TIEMPO ----------------------
    thr_mc  = 3.0 * mean_mc_t
    thr_ga  = 3.0 * mean_ga_t
    thr_gai = 3.0 * mean_gai_t

    mask_mc  = mc_tiempos  <= thr_mc
    mask_ga  = ga_tiempos  <= thr_ga
    mask_gai = gai_tiempos <= thr_gai

    excl_mc  = (~mask_mc).sum()
    excl_ga  = (~mask_ga).sum()
    excl_gai = (~mask_gai).sum()

    # Informes por pantalla
    print(f"[FILTRADO] MC IS  -> umbral tiempo = {thr_mc:.3f}s | excluidos: {excl_mc} de {n_iter}")
    print(f"[FILTRADO] GA     -> umbral tiempo = {thr_ga:.3f}s | excluidos: {excl_ga} de {n_iter}")
    print(f"[FILTRADO] GA+IS  -> umbral tiempo = {thr_gai:.3f}s | excluidos: {excl_gai} de {n_iter}")

    # --- Gráfica Distancia vs Tiempo (con filtrado aplicado) ---
    plt.figure(figsize=(10, 6))

    plt.scatter(mc_dists[mask_mc],  mc_tiempos[mask_mc],
                color="blue",  s=5, alpha=0.50, label="Monte-Carlo IS (MC IS)")
    plt.scatter(ga_dists[mask_ga],  ga_tiempos[mask_ga],
                color="red",   s=5, alpha=0.50, label="Algoritmo Genético (GA)")
    plt.scatter(gai_dists[mask_gai], gai_tiempos[mask_gai],
                color="green", s=5, alpha=0.50, label="Genético+IS (GA+IS)")

    plt.xlabel("Distancia")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title("MC IS vs GA vs GA+IS \nClientes: 60; Vehículos: 8; Capacidad: 200")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
