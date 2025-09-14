# !/usr/bin/env python3
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
    n_iter = 100  # ajusta si quieres más/menos repeticiones

    # Recogida de datos
    mc_tiempos,  mc_dists  = recoger_datos(medir_mc,  n_iter)
    ga_tiempos,  ga_dists  = recoger_datos(medir_ga,  n_iter)
    gai_tiempos, gai_dists = recoger_datos(medir_ga_is, n_iter)

    # --- Cálculo de estadísticas descriptivas (media, cuartiles, desviación típica) ---
    # Utilizaremos la desviación típica porque es más representativa que la varianza.
    # La desviación típica está en la misma unidad que la media (por ejemplo, metros),
    # mientras que la varianza está en unidades cuadradas (metros cuadrados).

    # Para Monte Carlo
    mc_stats_dist = {
        "media": mc_dists.mean(),
        "desviacion_tipica": mc_dists.std(),
        "cuartiles": np.percentile(mc_dists, [25, 50, 75])
    }
    mc_stats_tiempo = {
        "media": mc_tiempos.mean(),
        "desviacion_tipica": mc_tiempos.std(),
        "cuartiles": np.percentile(mc_tiempos, [25, 50, 75])
    }

    # Para Algoritmo Genético
    ga_stats_dist = {
        "media": ga_dists.mean(),
        "desviacion_tipica": ga_dists.std(),
        "cuartiles": np.percentile(ga_dists, [25, 50, 75])
    }
    ga_stats_tiempo = {
        "media": ga_tiempos.mean(),
        "desviacion_tipica": ga_tiempos.std(),
        "cuartiles": np.percentile(ga_tiempos, [25, 50, 75])
    }
    
    # Para Algoritmo Genético + IS
    gai_stats_dist = {
        "media": gai_dists.mean(),
        "desviacion_tipica": gai_dists.std(),
        "cuartiles": np.percentile(gai_dists, [25, 50, 75])
    }
    gai_stats_tiempo = {
        "media": gai_tiempos.mean(),
        "desviacion_tipica": gai_tiempos.std(),
        "cuartiles": np.percentile(gai_tiempos, [25, 50, 75])
    }

    # --- Impresión de los resultados ---
    print("\n--- Estadísticas de Distancia ---")
    print(f"MC IS: Media={mc_stats_dist['media']:.2f}, Desviación={mc_stats_dist['desviacion_tipica']:.2f}, Cuartiles={mc_stats_dist['cuartiles']}")
    print(f"GA:    Media={ga_stats_dist['media']:.2f}, Desviación={ga_stats_dist['desviacion_tipica']:.2f}, Cuartiles={ga_stats_dist['cuartiles']}")
    print(f"GA+IS: Media={gai_stats_dist['media']:.2f}, Desviación={gai_stats_dist['desviacion_tipica']:.2f}, Cuartiles={gai_stats_dist['cuartiles']}")

    print("\n--- Estadísticas de Tiempo (s) ---")
    print(f"MC IS: Media={mc_stats_tiempo['media']:.3f}, Desviación={mc_stats_tiempo['desviacion_tipica']:.3f}, Cuartiles={mc_stats_tiempo['cuartiles']}")
    print(f"GA:    Media={ga_stats_tiempo['media']:.3f}, Desviación={ga_stats_tiempo['desviacion_tipica']:.3f}, Cuartiles={ga_stats_tiempo['cuartiles']}")
    print(f"GA+IS: Media={gai_stats_tiempo['media']:.3f}, Desviación={gai_stats_tiempo['desviacion_tipica']:.3f}, Cuartiles={gai_stats_tiempo['cuartiles']}")

    # ---------------------- Filtrado por 3× media de TIEMPO ----------------------
    thr_mc  = 3.0 * mc_stats_tiempo['media']
    thr_ga  = 3.0 * ga_stats_tiempo['media']
    thr_gai = 3.0 * gai_stats_tiempo['media']

    mask_mc  = mc_tiempos  <= thr_mc
    mask_ga  = ga_tiempos  <= thr_ga
    mask_gai = gai_tiempos <= thr_gai

    excl_mc  = (~mask_mc).sum()
    excl_ga  = (~mask_ga).sum()
    excl_gai = (~mask_gai).sum()

    # Informes por pantalla
    print(f"\n[FILTRADO] MC IS  -> umbral tiempo = {thr_mc:.3f}s | excluidos: {excl_mc} de {n_iter}")
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
    plt.title("MC IS vs GA vs GA+IS \nClientes: 120; Vehículos: 8; Capacidad: 200")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()