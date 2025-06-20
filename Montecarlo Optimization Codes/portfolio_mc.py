import numpy as np
import matplotlib.pyplot as plt

# Retornos simulados de activos (retorno medio anual, volatlidad)
# Desviacion estandar = desviacion estandar
activos = {
    "Accion A": (0.12, 0.25),
    "Accion B": (0.08, 0.15),
    "Accion C": (0.10, 0.20),
    "Bono D": (0.05, 0.10)
}

# Numero de carteras aleatorias a generar
num_carteras = 10000
tasa_libre_riesgo = 0.02  # 2% de tasa libre de riesgo

# Extraer datos
nombres_activos = list(activos.keys())
retornos_medios = np.array([activos[a][0] for a in nombres_activos])
desviaciones = np.array([activos[a][1] for a in nombres_activos])
matriz_covarianza = np.diag(desviaciones**2)  # Simplificado: no existe correlacion

# Simulacion Monte Carlo
retornos_cartera = []
riesgos_cartera = []
pesos_cartera = []
ratios_sharpe = []

for _ in range(num_carteras):
    pesos = np.random.random(len(activos))
    pesos /= np.sum(pesos)  # Normalizacion por pesos

    retorno_port = np.sum(pesos * retornos_medios)
    riesgo_port = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
    ratio_sharpe = (retorno_port - tasa_libre_riesgo) / riesgo_port

    retornos_cartera.append(retorno_port)
    riesgos_cartera.append(riesgo_port)
    pesos_cartera.append(pesos)
    ratios_sharpe.append(ratio_sharpe)

# Buscar la cartera optima (maximo Sharpe)
indice_optimo = np.argmax(ratios_sharpe)
cartera_optima = pesos_cartera[indice_optimo]
retorno_optimo = retornos_cartera[indice_optimo]
riesgo_optimo = riesgos_cartera[indice_optimo]

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.scatter(riesgos_cartera, retornos_cartera, c=ratios_sharpe, cmap='viridis', alpha=0.5)
plt.colorbar(label='Ratio de Sharpe')
plt.scatter(riesgo_optimo, retorno_optimo, color='red', marker='*', s=200, label='Cartera Optima')
plt.xlabel('Riesgo (Desviacion estandar)')
plt.ylabel('Retorno esperado')
plt.title(f'Optimizacion de cartera con Monte Carlo\n \
          Carteras: {num_carteras}; Activos: {len(activos)}; Tasa libre de riesgo: {tasa_libre_riesgo}')
plt.legend()
plt.show()

# Imprimir resultados
print("Pesos de la cartera optima:")
for i, activo in enumerate(nombres_activos):
    print(f"{activo}: {cartera_optima[i]:.2%}")
print(f"Retorno esperado: {retorno_optimo:.2%}, Riesgo: {riesgo_optimo:.2%}")
