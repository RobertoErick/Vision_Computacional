import numpy as np
import csv

def calcular_matriz_distancias(resultados_path, archivo_csv="matriz_distancias.csv", longitud_deseada=1500):
    # Cargar los resultados almacenados
    resultados = np.load(resultados_path, allow_pickle=True).item()

    # Obtener los nombres de las plantas
    plantas = list(resultados.keys())
    n = len(plantas)

    # Crear una matriz vacía para almacenar las distancias
    matriz_distancias = np.zeros((n, n))

    # Normalizar todos los perfiles a la misma longitud
    def normalizar_perfil(perfil, longitud_deseada):
        x_original = np.linspace(0, 1, len(perfil))
        x_deseado = np.linspace(0, 1, longitud_deseada)
        return np.interp(x_deseado, x_original, perfil)

    # Comparar cada planta con todas las demás
    for i, planta_x in enumerate(plantas):
        for j, planta_y in enumerate(plantas):
            # Mostrar el progreso de la comparación en consola
            print(f"Comparando {planta_x} con {planta_y}...")

            # Si estamos en la diagonal, la distancia es 0 (misma planta)
            if i == j:
                matriz_distancias[i, j] = 0
                continue

            # Obtener los perfiles de ambas plantas
            perfiles_x = resultados[planta_x]
            perfiles_y = resultados[planta_y]

            # Calcular la mejor distancia entre todos los perfiles de las dos plantas
            mejor_distancia = float('inf')
            for perfil_x in perfiles_x:
                for perfil_y in perfiles_y:
                    # Normalizar ambos perfiles
                    perfil_x_normalizado = normalizar_perfil(perfil_x["perfil_avp"], longitud_deseada)
                    perfil_y_normalizado = normalizar_perfil(perfil_y["perfil_avp"], longitud_deseada)

                    # Calcular la distancia euclidiana
                    distancia = np.sqrt(np.sum((perfil_x_normalizado - perfil_y_normalizado) ** 2))
                    mejor_distancia = min(mejor_distancia, distancia)

            matriz_distancias[i, j] = mejor_distancia

    # Guardar la matriz en un archivo CSV
    with open(archivo_csv, mode="w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        # Escribir los encabezados (nombres de las plantas)
        escritor.writerow([""] + plantas)
        # Escribir las filas de la matriz
        for i, planta in enumerate(plantas):
            escritor.writerow([planta] + list(matriz_distancias[i]))

    print(f"Matriz de distancias guardada en: {archivo_csv}")

# Ruta del archivo de resultados
resultados_path = "resultados_avp_dct.npy"

# Generar y guardar la matriz de distancias
calcular_matriz_distancias(resultados_path, archivo_csv="matriz_distancias.csv")
