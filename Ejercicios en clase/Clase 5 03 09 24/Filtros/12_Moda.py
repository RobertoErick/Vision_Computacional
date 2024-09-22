import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fundion para crea la matriz de dispersion de la imagen original en escala de grises
def crear_matriz_dispersion(imagen, promedio):
    nueva_matriz_de_dispersion = imagen.copy()

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            
            nueva_matriz_de_dispersion[i][j] = abs(imagen[i,j] - promedio) * 100 / promedio

    return nueva_matriz_de_dispersion

# Funcion para crear la matriz de la imagen original en escala de grises
def histograma_imagen(imagen):
    plt.hist(imagen.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)

    plt.title('Histograma de Barras - Imagen Original')
    plt.xlabel('Valor de Intensidad (0-255)')
    plt.ylabel('Número de Píxeles')

    plt.show()
 
# Función para crear la matriz de la dispersion de los pixeles
def histograma_matriz_dispersion(matriz_de_dispersion):

    # Histograma de la matriz de dispersion
    plt.hist(matriz_de_dispersion.ravel(), bins=100, range=[0, 100], color='gray', alpha=0.7)

    plt.suptitle('Histograma de Barras - matriz de dispersion mediana')
    plt.title('A continuacion, ingresa el valor del porcentaje para ser reemplazadio por la mediana: %i' %moda)
    plt.xlabel('Valor de Intensidad (0-255)')
    plt.ylabel('Número de Píxeles')

    plt.show()

# Funciones para crear las modas de las imagenes
def imagen_filtrada():
    for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                if matriz_de_dispersion[i][j] >= filtro :
                    imagen_filtrada_con_moda[i][j] = moda

def imagen_filtrada_multimoda():
    modas_matrices = []
    modas_sumas = []
    
    # Crear una imagen filtrada por cada moda
    for moda in modas:
        imagen_temp = np.zeros_like(imagen)  # Crear una copia vacía de la imagen
        
        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                if matriz_de_dispersion[i][j] >= filtro:
                    imagen_temp[i][j] = moda

        # Calcular la suma de la matriz filtrada
        suma_matriz = np.sum(imagen_temp)
        modas_matrices.append(imagen_temp)
        modas_sumas.append(suma_matriz)

    # Encontrar la moda que genere la suma más pequeña
    indice_menor_suma = np.argmin(modas_sumas)
    imagen_filtrada_con_moda = modas_matrices[indice_menor_suma]
    
    print(f"La moda seleccionada es {modas[indice_menor_suma]} con suma {modas_sumas[indice_menor_suma]}")

    return imagen_filtrada_con_moda


# Cargar la imagen en escala de grises
imagen_color = cv2.imread("imagen_color_ruido_sal.png")
if imagen_color is None:
    print("Error: No se pudo cargar la imagen")
else:
    # Convertir la imagen a escala de grises
    imagen = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # Crear diferentes imagenes para mostrar el resultado
    imagen_filtrada_con_moda = imagen.copy()
    matriz_de_dispersion = imagen.copy()

    # Promedio de la imagen original
    promedio = np.mean(imagen)

    # Los siguientes pasos son para obtener la moda o multiples modas
    # Aplanar la imagen recortada
    valores, conteos = np.unique(imagen.flatten(), return_counts=True)

    # Obtener el valor máximo de frecuencia
    max_frecuencia = np.max(conteos)

    # Obtener todas las modas (valores con la máxima frecuencia)
    modas = valores[conteos == max_frecuencia]

    # Si hay más de una moda
    if len(modas) > 1:
        print(f"Hay múltiples modas: {modas}")
    else:
        print(f"La moda es: {modas[0]}")
        moda = modas[0]
    # Hasta aqui sabemos las modas o moda que se tiene en la imagen

    # Creacion de la matriz de dispersion
    matriz_de_dispersion = crear_matriz_dispersion(imagen, promedio)
 
    # Histograma de la imagen original en escala de grises
    histograma_imagen(imagen)

    # Crear el histograma de la matriz dispersion con informacion para el usuario
    histograma_matriz_dispersion(matriz_de_dispersion)

    # Filtro que se le va a colocar a la imagen (a partir de qué número de la matriz de dispersion va a reemplazar los valores en la imagen)
    print("Mediana de la imagen (valor por el que va a cambiar los pixeles seleccionados): ", moda)
    print("Valor recomendado con las imagenes de prueba: 97")
    filtro = int(input("Introduce el filtro que quieres colocar (0 - 100): "))

    # Reemplazar los valores por la moda
    if len(modas) > 1:
        # Si hay mas de una moda usamos un mulltimodal
        imagen_filtrada_multimoda()
    else:
        # Si solo es una moda procedemos normal
        imagen_filtrada()

    # Promedio de la imagen filtrada resultante
    promedio_final = np.mean(imagen_filtrada_con_moda)

    # Obtener la matriz de dispersion resultante
    matriz_de_dispersion_resultante = crear_matriz_dispersion(imagen_filtrada_con_moda, promedio_final)

    # Dispersion de la matriz resultante final
    print("El valor final de la sumatoria de la matriz resultante es la siguiente: ", np.sum(matriz_de_dispersion_resultante))

    np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
    np.savetxt('matriz_de_dispersion_moda.csv', matriz_de_dispersion, delimiter=',', fmt='%d') 
    np.savetxt('imagen_filtrada_con_moda.csv', imagen_filtrada_con_moda, delimiter=',', fmt='%d')
    np.savetxt('matriz_de_dispersion_moda_resultante.csv', matriz_de_dispersion_resultante, delimiter=',', fmt='%d')

    # Mostrar la imagen recortada
    cv2.imshow("Imagen original", imagen)
    cv2.imshow("matriz de dispersion moda", matriz_de_dispersion)
    cv2.imshow("imagen filtrada con moda", imagen_filtrada_con_moda)
    cv2.imshow("Matriz de dispersion resultante", matriz_de_dispersion_resultante)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen en los archivos
    cv2.imwrite("imagen_filtrada_con_moda.png", imagen_filtrada_con_moda)    