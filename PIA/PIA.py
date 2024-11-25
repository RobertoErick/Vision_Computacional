import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

# -------------------------------------Seccion Recortes de imagen------------------------------------------- #

# Funcion para obtener los puntos extremos de la hoja (Puntos A, B C y D)
# Se obtienen 2 veces, antes de ser rotados y despues de ser rotado
def extremos(binary_image):
    non_zero_pixels = np.nonzero(binary_image)

    # Encontrar los límites de recorte (A, C, B, D)
    A = np.min(non_zero_pixels[0])  # Límite superior (primer píxel no cero en filas)
    C = np.max(non_zero_pixels[0])  # Límite inferior (último píxel no cero en filas)
    B = np.min(non_zero_pixels[1])  # Límite izquierdo (primer píxel no cero en columnas)
    D = np.max(non_zero_pixels[1])  # Límite derecho (último píxel no cero en columnas)

    print("A: ",A,"\nB: ",B,"\nC: ",C,"\nD: ",D)

    return A, B, C, D

# ---------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------Seccion DCT------------------------------------------------- #

# Calcula el coeficiente DCTk para un perfil dado
def calcular_dctk(perfil, k):
    # perfil: perfil AVP antes de ser procesado
    # k: indice de coeficiente DCT a calcular

    N = len(perfil)  # Longitud del perfil
    dctk = 0.0  # Inicializar el coeficiente

    for n in range(N):
        dctk += perfil[n] * np.cos((np.pi / N) * (n + 0.5) * k)

    return dctk

# Calcula la IDCTi para reconstruir un valor a partir de los coeficientes DCT.
def calcular_idcti(dct_coeffs, i):
    # dct_coeffs: Coeficientes DCT
    # i: El índice del valor reconstruido.

    N = len(dct_coeffs)  # Longitud de los coeficientes DCT
    idcti = dct_coeffs[0] / 2  # Primer término (DCT_0 / 2)

    for k in range(1, N):  # Desde DCT_1 hasta DCT_{N-1}
        idcti += dct_coeffs[k] * np.cos((np.pi / N) * k * (i + 0.5))

    return idcti / N  # Dividir por N para obtener el valor final

# Reconstruye el perfil completo usando la IDCT.
def reconstruir_perfil_completo(dct_coeffs):
    # dct_coeffs: Los coeficientes DCT.

    N = len(dct_coeffs)  # Longitud del perfil (número de coeficientes DCT)
    perfil_reconstruido = []

    for i in range(N):  # Iterar sobre todos los índices
        idcti = calcular_idcti(dct_coeffs, i)  # Calcular IDCTi para el índice actual
        perfil_reconstruido.append(idcti)  # Agregar el valor reconstruido

    return perfil_reconstruido

# Calcula el error EDCT entre el perfil original y el reconstruido.
def calcular_edct(perfil_original, perfil_reconstruido):
    # perfil_original: El perfil original.
    # perfil_reconstruido: El perfil reconstruido.

    N = len(perfil_original)  # Longitud del perfil
    error_cuadrado = np.sum((np.array(perfil_original) - np.array(perfil_reconstruido))**2)
    edct = np.sqrt(error_cuadrado / N)
    return edct

# ---------------------------------------------------------------------------------------------------------- #
# ---------------------------------------Seccion AVP-DCT Matching------------------------------------------- #

# Normaliza la longitud de la curva AVP-DCT, esto para que tenga la misma longitud todos los perfiles (dataset y la imagen a comparar)
def normalizar_longitud(perfil, longitud_deseada=1500):
    # perfil: Perfil AVP-DCT Orginal de la imagen a comparar
    # longitud_deseada: Longitud que va a tener el perfil normalizado 

    x_original = np.linspace(0, 1, len(perfil))
    x_deseado = np.linspace(0, 1, longitud_deseada)
    perfil_normalizado = np.interp(x_deseado, x_original, perfil)

    return perfil_normalizado

# Distancia entre un perfil AVP-DCT de la imagen a comparar y el dataset
def calcular_distancia_euclidiana(perfil1, perfil2):
    return np.sqrt(np.sum((np.array(perfil1) - np.array(perfil2))**2))

# Guarda las mejores distancias por planta en un archivo .csv.
def guardar_mejores_distancias_csv(distancias, archivo_csv="mejores_distancias.csv"):
    # distancias: Lista de tuplas (distancia, planta).
    # archivo_csv: Nombre del archivo .csv donde se guardarán los datos.

    # Agrupar por planta y obtener la distancia mínima para cada planta
    mejores_distancias = {}
    for distancia, planta in distancias:
        if planta not in mejores_distancias or distancia < mejores_distancias[planta]:
            mejores_distancias[planta] = distancia

    # Guardar en un archivo CSV
    with open(archivo_csv, mode="w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["Planta", "Mejor Distancia Euclidiana"])
        for planta, distancia in mejores_distancias.items():
            escritor.writerow([planta, distancia])

    print(f"Mejores distancias guardadas en el archivo: {archivo_csv}")

def mostrar_mejores_distancias_por_planta(distancias):
    """
    Muestra un diagrama de barras con las mejores distancias para cada planta.

    Args:
        distancias (list): Lista de tuplas (distancia, planta).
    """
    # Agrupar por planta y obtener la distancia mínima para cada planta
    mejores_distancias = {}
    for distancia, planta in distancias:
        if planta not in mejores_distancias or distancia < mejores_distancias[planta]:
            mejores_distancias[planta] = distancia

    # Separar nombres de plantas y sus distancias
    plantas = list(mejores_distancias.keys())
    distancias_minimas = list(mejores_distancias.values())

    # Crear el gráfico de barras
    plt.figure(figsize=(12, 6))
    plt.bar(plantas, distancias_minimas, color='mediumpurple')
    plt.xlabel("Planta")
    plt.ylabel("Distancia Euclidiana")
    plt.title("Mejores Distancias por Planta")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def avp_matching(perfil_reconstruido, resultados_path, k=3, longitud_deseada=1500):
    """
    Realiza el AVP Matching utilizando k-NN y genera un gráfico de las mejores distancias por planta.
    """
    perfil_nuevo_normalizado = normalizar_longitud(perfil_reconstruido, longitud_deseada)

    # Cargar los resultados almacenados
    resultados = np.load(resultados_path, allow_pickle=True).item()

    distancias = []
    for planta, perfiles in resultados.items():
        for perfil_guardado in perfiles:
            perfil_guardado_normalizado = normalizar_longitud(perfil_guardado["perfil_avp"], longitud_deseada)
            distancia = calcular_distancia_euclidiana(perfil_nuevo_normalizado, perfil_guardado_normalizado)
            distancias.append((distancia, planta))

    distancias.sort(key=lambda x: x[0])

    # Mostrar mejores distancias por planta
    mostrar_mejores_distancias_por_planta(distancias)

    # Seleccionar los k vecinos más cercanos
    vecinos = distancias[:k]

    # Contar la frecuencia de cada planta entre los vecinos
    conteo = {}
    for _, planta in vecinos:
        conteo[planta] = conteo.get(planta, 0) + 1

    planta_probable = max(conteo, key=conteo.get)
    return planta_probable

# ---------------------------------------------------------------------------------------------------------- #
# ----------------------------------------Seccion Valle Global------------------------------------------------- #

def calcular_histograma(imagen):
    # Calcula el histograma de la imagen en escala de grises
    histograma, _ = np.histogram(imagen.ravel(), bins=256, range=(0, 256))
    return histograma

def calcular_grupo_varianza(histograma, total_pixeles):
    # Cálculo de varianza entre los grupos 
    suma_total = np.sum([i * histograma[i] for i in range(256)])
    suma_b = 0
    w_b = 0
    varianza_max = 0
    mejor_umbral = 0

    for umbral in range(256):
        w_b += histograma[umbral]
        w_f = total_pixeles - w_b
        if w_b == 0 or w_f == 0:
            continue
        
        suma_b += umbral * histograma[umbral]
        m_b = suma_b / w_b if w_b != 0 else 0
        m_f = (suma_total - suma_b) / w_f if w_f != 0 else 0

        varianza_entre_clases = w_b * w_f * (m_b - m_f) ** 2

        if varianza_entre_clases > varianza_max:
            varianza_max = varianza_entre_clases
            mejor_umbral = umbral

    return mejor_umbral

def calcular_desviacion_estandar(imagen):
    # Cálculo de la desviación estándar
    return np.std(imagen)

def encontrar_picos(histograma):
    # Encontrar picos en el histograma (máximos locales)
    picos = []
    for i in range(1, len(histograma) - 1):
        if histograma[i] > histograma[i - 1] and histograma[i] > histograma[i + 1]:
            picos.append(i)
    return picos

def calcular_valle_global(imagen):
    # Obtener el histograma
    histograma = calcular_histograma(imagen)
    total_pixeles = imagen.size
    
    # Calcular el umbral utilizando grupo varianza
    mejor_umbral = calcular_grupo_varianza(histograma, total_pixeles)
    
    # Calcular la desviación estándar
    desviacion_estandar = calcular_desviacion_estandar(imagen)

    # Encontrar los picos del histograma
    picos = encontrar_picos(histograma)
    
    return mejor_umbral, desviacion_estandar, picos

def aplicar_umbral(imagen, umbral):
    # Binarización con el umbral calculado
    _, imagen_umbralizada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    return imagen_umbralizada

# ---------------------------------------------------------------------------------------------------------- #
# ----------------------------------------Seccion Principal------------------------------------------------- #

# Cargar dataset anteriormente procesado y verificar contenido
resultados_path = 'resultados_avp_dct.npy'  # Archivo de resultados
if not resultados_path:
    raise ValueError("El diccionario está vacío. Verifica la ruta")

# Cargar la imagen y verificar si se cargó correctamente
image = cv2.imread('amabis.jpg')  # Imagen a predecir
if image is None:
    raise FileNotFoundError("La imagen no se pudo cargar. Verifica la ruta.")

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imagen = (gray_image * 255).astype(np.uint8)  # Normalizar la imagen a valores de 0 a 255

# Calcular el umbral, desviación estándar y los picos
mejor_umbral, desviacion_estandar, picos = calcular_valle_global(imagen)

# Aplicar el umbral a la imagen
binary_image = aplicar_umbral(imagen, mejor_umbral)

# Mostrar la imagen original y la imagen umbralizada
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title(f'Imagen umbralizada (Umbral: {mejor_umbral})')

# Mostrar el histograma de la imagen
plt.subplot(1, 3, 3)
histograma = calcular_histograma(imagen)
plt.plot(histograma, color='black')
plt.title('Histograma')
plt.xlabel('Intensidad de píxel')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Usamos np.nonzero para encontrar los píxeles que no son fondo (es decir, píxeles diferentes de 0)
non_zero_pixels = np.nonzero(binary_image)

# Obtenemos los datos de los extremos antes de ser rotados
A, B, C, D = extremos(binary_image)

# Se obtiene la coordenada de la fila en B y D
y_B = np.min(non_zero_pixels[0][non_zero_pixels[1] == B])
y_D = np.min(non_zero_pixels[0][non_zero_pixels[1] == D])

# Calcular el ángulo necesario para alinear B y D
delta_y = y_D - y_B
delta_x = D - B
angle = math.degrees(math.atan2(delta_y, delta_x))

print(f"Ángulo calculado para alinear B y D: {angle:.2f} grados")

# Rotar la imagen usando el ángulo calculado
center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
binary_image_rotated = cv2.warpAffine(binary_image, M, (binary_image.shape[1], binary_image.shape[0]))

# Dibujar puntos B y D antes y después de la rotación
image_with_points = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
rotated_with_points = cv2.cvtColor(binary_image_rotated, cv2.COLOR_GRAY2BGR)

cv2.circle(image_with_points, (B, y_B), 5, (0, 0, 255), -1)  # Punto B antes de rotación
cv2.circle(image_with_points, (D, y_D), 5, (255, 0, 0), -1)  # Punto D antes de rotación

# Transformar coordenadas después de la rotación
B_rot = np.dot(M, np.array([B, y_B, 1]))
D_rot = np.dot(M, np.array([D, y_D, 1]))

cv2.circle(rotated_with_points, (int(B_rot[0]), int(B_rot[1])), 5, (0, 0, 255), -1)  # Punto B después de rotación
cv2.circle(rotated_with_points, (int(D_rot[0]), int(D_rot[1])), 5, (255, 0, 0), -1)  # Punto D después de rotación

# Mostrar imágenes
cv2.imshow('Imagen Original con Puntos', image_with_points)
cv2.imshow('Imagen Rotada con Puntos', rotated_with_points)

# Se vuelve a obtener los extremos de la imagen para su recorte
A, B, C, D = extremos(binary_image_rotated)

# Recortar la imagen original utilizando los límites encontrados
cropped_image = binary_image_rotated[A:C+1, B:D+1]

# Crear el histograma de la proyección Top (de B a D)
top_projection = []

# Recorremos de B a D para la proyección Top
for col in range(B, D + 1):
    # Para cada columna, buscamos el primer píxel blanco (255) desde arriba
    for row in range(cropped_image.shape[0]):
        if cropped_image[row, col - B] == 255:
            top_projection.append(row)  # Guardamos la posición del primer píxel blanco
            break
    else:
        # Si no encontramos un píxel blanco, agregamos el valor de la altura total
        top_projection.append(cropped_image.shape[0])
    
# Crear el histograma de la proyección Left (de A a C)
left_projection = []

# Recorremos de A a C para la proyección Left
for row in range(A, C + 1):
    # Para cada fila, buscamos el primer píxel blanco (255) desde la izquierda
    for col in range(cropped_image.shape[1]):
        if cropped_image[row - A, col] == 255:
            left_projection.append(col)  # Guardamos la posición del primer píxel blanco
            break
    else:
        # Si no encontramos un píxel blanco, agregamos el valor del ancho total
        left_projection.append(cropped_image.shape[1])

# Crear el histograma de la proyección Bottom (de B a D)
bottom_projection = []

# Recorremos de B a D para la proyección Bottom
for col in range(B, D + 1):
    # Para cada columna, buscamos el último píxel blanco (255) desde arriba hacia abajo
    for row in range(cropped_image.shape[0] - 1, -1, -1):  # Recorremos desde la parte inferior hacia arriba
        if cropped_image[row, col - B] == 255:
            bottom_projection.append(row)  # Guardamos la posición del último píxel blanco
            break
    else:
        # Si no encontramos un píxel blanco, agregamos el valor de 0 (ningún píxel blanco encontrado)
        bottom_projection.append(0)

# Crear el histograma de la proyección Bottom (de B a D)
right_projection = []

# Recorremos de A a C para la proyección Bottom
for row in range(A, C + 1):
    # Para cada fila, buscamos el último píxel blanco (255) de derecha a izquierda
    for col in range(cropped_image.shape[1] - 1, -1, -1):  # Recorremos desde la parte derecha hacia izquierda
        if cropped_image[row - A, col] == 255:
            right_projection.append(col)  # Guardamos la posición del último píxel blanco
            break
    else:
        # Si no encontramos un píxel blanco, agregamos el valor de 0 (ningún píxel blanco encontrado)
        right_projection.append(0)

# Mostrar las imágenes
cv2.imshow('Imagen Original', image)
cv2.imshow('Imagen en Blanco y Negro', gray_image)
cv2.imshow('Imagen Binarizada', binary_image_rotated)
cv2.imshow('Imagen Recortada', cropped_image)

# Subplot 1: Top
plt.subplot(2, 2, 1)
plt.plot(top_projection, color='blue')
plt.title('Top View')

# Subplot 2: Left
plt.subplot(2, 2, 2)
plt.plot(left_projection, color='green')
plt.title('Left View')

# Subplot 3: Bottom
plt.subplot(2, 2, 3)
plt.plot(bottom_projection, color='red')
plt.title('Bottom View')

# Subplot 4: Right
plt.subplot(2, 2, 4)
plt.plot(right_projection, color='orange')
plt.title('Right View')

plt.tight_layout()
plt.show()

# Unir los perfiles en el orden: Arriba -> Izquierda -> Abajo -> Derecha
perfil_continuo = np.concatenate((top_projection, right_projection, bottom_projection, left_projection))
    
# Crear un gráfico de línea del perfil AVP continuo
plt.figure(figsize=(10, 4))
plt.plot(perfil_continuo, color='blue', lw=1)
plt.title("Perfil AVP Continuo de la Hoja")
plt.xlabel("Puntos del contorno")
plt.ylabel("Distancia al borde")
plt.grid(True)
plt.show()

# k: indice de coeficiente DCT a calcular
k = 3

# Coeficientes DCT
dct_coeffs = [calcular_dctk(perfil_continuo, k) for k in range(len(perfil_continuo))]  

# Reconstruir todo el perfil automáticamente
perfil_reconstruido = reconstruir_perfil_completo(dct_coeffs)

# Mostrar el perfil reconstruido
#print("Perfil reconstruido:", perfil_reconstruido)

plt.figure(figsize=(12, 6))
plt.plot(perfil_continuo, label="Perfil Original", color="blue")
plt.plot(perfil_reconstruido, label="Perfil Reconstruido", linestyle="--", color="red")
plt.title("Comparación del Perfil Original y Reconstruido")
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.legend()
plt.show()

# Calcula el error EDCT entre el perfil original y el reconstruido.
edct = calcular_edct(perfil_continuo, perfil_reconstruido)
#print(f"Error EDCT: {edct}")

# Realizar AVP Matching
planta_detectada = avp_matching(perfil_reconstruido, resultados_path, k=3)
print(f"La planta más probable es: {planta_detectada}")

cv2.waitKey(0)
cv2.destroyAllWindows()