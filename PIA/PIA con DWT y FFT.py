import cv2
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import math
import pywt

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
# ---------------------------------------Seccion DWT y FFT-------------------------------------------------- #

def calcular_energia_transformada(coefs, n):
    """
    Calcula el porcentaje de energía acumulada en los primeros 'n' coeficientes.
    """
    coefs_recortados = coefs[:n]
    energia_total = np.sum(np.array(coefs)**2)
    energia_acumulada = np.sum(np.array(coefs_recortados)**2)
    return (energia_acumulada / energia_total) * 100

def calcular_energia_por_transformada(perfil, max_coefs=100):
    """
    Calcula el porcentaje de energía para DCT, DWT y FFT.
    """
    resultados = {"DCT": [], "DWT": [], "FFT": []}
    
    # DCT
    dct_coefs = [calcular_dctk(perfil, k) for k in range(len(perfil))]
    
    # DWT (usando Daubechies Wavelet 'db1')
    dwt_coefs, _ = pywt.dwt(perfil, 'db1')
    dwt_coefs = sorted(np.abs(dwt_coefs), reverse=True)  # Ordenar por magnitud
    
    # FFT
    fft_coefs = np.abs(fft(perfil))
    fft_coefs = sorted(fft_coefs, reverse=True)  # Ordenar por magnitud
    
    # Calcular energía para cada número de coeficientes
    for n in range(1, max_coefs + 1):
        resultados["DCT"].append(calcular_energia_transformada(dct_coefs, n))
        resultados["DWT"].append(calcular_energia_transformada(dwt_coefs, n))
        resultados["FFT"].append(calcular_energia_transformada(fft_coefs, n))
    
    return resultados

def graficar_energia_transformada(resultados, max_coefs=100):
    """
    Grafica el porcentaje de energía retenida para DCT, DWT y FFT.
    """
    x = range(1, max_coefs + 1)  # Coeficiente largo
    plt.figure(figsize=(10, 6))
    for metodo, energias in resultados.items():
        plt.plot(x, energias, label=f"{metodo} Energy")
    
    plt.title("Porcentaje de Energía vs Coeficiente Largo")
    plt.xlabel("Número de Coeficientes")
    plt.ylabel("Porcentaje de Energía (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# ----------------------------------------Seccion Principal------------------------------------------------- #

# Cargar dataset anteriormente procesado y verificar contenido
resultados_path = 'resultados_avp_dct.npy'  # Archivo de resultados
if not resultados_path:
    raise ValueError("El diccionario está vacío. Verifica la ruta")

# Cargar la imagen y verificar si se cargó correctamente
image = cv2.imread('hoja_rotada2.jpg')  # Imagen a predecir
if image is None:
    raise FileNotFoundError("La imagen no se pudo cargar. Verifica la ruta.")

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calcular el histograma de la imagen en escala de grises
histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0, 255))

# Encontrar el valor dominante del fondo (el valor con más píxeles)
fondo_valor = np.argmax(histogram)
print("Valor dominante del fondo: ", fondo_valor)

# Definir un rango para calcular el fondo a travez de un pivote (como vecindarios) selecciona el fondo en el histograma y denota la hoja
rango_fondo = 60  # Si el fondo es constante (segun el Dataset lo es) no habra problema con este valor

lambda_min_fondo = max(fondo_valor - rango_fondo, 0)
lambda_max_fondo = min(fondo_valor + rango_fondo, 255)

print("lambda mínima (fondo): ", lambda_min_fondo)
print("lambda máxima (fondo): ", lambda_max_fondo)

# Binarización excluyendo el fondo
binary_image = np.where(
    (gray_image < lambda_min_fondo) | (gray_image > lambda_max_fondo), 1, 0).astype(np.uint8) * 255

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

# Mostrar el histograma que excluye el fondo
plt.figure()
plt.title("Histograma de Píxeles")
plt.xlabel("Valor de Intensidad")
plt.ylabel("Número de píxeles")
plt.xlim([0, 255])  # Solo queremos ver los valores de 0 a 255
plt.plot(bin_edges[0:-1], histogram)  # Excluyendo el último valor para evitar desbordamientos
plt.show()

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

# Usar el perfil continuo previamente calculado
resultados_energia = calcular_energia_por_transformada(perfil_continuo, max_coefs=150)
graficar_energia_transformada(resultados_energia, max_coefs=150)