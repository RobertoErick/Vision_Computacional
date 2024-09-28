import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

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

# Cargar la imagen y convertirla a escala de grises
ruta_imagen = 'imagen.png'  
imagen = io.imread(ruta_imagen, as_gray=True)
imagen = (imagen * 255).astype(np.uint8)  # Normalizar la imagen a valores de 0 a 255

# Calcular el umbral, desviación estándar y los picos
mejor_umbral, desviacion_estandar, picos = calcular_valle_global(imagen)

# Aplicar el umbral a la imagen
imagen_umbralizada = aplicar_umbral(imagen, mejor_umbral)

# Imprimir los resultados
print(f'Mejor umbral por valle global: {mejor_umbral}')
print(f'Desviación estándar de la imagen: {desviacion_estandar}')
print(f'Picos encontrados en el histograma: {picos}')

# Guardar matriz original y procesada
np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
np.savetxt('imagen_resultante.csv', imagen_umbralizada, delimiter=',', fmt='%d')

# Guardar la imagen resultante
cv2.imwrite("imagen unbralizada.png", imagen_umbralizada)

# Mostrar la imagen original y la imagen umbralizada
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 3, 2)
plt.imshow(imagen_umbralizada, cmap='gray')
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
