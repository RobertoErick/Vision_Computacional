import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def calcular_histograma(imagen):
    histograma, _ = np.histogram(imagen.ravel(), bins=256, range=(0, 256))
    return histograma

def calcular_entropia(probabilidades):
    probabilidades = probabilidades[probabilidades > 0]  
    return -np.sum(probabilidades * np.log10(probabilidades))

def umbral_por_entropia(imagen):
    histograma = calcular_histograma(imagen)
    total_pixeles = imagen.size

    max_entropia = -np.inf
    mejor_umbral = 0

    for umbral in range(256):
        fondo = histograma[:umbral]
        primer_plano = histograma[umbral:]

        prob_fondo = fondo / np.sum(fondo) if np.sum(fondo) > 0 else np.zeros_like(fondo)
        prob_primer_plano = primer_plano / np.sum(primer_plano) if np.sum(primer_plano) > 0 else np.zeros_like(primer_plano)

        entropia_fondo = calcular_entropia(prob_fondo)
        entropia_primer_plano = calcular_entropia(prob_primer_plano)

        entropia_total = entropia_fondo + entropia_primer_plano

        if entropia_total > max_entropia:
            max_entropia = entropia_total
            mejor_umbral = umbral

    _, imagen_umbralizada = cv2.threshold(imagen, mejor_umbral, 255, cv2.THRESH_BINARY)

    return mejor_umbral, imagen_umbralizada

ruta_imagen = 'imagen a color.png'  
imagen = io.imread(ruta_imagen, as_gray=True)
imagen = (imagen * 255).astype(np.uint8)  #
mejor_umbral, imagen_umbralizada = umbral_por_entropia(imagen)

# Guardar las matrizes de las imagenes originales y resultantes
np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
np.savetxt('imagen_resultante.csv', imagen_umbralizada, delimiter=',', fmt='%d')

# Guardar la imagen resultante
cv2.imwrite("imagen unbralizada.png", imagen_umbralizada)

print(f'Mejor umbral por entropía: {mejor_umbral}')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(imagen_umbralizada, cmap='gray')
plt.title('Imagen umbralizada por entropía')

plt.show()