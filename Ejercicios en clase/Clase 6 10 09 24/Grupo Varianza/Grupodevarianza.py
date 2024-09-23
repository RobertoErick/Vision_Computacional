import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar la imagen y convertir a escala de grises
imagen = cv2.imread('imagen a color.png', cv2.IMREAD_GRAYSCALE)

# 2. Calcular el histograma (valores de intensidad y sus frecuencias)
histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
p_i = histograma / np.sum(histograma)  # Probabilidades p_i para cada nivel de gris

# 3. Función para calcular el umbral con η más cercano a 1 e imprimir todos los valores de η
def calcular_umbral_cercano(histograma, p_i):
    intensidades = np.arange(256)
    mu_T = np.sum(intensidades * p_i.flatten())  # Media total (μT)

    mejor_diferencia = float('inf')
    mejor_umbral = 0
    mejor_eta = 0

    # Lista para guardar los valores de eta y el umbral correspondiente
    valores_eta = []

    for t in range(1, 256):  # Iteramos sobre cada umbral posible
        pi_0 = np.sum(p_i[:t])
        pi_1 = np.sum(p_i[t:])

        if pi_0 == 0 or pi_1 == 0:
            continue

        mu_0 = np.sum(intensidades[:t] * p_i[:t].flatten()) / pi_0
        mu_1 = np.sum(intensidades[t:] * p_i[t:].flatten()) / pi_1

        # Varianza entre clases
        sigma_B2 = pi_0 * (mu_0 - mu_T)**2 + pi_1 * (mu_1 - mu_T)**2

        # Varianza total
        sigma_T2 = np.sum((intensidades - mu_T)**2 * p_i.flatten())

        # Calcular eta
        eta = sigma_B2 / sigma_T2

        # Guardamos el umbral y el eta para inspeccionar
        valores_eta.append((t, eta))

        # Comparar la diferencia con 1
        diferencia = abs(eta - 1)
        if diferencia < mejor_diferencia:
            mejor_diferencia = diferencia
            mejor_eta = eta
            mejor_umbral = t

    # Imprimir todos los valores de eta
    for umbral, eta in valores_eta:
        print(f"Umbral: {umbral}, eta: {eta:.4f}")

    # Graficar los valores de eta en función de los umbrales
    umbrales, etas = zip(*valores_eta)
    plt.plot(umbrales, etas)
    plt.xlabel('Umbral')
    plt.ylabel('eta')
    plt.title('Valores de eta para cada umbral')
    plt.show()

    return mejor_umbral, mejor_eta

# 4. Aplicar la función para encontrar el umbral con η más cercano a 1
mejor_umbral, mejor_eta = calcular_umbral_cercano(histograma, p_i)
print(f"\nMejor umbral: {mejor_umbral}, con η más cercano a 1: {mejor_eta:.4f}")

# 5. Umbralizar la imagen con el mejor umbral encontrado
_, imagen_umbralizada = cv2.threshold(imagen, mejor_umbral, 255, cv2.THRESH_BINARY)

# Guardar matriz original y procesada
np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
np.savetxt('imagen_resultante.csv', imagen_umbralizada, delimiter=',', fmt='%d')

# Guardar la imagen resultante
cv2.imwrite("imagen unbralizada.png", imagen_umbralizada)

# Mostrar la imagen original y la umbralizada
plt.subplot(1, 2, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')

plt.subplot(1, 2, 2)
plt.imshow(imagen_umbralizada, cmap='gray')
plt.title('Imagen Umbralizada')

plt.show()

# 6. Graficar el histograma de la imagen
plt.plot(histograma)
plt.xlabel('Intensidad de píxeles')
plt.ylabel('Frecuencia')
plt.title('Histograma de la imagen')
plt.show()
