import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

thresholds = [64, 128, 192]


imagen_umbralizada = np.zeros_like(imagen)

imagen_umbralizada[imagen <= thresholds[0]] = 0
imagen_umbralizada[(imagen > thresholds[0]) & (imagen <= thresholds[1])] = 85
imagen_umbralizada[(imagen > thresholds[1]) & (imagen <= thresholds[2])] = 170
imagen_umbralizada[imagen > thresholds[2]] = 255

plt.hist(imagen_umbralizada.ravel(), bins=16, range=[0, 256], color='gray', alpha=0.7)

plt.title('Histograma de Imagen con 4 Colores')
plt.xlabel('Intensidad de Gris')
plt.ylabel('Número de Píxeles')

plt.show()

cv2.imshow('Imagen con Umbral de 4 Colores', imagen_umbralizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('imagen_umbralizada_4_colores.png', imagen_umbralizada)
