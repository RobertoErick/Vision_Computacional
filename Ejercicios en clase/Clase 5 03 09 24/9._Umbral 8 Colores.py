import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

thresholds = [32, 64, 96, 128, 160, 192, 224]


imagen_umbralizada = np.zeros_like(imagen)

imagen_umbralizada[imagen <= thresholds[0]] = 0
imagen_umbralizada[(imagen > thresholds[0]) & (imagen <= thresholds[1])] = 36  
imagen_umbralizada[(imagen > thresholds[1]) & (imagen <= thresholds[2])] = 72  
imagen_umbralizada[(imagen > thresholds[2]) & (imagen <= thresholds[3])] = 109 
imagen_umbralizada[(imagen > thresholds[3]) & (imagen <= thresholds[4])] = 145  
imagen_umbralizada[(imagen > thresholds[4]) & (imagen <= thresholds[5])] = 182  
imagen_umbralizada[(imagen > thresholds[5]) & (imagen <= thresholds[6])] = 218  
imagen_umbralizada[imagen > thresholds[6]] = 255  

plt.hist(imagen_umbralizada.ravel(), bins=16, range=[0, 256], color='gray', alpha=0.7)

plt.title('Histograma de Imagen con 8 Colores')
plt.xlabel('Intensidad de Gris')
plt.ylabel('Número de Píxeles')

plt.show()

cv2.imshow('Imagen con Umbral de 8 Colores', imagen_umbralizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('imagen_umbralizada_8_colores.png', imagen_umbralizada)
