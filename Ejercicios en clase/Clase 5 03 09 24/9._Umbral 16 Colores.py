import cv2
import numpy as np
import matplotlib.pyplot as plt


imagen = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

thresholds = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]

imagen_umbralizada = np.zeros_like(imagen)

imagen_umbralizada[imagen <= thresholds[0]] = 0
imagen_umbralizada[(imagen > thresholds[0]) & (imagen <= thresholds[1])] = 17    
imagen_umbralizada[(imagen > thresholds[1]) & (imagen <= thresholds[2])] = 34    
imagen_umbralizada[(imagen > thresholds[2]) & (imagen <= thresholds[3])] = 51    
imagen_umbralizada[(imagen > thresholds[3]) & (imagen <= thresholds[4])] = 68    
imagen_umbralizada[(imagen > thresholds[4]) & (imagen <= thresholds[5])] = 85    
imagen_umbralizada[(imagen > thresholds[5]) & (imagen <= thresholds[6])] = 102   
imagen_umbralizada[(imagen > thresholds[6]) & (imagen <= thresholds[7])] = 119   
imagen_umbralizada[(imagen > thresholds[7]) & (imagen <= thresholds[8])] = 136   
imagen_umbralizada[(imagen > thresholds[8]) & (imagen <= thresholds[9])] = 153   
imagen_umbralizada[(imagen > thresholds[9]) & (imagen <= thresholds[10])] = 170  
imagen_umbralizada[(imagen > thresholds[10]) & (imagen <= thresholds[11])] = 187 
imagen_umbralizada[(imagen > thresholds[11]) & (imagen <= thresholds[12])] = 204 
imagen_umbralizada[(imagen > thresholds[12]) & (imagen <= thresholds[13])] = 221 
imagen_umbralizada[(imagen > thresholds[13]) & (imagen <= thresholds[14])] = 238 
imagen_umbralizada[imagen > thresholds[14]] = 255  

plt.hist(imagen_umbralizada.ravel(), bins=16, range=[0, 256], color='gray', alpha=0.7)

plt.title('Histograma de Imagen con 16 Colores')
plt.xlabel('Intensidad de Gris')
plt.ylabel('Número de Píxeles')

plt.show()

cv2.imshow('Imagen con Umbral de 16 Colores', imagen_umbralizada)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('imagen_umbralizada_16_colores.png', imagen_umbralizada)
