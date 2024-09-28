import cv2
import numpy as np
import matplotlib.pyplot as plt


# Cargamos la imagen 
imagen_color = cv2.imread("Imagen a Blanco y negro.png")
if imagen_color is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que el archivo está en el directorio correcto.")
else:
    # Convertimos la imagen a escala de grises
    imagen = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # Los puntos donde se va a dividir la Umbralizacion
    thresholds = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240]

    # Devuelve la imagen umralizada solo con ceros
    imagen_umbralizada = np.zeros_like(imagen)

    # Divide la umbralizacion en 2 colores
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

    # Histograma
    plt.hist(imagen_umbralizada.ravel(), bins=16, range=[0, 256], color='gray', alpha=0.7)

    plt.title('Histograma de Imagen con 16 Colores')
    plt.xlabel('Intensidad de Gris')
    plt.ylabel('Número de Píxeles')

    plt.show()

    # Guardar la matriz de la imagen en escala de grises en un archivo CSV
    np.savetxt('Imagen_escala_de_grises.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'Imagen_escala_de_grises.csv'.")

    # Guardar la matriz de la imagen umbralizada en un archivo CSV
    np.savetxt('imagen_umbralizada_16_colores.csv', imagen_umbralizada, delimiter=',', fmt='%d')
    print("La matriz de la imagen umbralizada se ha guardado en 'imagen_umbralizada_16_colores'.")

    cv2.imshow('Imagen con Umbral de 16 Colores', imagen_umbralizada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardamos la imagen en los archivos
    cv2.imwrite('imagen_umbralizada_16_colores.png', imagen_umbralizada)