import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargamos la imagen 
imagen_color = cv2.imread("imagen a color.png")
if imagen_color is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que el archivo está en el directorio correcto.")
else:
    # Convertimos la imagen a escala de grises
    imagen = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # Los puntos donde se va a dividir la Umbralizacion
    thresholds = [32, 64, 96, 128, 160, 192, 224]

    # Devuelve la imagen umralizada solo con ceros
    imagen_umbralizada = np.zeros_like(imagen)

    # Divide la umbralizacion en 2 colores
    imagen_umbralizada[imagen <= thresholds[0]] = 0
    imagen_umbralizada[(imagen > thresholds[0]) & (imagen <= thresholds[1])] = 36  
    imagen_umbralizada[(imagen > thresholds[1]) & (imagen <= thresholds[2])] = 72  
    imagen_umbralizada[(imagen > thresholds[2]) & (imagen <= thresholds[3])] = 109 
    imagen_umbralizada[(imagen > thresholds[3]) & (imagen <= thresholds[4])] = 145  
    imagen_umbralizada[(imagen > thresholds[4]) & (imagen <= thresholds[5])] = 182  
    imagen_umbralizada[(imagen > thresholds[5]) & (imagen <= thresholds[6])] = 218  
    imagen_umbralizada[imagen > thresholds[6]] = 255  

    # Histograma
    plt.hist(imagen_umbralizada.ravel(), bins=16, range=[0, 256], color='gray', alpha=0.7)

    plt.title('Histograma de Imagen con 8 Colores')
    plt.xlabel('Intensidad de Gris')
    plt.ylabel('Número de Píxeles')

    plt.show()

    # Guardar la matriz de la imagen en escala de grises en un archivo CSV
    np.savetxt('Imagen_escala_de_grises.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'Imagen_escala_de_grises.csv'.")

    # Guardar la matriz de la imagen umbralizada en un archivo CSV
    np.savetxt('imagen_umbralizada_8_colores.csv', imagen_umbralizada, delimiter=',', fmt='%d')
    print("La matriz de la imagen umbralizada se ha guardado en 'imagen_umbralizada_8_colores'.")

    # Mostramos la imagen que se ah generado
    cv2.imshow('Imagen con Umbral de 8 Colores', imagen_umbralizada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardamos la imagen en los archivos
    cv2.imwrite('imagen_umbralizada_8_colores.png', imagen_umbralizada)