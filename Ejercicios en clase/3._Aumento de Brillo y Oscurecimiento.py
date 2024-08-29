import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('Lenna.png')

# Comprobar que la imagen se subio correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    # Mayor el numero mas claro (0 - 255)
    # Menor el numero mas oscuro (-255 - 0)
    # P(x,y) = I 
    brillo = 170
    bright_image = cv2.convertScaleAbs(image, alpha=1, beta=brillo)

    # Mostrar las imagenes originales y la modificada
    cv2.imshow('Imagen Original', image)
    cv2.imshow('Imagen con Brillo Aumentado', bright_image)

    # Esperar hasta que se presione una tecla para quitar las imagenes
    cv2.waitKey(0)
    cv2.destroyAllWindows()