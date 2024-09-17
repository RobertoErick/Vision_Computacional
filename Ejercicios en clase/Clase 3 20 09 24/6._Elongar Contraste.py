import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread("Lenna.png")

# Verificar si la imagen se cargo correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Convertir la imagen en escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Valores para aumentar el contraste y Elongar la imagen
    # Q0 = P0*gamma+beta
    alpha = 2.0 
    beta = -100 
    imagen_mejorada = cv2.convertScaleAbs(imagen_gris, alpha=alpha, beta=beta)

    # Mostrar las imagenes en las ventanas
    cv2.imshow('Imagen Original', imagen_gris)
    cv2.imshow('Imagen con Mejora de Contraste', imagen_mejorada)

    # No quitar las imagenes hasta que se presione una tecla
    cv2.waitKey(0)
    cv2.destroyAllWindows()