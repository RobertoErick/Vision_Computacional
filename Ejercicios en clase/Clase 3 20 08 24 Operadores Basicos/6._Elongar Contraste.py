import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('imagen a color.png', cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargo correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Valores para aumentar el contraste y Elongar la imagen
    # Q0 = P0*gamma+beta
    alpha = 2.0 
    beta = -100 
    imagen_mejorada = cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

    # Matriz de la imagen original
    np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen original se ha guardado en 'imagen_original.csv'.")

    # Matriz de la imagen cambio de brillo
    np.savetxt('imagen_mejorada.csv',imagen_mejorada, delimiter=',', fmt='%d')
    print("La matriz de la imagen mejorada se ha guardado en 'imagen_mejorada.csv'.")

    # Mostrar las imagenes en las ventanas
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen con Mejora de Contraste', imagen_mejorada)

    # No quitar las imagenes hasta que se presione una tecla
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guarda la imagen en los archivos
    cv2.imwrite('Imagen con Mejora de Contraste.png', imagen_mejorada)