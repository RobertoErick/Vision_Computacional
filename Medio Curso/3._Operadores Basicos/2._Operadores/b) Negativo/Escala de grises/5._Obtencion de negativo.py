import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('Imagen a Blanco y negro.png', cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    #   Convertir la imagen a negativo
    #   Q(x,y) = 255 - P(x,y)
    imagen_negativa = cv2.bitwise_not(imagen)

    # Matriz de la imagen original
    np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen original se ha guardado en 'imagen_original.csv'.")

    # Matriz de la imagen cambio de brillo
    np.savetxt('imagen_negativa.csv', imagen_negativa, delimiter=',', fmt='%d')
    print("La matriz de la imagen negativa se ha guardado en 'imagen_negativa.csv'.")

    # Mostrar la imagen original y la imagen negativa
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen Negativa', imagen_negativa)

    # Esperar a que se presione una tecla y cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen en los archivos
    cv2.imwrite('Imagen Negativa.png', imagen_negativa)
