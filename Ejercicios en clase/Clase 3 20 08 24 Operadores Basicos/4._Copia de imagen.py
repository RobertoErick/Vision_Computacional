import cv2
import numpy as np

# Cargar imagen
imagen = cv2.imread('imagen a color.png', cv2.IMREAD_GRAYSCALE)

# Comprobar que la imagen se cargo correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    #   Guardar imagen
    #   Q(x,y) = P(x,y)
    imagen_Copia = imagen.copy()

    # Matriz de la imagen original
    np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen original se ha guardado en 'imagen_original.csv'.")

    # Matriz de la imagen cambio de brillo
    np.savetxt('imagen_copia.csv', imagen_Copia, delimiter=',', fmt='%d')
    print("La matriz de la imagen copia se ha guardado en 'imagen_copia.csv'.")

    # Mostrar imagen
    cv2.imshow('Original', imagen)
    cv2.imshow('Copia', imagen_Copia)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar imagen
    cv2.imread('Copia.png', imagen_Copia)