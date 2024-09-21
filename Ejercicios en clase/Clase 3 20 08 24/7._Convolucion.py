import cv2
import numpy as np

# Cargar la imagen en escala de grises
imagen = cv2.imread('imagen a color.png', cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se cargo correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Definir el kernel (ejemplo: detector de bordes Sobel)
    kernel = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])

    # Aplicar la convolución
    imagen_convolucionada = cv2.filter2D(imagen, -1, kernel)

    # Matriz de la imagen original
    np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen original se ha guardado en 'imagen_original.csv'.")

    # Matriz de la imagen cambio de brillo
    np.savetxt('imagen_convolucionada.csv',imagen_convolucionada, delimiter=',', fmt='%d')
    print("La matriz de la imagen convolucionada se ha guardado en 'imagen_convolucionada.csv'.")

    # Mostrar la imagen original y la imagen convolucionada
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen Convolucionada', imagen_convolucionada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen en los archivos
    cv2.imwrite('Imagen Convolucionada.png', imagen_convolucionada)
