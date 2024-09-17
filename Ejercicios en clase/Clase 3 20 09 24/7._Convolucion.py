import cv2
import numpy as np

# Cargar la imagen en escala de grises
imagen = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# Definir el kernel (ejemplo: detector de bordes Sobel)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Aplicar la convolución
imagen_convolucionada = cv2.filter2D(imagen, -1, kernel)

# Mostrar la imagen original y la imagen convolucionada
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Imagen Convolucionada', imagen_convolucionada)
cv2.waitKey(0)
cv2.destroyAllWindows()
