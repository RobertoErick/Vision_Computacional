import cv2
import numpy as np

# Lee la imagen
imagen = cv2.imread("Lenna.png")
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Mostrar la imagen original
    cv2.imshow("Imagen Original", imagen)
    
    # Definir un kernel de suavizado (blur) 3x3
    kernel_suavizado = np.ones((3, 3), np.float32) / 9

    # Aplicar la convolución con el kernel de suavizado
    imagen_suavizada = cv2.filter2D(imagen, -1, kernel_suavizado)

    # Mostrar la imagen suavizada
    cv2.imshow("Imagen Suavizada", imagen_suavizada)

    # Aplicar el filtro Sobel para la detección de bordes
    sobelx = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)  # Sobel en dirección x
    sobely = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)  # Sobel en dirección y
    sobel_combined = cv2.magnitude(sobelx, sobely)  # Magnitud combinada

    # Convertir a escala de grises y luego a uint8
    sobel_combined = np.uint8(np.absolute(sobel_combined))

    # Mostrar la imagen con detección de bordes (Sobel)
    cv2.imshow("Imagen con Detección de Bordes (Sobel)", sobel_combined)

    # Esperar a que se presione una tecla y cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()