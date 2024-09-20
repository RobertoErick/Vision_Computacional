import cv2
import numpy as np

# Lee la imagen
imagen = cv2.imread("imagen de prueba.png")
if imagen is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que el archivo está en el directorio correcto.")
else:
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Guardar la matriz de la imagen en escala de grises en un archivo CSV
    np.savetxt('Original_Shifting_izq_a_der.csv', gray_image, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'Original_Shifting_izq_a_der.csv'.")

    gray_copy = gray_image.copy()

    #Recorrer el arreglo de la imagen en escala de grises
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if j != 0:
                gray_copy[i][j] = gray_copy[i][j-1]/ (gray_image[i][j-1]/gray_image[i][j])   

    np.savetxt('Shifting_izq_a_der.csv', gray_copy, delimiter=',', fmt='%d')

    # Mostrar la imagen en blanco y negro
    cv2.imshow("Imagen a Blanco y negro", gray_image)
    cv2.imshow("shifting izquierda a derecha", gray_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("shifting izquierda a derecha.png", gray_copy)