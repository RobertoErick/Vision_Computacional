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
    np.savetxt('Original_Shifting_diag_izq_arr_a_der_aba.csv', gray_image, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'Original_Shifting_diag_izq_arr_a_der_aba.csv'.")

    gray_copy = gray_image.copy()
    
    # Recorrer la imagen en diagonales desde la esquina superior izquierda a la inferior derecha
    rows, cols = gray_image.shape
    for diag in range(rows + cols - 1):  # Esto recorre todas las diagonales posibles
        # La fila inicial es el mínimo entre la diagonal y el número de filas - 1
        # La columna inicial es la diferencia entre la diagonal y la fila
        for i in range(max(0, diag - cols + 1), min(rows, diag + 1)):
            j = diag - i
            if i > 0 and j > 0:  # Evita el borde
                gray_copy[i][j] = gray_copy[i-1][j-1] / (gray_image[i-1][j-1] / gray_image[i][j])

    np.savetxt('Shifting_diag_izq_arr_a_der_aba.csv', gray_copy, delimiter=',', fmt='%d')

    # Mostrar la imagen en blanco y negro
    cv2.imshow("Imagen a Blanco y negro", gray_image)
    cv2.imshow("shifting diagonal izquierda arriba a derecha abajo", gray_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("shifting diagonal izquierda arriba a derecha abajo.png", gray_copy)
