import cv2
import numpy as np

# Lee la imagen
imagen = cv2.imread("imagen a color.png")
if imagen is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que el archivo está en el directorio correcto.")
else:
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Guardar la matriz de la imagen en escala de grises en un archivo CSV
    np.savetxt('Original_Shifting_diag_der_arr_a_izq_aba.csv', gray_image, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'Original_Shifting_diag_der_arr_a_izq_aba.csv'.")

    gray_copy = gray_image.copy()
    
    rows, cols = gray_image.shape
    # Recorrer la imagen en diagonales desde la esquina superior derecha a la inferior izquierda
    for diag in range(-cols + 1, rows):  # Esto recorre todas las diagonales posibles
        for i in range(max(0, diag), min(rows, cols + diag)):  # Limitar el recorrido a los bordes de la imagen
            j = cols - 1 - (i - diag)  # Columna que depende de la fila y la diagonal
            if i > 0 and j < cols - 1:  # Evitar el borde
                gray_copy[i][j] = gray_copy[i-1][j+1] / (gray_image[i-1][j+1] / gray_image[i][j])

    np.savetxt('Shifting_diag_der_arr_a_izq_aba.csv', gray_copy, delimiter=',', fmt='%d')

    # Mostrar la imagen en blanco y negro
    cv2.imshow("Imagen a Blanco y negro", gray_image)
    cv2.imshow("shifting diagonal derecha arriba a izquierda abajo", gray_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("shifting diagonal derecha arriba a izquierda abajo.png", gray_copy)
