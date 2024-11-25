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
    np.savetxt('Original_Shifting_diag_izq_aba_a_der_arr.csv', gray_image, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'Original_Shifting_diag_izq_aba_a_der_arr.csv'.")

    gray_copy = gray_image.copy()
    
    rows, cols = gray_image.shape
    # Recorrer la imagen en diagonales desde la esquina inferior izquierda a la superior derecha
    for diag in range(rows + cols - 1):  # Recorrer todas las diagonales posibles
        for i in range(min(diag, rows - 1), max(-1, diag - cols), -1):  # Limitar el recorrido dentro de los bordes
            j = diag - i  # Columna correspondiente a la fila
            if i < rows - 1 and j > 0:  # Evitar los bordes
                gray_copy[i][j] = gray_copy[i+1][j-1] / (gray_image[i+1][j-1] / gray_image[i][j])

    np.savetxt('Shifting_diag_izq_aba_a_der_arr.csv', gray_copy, delimiter=',', fmt='%d')

    # Mostrar la imagen en blanco y negro
    cv2.imshow("Imagen a Blanco y negro", gray_image)
    cv2.imshow("shifting diagonal izquierda abajo a derecha arriba", gray_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("shifting diagonal izquierda abajo a derecha arriba.png", gray_copy)
