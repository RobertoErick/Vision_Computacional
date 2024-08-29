import cv2
import numpy as np

# Lee la imagen
imagen = cv2.imread("Incendio.png")
if imagen is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que el archivo 'Incendio.png' está en el directorio correcto.")
else:
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Guardar la matriz de la imagen en escala de grises en un archivo CSV
    np.savetxt('imagen_gris.csv', gray_image, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'imagen_gris.csv'.")

    gray_copy = gray_image.copy()

    # Coordenada Y en plano cartesiano
    print(gray_image.shape[0])
    # Coordenada X en plano cartesiano
    print(gray_image.shape[1])

    # Recorrer el arreglo de la imagen en escala de grises
    for i in range(gray_image.shape[0]):        # Y
        for j in range(gray_image.shape[1]):    # X
            if j != 0:                          # El inicio de X no se va a contar
                gray_copy[i][j] = round(gray_copy[i][j-1]/ (gray_image[i][j]/gray_image[i][j-1]))
            else:
                gray_copy[i][j] = gray_image[i][j]

    np.savetxt('imagen_gris_copia.csv', gray_copy, delimiter=',', fmt='%d')

    # Mostrar la imagen en blanco y negro
    cv2.imshow("Imagen a Blanco y negro", gray_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    cv2.imshow("shifting", gray_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()