import cv2
import numpy as np

# Lee la imagen en escala de grises
imagen = cv2.imread('imagen a color.png', cv2.IMREAD_GRAYSCALE)

# Revisar la imagen que se haya cargado correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    #   Mayor el valor mas claro
    #   (9 - 255) Mas claro
    #   (-255 - 0) Mas oscuro
    valor = 100
    imagen_aclarada = cv2.convertScaleAbs(imagen, alpha=1, beta=valor)

    # Matriz de la imagen original
    np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen original se ha guardado en 'imagen_original.csv'.")

    # Matriz de la imagen aclarada
    np.savetxt('imagen_aclarada.csv', imagen_aclarada, delimiter=',', fmt='%d')
    print("La matriz de la imagen aclarada se ha guardado en 'imagen_aclarada.csv'.")

    # Mostrar las imagenes
    cv2.imshow("Imagen original", imagen)
    cv2.imshow("Imagen aclarada", imagen_aclarada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen en los archivos
    cv2.imwrite("Imagen aclarada.png", imagen_aclarada)