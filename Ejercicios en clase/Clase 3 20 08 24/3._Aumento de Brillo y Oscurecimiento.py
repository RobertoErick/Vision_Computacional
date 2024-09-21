import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('imagen a color.png', cv2.IMREAD_GRAYSCALE)

def cambiar_brillo(imagen, factor):
    imagen = imagen.astype(np.float32)  #   Convertir a float para evitar overflow/underflow
    imagen = imagen * factor            #   Multiplicar por el factor
    imagen = np.clip(imagen, 0, 255)    #   Asegurar que los valores estén en el rango 0-255
    return imagen.astype(np.uint8)   #   Convertir de nuevo a uint8

# Comprobar que la imagen se subio correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    #   Numero para aumentar o disminuir el brillo
    #   Mas de 1.0 para aumentar
    #   Menos de 1.0 para disminuir
    factor = 1.5
    cambio_de_brillo = cambiar_brillo(imagen, factor)

    # Matriz de la imagen original
    np.savetxt('imagen_original.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen original se ha guardado en 'imagen_original.csv'.")

    # Matriz de la imagen cambio de brillo
    np.savetxt('imagen_cambio_brillo.csv', cambio_de_brillo, delimiter=',', fmt='%d')
    print("La matriz de la imagen con cambio de brillo se ha guardado en 'imagen_cambio_brillo.csv'.")

    # Mostrar las imagenes originales y la modificada
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen con Brillo Aumentado', cambio_de_brillo)

    # Esperar hasta que se presione una tecla para quitar las imagenes
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Imagen guardada en los archivos
    cv2.imwrite('Imagen con Brillo Aumentado.png', cambio_de_brillo)