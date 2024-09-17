import cv2
import numpy as np

#   Cargar la imagen
img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

def cambiar_brillo(img, factor):
    img = img.astype(np.float32)  #   Convertir a float para evitar overflow/underflow
    img = img * factor            #   Multiplicar por el factor
    img = np.clip(img, 0, 255)    #   Asegurar que los valores estén en el rango 0-255
    return img.astype(np.uint8)          #   Convertir de nuevo a uint8

#   Comprobar que la imagen se subio correctamente
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    #   Numero para aumentar o disminuir el brillo
    #   Mas de 1.0 para aumentar
    #   Menos de 1.0 para disminuir
    factor = 0.2

    #   Mostrar las imagenes originales y la modificada
    cv2.imshow('Imagen Original', img)
    cv2.imshow('Imagen con Brillo Aumentado', cambiar_brillo(img, factor))

    #   Esperar hasta que se presione una tecla para quitar las imagenes
    cv2.waitKey(0)
    cv2.destroyAllWindows()