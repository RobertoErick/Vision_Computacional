import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('Lenna.png')

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    # Definir el valor de aumento de brillo
    valor_brillo = 50  # Ajustar el valor del brillo

    # Crear una matriz con el mismo tamaño que la imagen y con el valor de brillo
    # Q0 = P0 + beta (Esto es mas aplicado al ejercicio 3, ya que aqui se crea una matriz mas clara para aplicarla a la imagen original)
    matriz_brillo = np.full(imagen.shape, valor_brillo, dtype=np.uint8)

    # Aumentar el brillo usando cv2.add
    imagen_brillante = cv2.add(imagen, matriz_brillo)

    # Mostrar la imagen original y la imagen con brillo aumentado
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen con Brillo Aumentado', imagen_brillante)

    # Esperar a que se presione una tecla y cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen con brillo aumentado si es necesario
    cv2.imwrite('imagen_brillante.jpg', imagen_brillante)
