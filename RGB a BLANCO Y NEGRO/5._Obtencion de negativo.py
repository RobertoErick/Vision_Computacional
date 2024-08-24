import cv2

# Cargar la imagen
imagen = cv2.imread('Lenna.png')

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Convertir la imagen a negativo
    # Q(x,y) = 255 - P(x,y)
    negativo = cv2.bitwise_not(imagen)

    # Mostrar la imagen original y la imagen negativa
    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen Negativa', negativo)

    # Esperar a que se presione una tecla y cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()
