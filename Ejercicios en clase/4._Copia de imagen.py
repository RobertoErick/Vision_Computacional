import cv2

# Cargar imagen
img = cv2.imread('LennaByN.png')

# Comprobar que la imagen se cargo correctamente
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    # Mostrar imagen
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar imagen
    # Q(x,y) = P(x,y)
    cv2.imwrite('LennaByNcopia.png', img)