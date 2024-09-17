import cv2

# Cargar imagen
img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# Comprobar que la imagen se cargo correctamente
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    # Guardar imagen
    # Q(x,y) = P(x,y)
    imgCopia = img.copy()

    # Mostrar imagen
    cv2.imshow('Original', img)
    cv2.imshow('Copia', imgCopia)
    cv2.waitKey(0)
    cv2.destroyAllWindows()