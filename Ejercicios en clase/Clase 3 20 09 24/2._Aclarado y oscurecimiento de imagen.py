import cv2

#   Lee la imagen en escala de grises
img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

#   Revisar la imagen que se haya cargado correctamente
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    #   Mayor el valor mas claro
    #   (9 - 255) Mas claro
    #   (-255 - 0) Mas oscuro
    valor = 100
    img_aclarada = cv2.convertScaleAbs(img, alpha=1, beta=valor)

    #   Utilizamos plt para poder ver las coordenadas donde se aclara
    cv2.imshow("Imagen original", img)
    cv2.imshow("Imagen aclarada", img_aclarada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()