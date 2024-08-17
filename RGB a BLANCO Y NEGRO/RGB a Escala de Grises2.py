import cv2

imagen = cv2.imread("Lenna.png")
cv2.imshow("Imagen", imagen)
print(imagen)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
cv2.imshow("Escala de grises", imagen_gris)
print(imagen_gris)
cv2.waitKey(0)