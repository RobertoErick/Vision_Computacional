import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image_path = "cubo.png"  # Cambia esto por la ruta de tu imagen
image = cv2.imread(image_path)
if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar el operador Sobel para encontrar bordes en la imagen
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Derivada en el eje X
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Derivada en el eje Y

# Calcular el gradiente total en cada punto
magnitude = cv2.magnitude(sobel_x, sobel_y)

# Definir un umbral para identificar bordes fuertes (potenciales esquinas)
umbral = np.max(magnitude) * 0.1  # Ajustar el umbral para que detecte más esquinas

# Crear una máscara que marque los bordes fuertes
bordes_fuertes = magnitude > umbral

# Convertir la máscara de bordes a una imagen binaria (blanco para bordes, negro para el resto)
imagen_binaria = np.zeros_like(gray_image)
imagen_binaria[bordes_fuertes] = 255

# Crear una copia de la imagen original para dibujar las esquinas
imagen_con_bordes = image.copy()

# Dibujar círculos donde se detectan bordes fuertes (esquinas)
for i in range(1, imagen_binaria.shape[0] - 1):
    for j in range(1, imagen_binaria.shape[1] - 1):
        if imagen_binaria[i, j] == 255:
            cv2.circle(imagen_con_bordes, (j, i), radius=5, color=(0, 255, 0), thickness=-1)

# Mostrar la imagen con las esquinas detectadas y resaltadas
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(imagen_con_bordes, cv2.COLOR_BGR2RGB))
plt.title('Esquinas Detectadas (Con Sobel)')
plt.axis('off')
plt.show()
