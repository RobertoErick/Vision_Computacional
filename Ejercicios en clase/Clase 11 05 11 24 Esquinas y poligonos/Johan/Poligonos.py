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

# Aplicar el detector de esquinas Harris
gray_image = np.float32(gray_image)
dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)

# Resultados de Harris, dilatar las esquinas para visualizarlas
dst = cv2.dilate(dst, None)

# Umbral para marcar las esquinas
threshold = 0.01 * dst.max()
esquinas = np.argwhere(dst > threshold)

# Dibujar círculos en las esquinas detectadas
imagen_con_esquinas = image.copy()
for esquina in esquinas:
    y, x = esquina  # Coordenadas (y, x)
    cv2.circle(imagen_con_esquinas, (x, y), 5, (0, 0, 255), 2)  # Esquinas en rojo

# Mostrar la imagen con las esquinas detectadas
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(imagen_con_esquinas, cv2.COLOR_BGR2RGB))
plt.title('Esquinas Detectadas')
plt.axis('off')
plt.show()

# Ordenar las esquinas detectadas
def ordenar_esquinas(esquinas):
    # Calcular el centroide de las esquinas
    centro = np.mean(esquinas, axis=0)
    
    # Ordenar las esquinas en función del ángulo polar relativo al centroide
    angles = np.arctan2(esquinas[:, 0] - centro[0], esquinas[:, 1] - centro[1])
    ordenadas = esquinas[np.argsort(angles)]  # Ordenar en base a los ángulos
    return ordenadas

# Ordenar las esquinas de manera que formen un polígono
esquinas_ordenadas = ordenar_esquinas(np.array(esquinas))

# Dibujar líneas para formar el polígono
imagen_con_poligono = imagen_con_esquinas.copy()

# Conectar las esquinas ordenadas en un polígono
for i in range(len(esquinas_ordenadas) - 1):
    pt1 = tuple(esquinas_ordenadas[i][::-1])  # Convertir de (y, x) a (x, y) para OpenCV
    pt2 = tuple(esquinas_ordenadas[i + 1][::-1])
    cv2.line(imagen_con_poligono, pt1, pt2, (0, 255, 0), 2)  # Conectar con líneas verdes

# Cerrar el polígono conectando la última esquina con la primera
cv2.line(imagen_con_poligono, tuple(esquinas_ordenadas[-1][::-1]), tuple(esquinas_ordenadas[0][::-1]), (0, 255, 0), 2)

# Mostrar la imagen final con el polígono
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(imagen_con_poligono, cv2.COLOR_BGR2RGB))
plt.title('Polígono Cerrado con Esquinas Detectadas')
plt.axis('off')
plt.show()

# Opcional: Guardar la imagen resultante
cv2.imwrite("poligono_cerrado.jpg", imagen_con_poligono)
