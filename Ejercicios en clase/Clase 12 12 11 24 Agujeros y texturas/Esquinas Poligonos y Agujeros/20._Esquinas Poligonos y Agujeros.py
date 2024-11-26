import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Cargar la imagen
image_path = "imagen.png"  # Cambia esto por la ruta de tu imagen
image = cv2.imread(image_path)

if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar el detector de esquinas Harris
gray_image = np.float32(gray_image)
dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

# Dilatar las esquinas para visualizarlas
dst = cv2.dilate(dst, None)

# Umbral relativo para considerar esquinas
threshold = 0.02 * dst.max()
esquinas = np.argwhere(dst > threshold)

# Dibujar círculos en las esquinas detectadas
imagen_con_esquinas = image.copy()
solo_esquinas = np.zeros_like(gray_image)

for esquina in esquinas:
    y, x = esquina  # Coordenadas (y, x)
    cv2.circle(imagen_con_esquinas, (x, y), 5, (0, 0, 255), 2)  # Esquinas en rojo
    cv2.circle(solo_esquinas, (x, y), 5, (255), 2)

cv2.imshow('Solo Esquinas Detectadas', solo_esquinas)
# Mostrar la imagen con las esquinas detectadas
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(imagen_con_esquinas, cv2.COLOR_BGR2RGB))
plt.title('Esquinas Detectadas')
plt.axis('off')
plt.show()

# Guardar la imagen resultante
cv2.imwrite('Imagen Esquinas Detectadas.png', imagen_con_esquinas)

# Convertir las esquinas a formato adecuado para agrupar
puntos = np.array([p[::-1] for p in esquinas], dtype=np.float32)  # Cambiar a (x, y)

clustering = DBSCAN(eps=10, min_samples=4).fit(puntos)  # Ajusta 'eps' según la distancia esperada entre puntos
labels = clustering.labels_

# Dibujar polígonos detectados
imagen_con_poligonos = image.copy()
solo_poligonos = np.zeros_like(gray_image)
poligonos = []

for label in set(labels):
    if label == -1:  # Ignorar ruido
        continue

    # Extraer puntos de un grupo
    grupo_puntos = puntos[labels == label]

    # Encontrar el contorno aproximado del polígono
    hull = cv2.convexHull(grupo_puntos.astype(np.int32))
    epsilon = 0.02 * cv2.arcLength(hull, True)
    aproximado = cv2.approxPolyDP(hull, epsilon, True)

    # Dibujar el polígono
    cv2.polylines(imagen_con_poligonos, [aproximado], True, (0, 255, 0), 2)
    cv2.polylines(solo_poligonos, [aproximado], True, (255), 2)
    poligonos.append(aproximado)

cv2.imshow('Solo poligonos detectadas', solo_poligonos)
# Mostrar la imagen con polígonos detectados
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(imagen_con_poligonos, cv2.COLOR_BGR2RGB))
plt.title(f'Polígonos Detectados: {len(poligonos)}')
plt.axis('off')
plt.show()

# Guardar la imagen resultante
cv2.imwrite('Imagen poligonos detectados.png', imagen_con_poligonos)

# Detectar agujeros
imagen_con_resultados = image.copy()
solo_agujeros = np.zeros_like(gray_image)
agujeros = 0

for poligono in poligonos:
    # Crear una máscara para el polígono
    mask = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.drawContours(mask, [poligono], -1, 255, thickness=cv2.FILLED)

    # Extraer intensidades dentro del polígono
    intensidades = gray_image[mask == 255]
    promedio = np.mean(intensidades)

    # Crear máscara para agujeros basados en la diferencia de intensidad
    lower_bound = promedio - 50
    upper_bound = promedio + 50
    mask_agujeros = ((gray_image < lower_bound) | (gray_image > upper_bound)) & (mask == 255)

    # Detectar contornos de los agujeros
    contornos, _ = cv2.findContours(mask_agujeros.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        # Dibujar el agujero en rojo
        cv2.drawContours(imagen_con_resultados, [contorno], -1, (0, 0, 255), 2)
        cv2.drawContours(solo_agujeros, [contorno], -1, (255), 2)
        agujeros += 1

    # Dibujar el polígono en verde
    cv2.drawContours(imagen_con_resultados, [poligono], -1, (0, 255, 0), 2)

cv2.imshow('Imagen poligonos y agujeros', solo_agujeros)
# Mostrar la imagen con polígonos y agujeros detectados
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(imagen_con_resultados, cv2.COLOR_BGR2RGB))
plt.title(f'Polígonos: {len(poligonos)}, Agujeros: {agujeros}')
plt.axis('off')
plt.show()

# Guardar la imagen resultante
cv2.imwrite('Imagen poligonos y agujeros.png', imagen_con_resultados)

# Guardar la matriz de la imagen en escala de grises en un archivo CSV
np.savetxt('Matriz Solo esquinas.csv', solo_esquinas, delimiter=',', fmt='%d')
print("La matriz de la imagen binarizada se ha guardado en 'Matriz Solo esquinas.csv'.")

# Guardar la matriz de la imagen en escala de grises en un archivo CSV
np.savetxt('Matriz Solo poligonos.csv', solo_poligonos, delimiter=',', fmt='%d')
print("La matriz de la imagen binarizada se ha guardado en 'Matriz Solo poligonos.csv'.")

# Guardar la matriz de la imagen en escala de grises en un archivo CSV
np.savetxt('Matriz Solo agujeros.csv', solo_agujeros, delimiter=',', fmt='%d')
print("La matriz de la imagen binarizada se ha guardado en 'Matriz Solo agujeros.csv'.")