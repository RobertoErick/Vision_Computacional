import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Cargar la imagen y verificar si se cargó correctamente
image = cv2.imread('imagen fondo rosa 2.png')  # Reemplaza con la ruta de tu imagen
if image is None:
    raise FileNotFoundError("La imagen no se pudo cargar. Verifica la ruta.")

# Paso 2: Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Paso 3: Calcular el histograma de la imagen en escala de grises (hasta 255 para incluir fondo)
histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0, 255))

# Paso 4: Encontrar el valor dominante del fondo (el valor con más píxeles)
fondo_valor = np.argmax(histogram)
print("Valor dominante del fondo: ", fondo_valor)

# Paso 5: Definir un rango para el fondo alrededor del valor dominante
# (asumiendo que el fondo tiene una variación leve de intensidad)
rango_fondo = 60  # Ajustar según la variabilidad del fondo

lambda_min_fondo = max(fondo_valor - rango_fondo, 0)
lambda_max_fondo = min(fondo_valor + rango_fondo, 255)

print("lambda mínima (fondo): ", lambda_min_fondo)
print("lambda máxima (fondo): ", lambda_max_fondo)

# Paso 6: Binarización excluyendo el fondo
binary_image = np.where(
    (gray_image < lambda_min_fondo) | (gray_image > lambda_max_fondo), 1, 0).astype(np.uint8) * 255

# Paso 8: Usamos np.nonzero para encontrar los píxeles que no son fondo (es decir, píxeles diferentes de 0)
non_zero_pixels = np.nonzero(binary_image)

# Paso 3: Encontrar los límites de recorte (A, C, B, D)
A = np.min(non_zero_pixels[0])  # Límite superior (primer píxel no cero en filas)
C = np.max(non_zero_pixels[0])  # Límite inferior (último píxel no cero en filas)
B = np.min(non_zero_pixels[1])  # Límite izquierdo (primer píxel no cero en columnas)
D = np.max(non_zero_pixels[1])  # Límite derecho (último píxel no cero en columnas)

print("A: ",A,"\nB: ",B,"\nC: ",C,"\nD: ",D)

# Paso 9: Recortar la imagen original utilizando los límites encontrados
cropped_image = binary_image[A:C+1, B:D+1]

# PASO NUEVO: Crear el histograma de la proyección Top (de B a D)
top_projection = []

# Recorremos de B a D para la proyección Top
for col in range(B, D + 1):
    # Para cada columna, buscamos el primer píxel blanco (255) desde arriba
    for row in range(cropped_image.shape[0]):
        if cropped_image[row, col - B] == 255:
            top_projection.append(row)  # Guardamos la posición del primer píxel blanco
            break
    else:
        # Si no encontramos un píxel blanco, agregamos el valor de la altura total
        top_projection.append(cropped_image.shape[0])
    
# PASO NUEVO: Crear el histograma de la proyección Left (de A a C)
left_projection = []

# Recorremos de A a C para la proyección Left
for row in range(A, C + 1):
    # Para cada fila, buscamos el primer píxel blanco (255) desde la izquierda
    for col in range(cropped_image.shape[1]):
        if cropped_image[row - A, col] == 255:
            left_projection.append(col)  # Guardamos la posición del primer píxel blanco
            break
    else:
        # Si no encontramos un píxel blanco, agregamos el valor del ancho total
        left_projection.append(cropped_image.shape[1])

# PASO NUEVO: Crear el histograma de la proyección Bottom (de B a D)
bottom_projection = []

# Recorremos de B a D para la proyección Bottom
for col in range(B, D + 1):
    # Para cada columna, buscamos el último píxel blanco (255) desde arriba hacia abajo
    for row in range(cropped_image.shape[0] - 1, -1, -1):  # Recorremos desde la parte inferior hacia arriba
        if cropped_image[row, col - B] == 255:
            bottom_projection.append(row)  # Guardamos la posición del último píxel blanco
            break
    else:
        # Si no encontramos un píxel blanco, agregamos el valor de 0 (ningún píxel blanco encontrado)
        bottom_projection.append(0)

# PASO NUEVO: Crear el histograma de la proyección Bottom (de B a D)
right_projection = []

# Recorremos de A a C para la proyección Bottom
for row in range(A, C + 1):
    # Para cada fila, buscamos el último píxel blanco (255) de derecha a izquierda
    for col in range(cropped_image.shape[1] - 1, -1, -1):  # Recorremos desde la parte derecha hacia izquierda
        if cropped_image[row - A, col] == 255:
            right_projection.append(col)  # Guardamos la posición del último píxel blanco
            break
    else:
        # Si no encontramos un píxel blanco, agregamos el valor de 0 (ningún píxel blanco encontrado)
        right_projection.append(0)

# Paso 8: Mostrar el histograma que excluye el fondo
plt.figure()
plt.title("Histograma de Píxeles (excluyendo fondo blanco)")
plt.xlabel("Valor de Intensidad")
plt.ylabel("Número de píxeles")
plt.xlim([0, 255])  # Solo queremos ver los valores de 0 a 255
plt.plot(bin_edges[0:-1], histogram)  # Excluyendo el último valor para evitar desbordamientos
plt.show()

# Mostrar las imágenes
cv2.imshow('Imagen Original', image)
cv2.imshow('Imagen en Blanco y Negro', gray_image)
cv2.imshow('Imagen Binarizada', binary_image)
cv2.imshow('Imagen Recortada', cropped_image)

"""
# Graficar la proyección Top
plt.figure()
plt.title("Proyección desde Arriba (Top View)")
plt.xlabel("Columna")
plt.ylabel("Primera posición del píxel blanco")
plt.plot(range(B, D + 1), top_projection)
plt.show()

# Graficar la proyección Left
plt.figure()
plt.title("Proyección desde la Izquierda (Left View)")
plt.xlabel("Fila")
plt.ylabel("Primera posición del píxel blanco")
plt.plot(range(A, C + 1), left_projection)
plt.show()

# Graficar la proyección Bottom
plt.figure()
plt.title("Proyección desde Abajo (Bottom View)")
plt.xlabel("Columna")
plt.ylabel("Última posición del píxel blanco")
plt.plot(range(B, D + 1), bottom_projection)
plt.show()

# Graficar la proyección Right
plt.figure()
plt.title("Proyección desde la Derecha (Right View)")
plt.xlabel("Fila")
plt.ylabel("Última posición del píxel blanco")
plt.plot(range(A, C + 1), right_projection)
plt.show()
"""

# Subplot 1: Top
plt.subplot(2, 2, 1)
plt.plot(top_projection, color='blue')
plt.title('Top View')

# Subplot 2: Left
plt.subplot(2, 2, 2)
plt.plot(left_projection, color='green')
plt.title('Left View')

# Subplot 3: Bottom
plt.subplot(2, 2, 3)
plt.plot(bottom_projection, color='red')
plt.title('Bottom View')

# Subplot 4: Right
plt.subplot(2, 2, 4)
plt.plot(right_projection, color='orange')
plt.title('Right View')

plt.tight_layout()
plt.show()

# Unir los perfiles en el orden: Arriba -> Izquierda -> Abajo -> Derecha
perfil_continuo = np.concatenate((top_projection, right_projection, bottom_projection, left_projection))
    
# Crear un gráfico de línea del perfil AVP continuo
plt.figure(figsize=(10, 4))
plt.plot(perfil_continuo, color='blue', lw=1)
plt.title("Perfil AVP Continuo de la Hoja")
plt.xlabel("Puntos del contorno")
plt.ylabel("Distancia al borde")
plt.grid(True)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
