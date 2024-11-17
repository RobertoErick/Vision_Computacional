import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Paso 1: Cargar la imagen y verificar si se cargó correctamente
image = cv2.imread('imagen fondo rosa 2.png')  # Reemplaza con la ruta de tu imagen
if image is None:
    raise FileNotFoundError("La imagen no se pudo cargar. Verifica la ruta.")

# Paso 2: Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Paso 3: Calcular el histograma de la imagen en escala de grises
histogram, bin_edges = np.histogram(gray_image, bins=256, range=(0, 255))

# Paso 4: Encontrar el valor dominante del fondo
fondo_valor = np.argmax(histogram)
print("Valor dominante del fondo: ", fondo_valor)

# Paso 5: Definir un rango para el fondo alrededor del valor dominante
rango_fondo = 60  # Ajustar según la variabilidad del fondo
lambda_min_fondo = max(fondo_valor - rango_fondo, 0)
lambda_max_fondo = min(fondo_valor + rango_fondo, 255)

print("lambda mínima (fondo): ", lambda_min_fondo)
print("lambda máxima (fondo): ", lambda_max_fondo)

# Paso 6: Binarización excluyendo el fondo
binary_image = np.where(
    (gray_image < lambda_min_fondo) | (gray_image > lambda_max_fondo), 1, 0).astype(np.uint8) * 255

# Paso 7: Encontrar los límites de recorte (A, C, B, D)
non_zero_pixels = np.nonzero(binary_image)
A = np.min(non_zero_pixels[0])  # Límite superior
C = np.max(non_zero_pixels[0])  # Límite inferior
B = np.min(non_zero_pixels[1])  # Límite izquierdo
D = np.max(non_zero_pixels[1])  # Límite derecho
cropped_image = binary_image[A:C+1, B:D+1]

# Proyección de perfiles en las direcciones Top, Left, Bottom, y Right
top_projection, left_projection, bottom_projection, right_projection = [], [], [], []

# Proyección Top
for col in range(B, D + 1):
    for row in range(cropped_image.shape[0]):
        if cropped_image[row, col - B] == 255:
            top_projection.append(row)
            break
    else:
        top_projection.append(cropped_image.shape[0])

# Proyección Left
for row in range(A, C + 1):
    for col in range(cropped_image.shape[1]):
        if cropped_image[row - A, col] == 255:
            left_projection.append(col)
            break
    else:
        left_projection.append(cropped_image.shape[1])

# Proyección Bottom
for col in range(B, D + 1):
    for row in range(cropped_image.shape[0] - 1, -1, -1):
        if cropped_image[row, col - B] == 255:
            bottom_projection.append(row)
            break
    else:
        bottom_projection.append(0)

# Proyección Right
for row in range(A, C + 1):
    for col in range(cropped_image.shape[1] - 1, -1, -1):
        if cropped_image[row - A, col] == 255:
            right_projection.append(col)
            break
    else:
        right_projection.append(0)

# Crear el perfil AVP continuo
perfil_continuo = np.concatenate((top_projection, right_projection, bottom_projection, left_projection))

# PASO NUEVO: Aplicar la Transformada Discreta del Coseno (DCT) al perfil AVP
dct_profile = dct(perfil_continuo, type=2)

# PASO NUEVO: Reducción de dimensionalidad usando PCA con solo 1 componente
pca = PCA(n_components=1)
dct_profile_reduced = pca.fit_transform(dct_profile.reshape(1, -1))

# PASO NUEVO: Crear un conjunto de entrenamiento con una sola característica
# Datos ficticios de entrenamiento para ilustración (cambiar por datos reales en un proyecto real)
X_train = np.random.rand(10, 1)  # 10 muestras, 1 componente cada una
y_train = np.random.randint(0, 2, 10)  # Etiquetas ficticias para clasificación (reemplaza con etiquetas reales)

# Entrenar el clasificador con una sola característica
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predicción con el perfil actual reducido
prediction = knn.predict(dct_profile_reduced)
print("La especie predicha es:", prediction)

# Mostrar las imágenes y gráficos
cv2.imshow('Imagen Original', image)
cv2.imshow('Imagen en Blanco y Negro', gray_image)
cv2.imshow('Imagen Binarizada', binary_image)
cv2.imshow('Imagen Recortada', cropped_image)

# Mostrar el perfil AVP continuo
plt.figure(figsize=(10, 4))
plt.plot(perfil_continuo, color='blue', lw=1)
plt.title("Perfil AVP Continuo de la Hoja")
plt.xlabel("Puntos del contorno")
plt.ylabel("Distancia al borde")
plt.grid(True)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
