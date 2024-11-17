import cv2
import numpy as np
import os

def generar_perfil_avp(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histograma para el fondo
    histogram, _ = np.histogram(gray_image, bins=256, range=(0, 255))
    fondo_valor = np.argmax(histogram)
    
    # Rango de fondo y binarización
    rango_fondo = 60
    lambda_min_fondo = max(fondo_valor - rango_fondo, 0)
    lambda_max_fondo = min(fondo_valor + rango_fondo, 255)
    binary_image = np.where((gray_image < lambda_min_fondo) | (gray_image > lambda_max_fondo), 1, 0).astype(np.uint8) * 255

    # Encontrar límites y recortar la imagen
    non_zero_pixels = np.nonzero(binary_image)
    A, C = np.min(non_zero_pixels[0]), np.max(non_zero_pixels[0])
    B, D = np.min(non_zero_pixels[1]), np.max(non_zero_pixels[1])
    cropped_image = binary_image[A:C+1, B:D+1]

    # Proyecciones AVP
    top_projection = [next((row for row in range(cropped_image.shape[0]) if cropped_image[row, col] == 255), cropped_image.shape[0]) for col in range(cropped_image.shape[1])]
    bottom_projection = [next((row for row in range(cropped_image.shape[0] - 1, -1, -1) if cropped_image[row, col] == 255), 0) for col in range(cropped_image.shape[1])]
    left_projection = [next((col for col in range(cropped_image.shape[1]) if cropped_image[row, col] == 255), cropped_image.shape[1]) for row in range(cropped_image.shape[0])]
    right_projection = [next((col for col in range(cropped_image.shape[1] - 1, -1, -1) if cropped_image[row, col] == 255), 0) for row in range(cropped_image.shape[0])]

    # Crear perfil AVP continuo
    perfil_continuo = np.concatenate((top_projection, right_projection, bottom_projection, left_projection))
    return perfil_continuo

# Directorio raíz donde están los datasets organizados por plantas
root_dir = r'C:\Users\tonit\OneDrive\Documentos\ITS Semestre 11\VISION COMPUTACIONAL\PIA\DATASETS\RGB'  # Ruta donde se encuentra el dataset
perfiles_avp_por_planta = {}

# Recorrer todas las carpetas y subcarpetas
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Obtener el nombre de la planta como el nombre de la carpeta actual
    planta_nombre = os.path.basename(dirpath)
    if planta_nombre not in perfiles_avp_por_planta:
        perfiles_avp_por_planta[planta_nombre] = []

    for filename in filenames:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dirpath, filename)
            try:
                perfil_avp = generar_perfil_avp(image_path)
                perfiles_avp_por_planta[planta_nombre].append(perfil_avp)
            except Exception as e:
                print(f"Error procesando {image_path}: {e}")

# Guardar los perfiles AVP por planta
np.save('perfiles_avp_por_planta.npy', perfiles_avp_por_planta)
print("Perfiles AVP guardados y clasificados por planta.")
