import cv2
import numpy as np
import os
import math

def calcular_dctk(perfil, k):
    N = len(perfil)
    return sum(perfil[n] * np.cos((np.pi / N) * (n + 0.5) * k) for n in range(N))

def calcular_idcti(dct_coeffs, i):
    N = len(dct_coeffs)
    idcti = dct_coeffs[0] / 2
    idcti += sum(dct_coeffs[k] * np.cos((np.pi / N) * k * (i + 0.5)) for k in range(1, N))
    return idcti / N

def calcular_edct(perfil_original, perfil_reconstruido):
    N = len(perfil_original)
    error_cuadrado = np.sum((np.array(perfil_original) - np.array(perfil_reconstruido))**2)
    return np.sqrt(error_cuadrado / N)

def alinear_hoja(binary_image):
    non_zero_pixels = np.nonzero(binary_image)
    B = np.min(non_zero_pixels[1])
    D = np.max(non_zero_pixels[1])

    y_B = np.min(non_zero_pixels[0][non_zero_pixels[1] == B])
    y_D = np.min(non_zero_pixels[0][non_zero_pixels[1] == D])

    delta_y = y_D - y_B
    delta_x = D - B
    angle = math.degrees(math.atan2(delta_y, delta_x))

    # Rotar la imagen usando el ángulo calculado
    center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(binary_image, M, (binary_image.shape[1], binary_image.shape[0]))

def generar_perfil_avp_con_dct(image_path):
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

    # Alinear la hoja para garantizar invarianza de rotación
    binary_image = alinear_hoja(binary_image)

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

    # Calcular DCT
    dct_coeffs = [calcular_dctk(perfil_continuo, k) for k in range(len(perfil_continuo))]

    # Reconstruir el perfil
    perfil_reconstruido = [calcular_idcti(dct_coeffs, i) for i in range(len(perfil_continuo))]

    # Calcular el error EDCT
    edct = calcular_edct(perfil_continuo, perfil_reconstruido)

    return {
        "perfil_avp": perfil_continuo,
        "dct_coeffs": dct_coeffs,
        "perfil_reconstruido": perfil_reconstruido,
        "edct": edct
    }

# Directorio raíz donde están los datasets organizados por plantas
root_dir = r'C:\Users\tonit\Desktop\Dataset original\leafsnap-dataset\dataset\images\field'
resultados_por_planta = {}

# Recorrer todas las carpetas y subcarpetas
for dirpath, dirnames, filenames in os.walk(root_dir):
    planta_nombre = os.path.basename(dirpath)
    if planta_nombre not in resultados_por_planta:
        resultados_por_planta[planta_nombre] = []

    for idx, filename in enumerate(filenames):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dirpath, filename)
            try:
                print(f"[INFO] Procesando {idx + 1}/{len(filenames)}: {filename}")
                resultado = generar_perfil_avp_con_dct(image_path)
                resultados_por_planta[planta_nombre].append(resultado)
            except Exception as e:
                print(f"[ERROR] Procesando {filename}: {e}")

# Guardar los resultados por planta
np.save('resultados_avp_dct.npy', resultados_por_planta)
print("Resultados AVP y DCT guardados por planta en formato diccionario.")
