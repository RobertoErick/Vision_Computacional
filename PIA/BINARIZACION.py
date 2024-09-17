import cv2
import numpy as np

# Función para cargar la imagen y convertirla a escala de grises
def cargar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen, imagen_gris

# Función para binarizar la imagen utilizando un umbral automático (Otsu)
def binarizar_imagen(imagen_gris):
    _, imagen_binaria = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imagen_binaria

# Función para recortar la imagen eliminando el fondo innecesario
def recortar_imagen(imagen_binaria):
    # Encontrar los contornos de la imagen binaria
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        raise ValueError("No se encontraron contornos en la imagen.")
    
    # Obtener el contorno más grande (el de la hoja)
    contorno_hoja = max(contornos, key=cv2.contourArea)
    
    # Obtener el rectángulo delimitador alrededor del contorno de la hoja
    x, y, ancho, alto = cv2.boundingRect(contorno_hoja)
    
    # Recortar la imagen original utilizando el rectángulo delimitador
    imagen_recortada = imagen_binaria[y:y+alto, x:x+ancho]
    
    return imagen_recortada

# Función principal de preprocesamiento
def preprocesar_imagen(ruta_imagen):
    imagen_original, imagen_gris = cargar_imagen(ruta_imagen)
    imagen_binaria = binarizar_imagen(imagen_gris)
    imagen_recortada = recortar_imagen(imagen_binaria)
    
    # Mostrar resultados (opcional)
    cv2.imshow('Imagen Original', imagen_original)
    cv2.imshow('Imagen Binaria', imagen_binaria)
    cv2.imshow('Imagen Recortada', imagen_recortada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imagen_recortada

# Ruta de la imagen a preprocesar
ruta_imagen = 'Huizache.jpg'  # Cambia esta ruta por la ubicación de tu imagen

# Ejecutar el preprocesamiento
imagen_final = preprocesar_imagen(ruta_imagen)

# Guardar la imagen preprocesada si lo deseas
cv2.imwrite('imagen_preprocesada.png', imagen_final)
