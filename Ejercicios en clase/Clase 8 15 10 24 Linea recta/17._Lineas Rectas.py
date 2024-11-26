import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

# Función para aplicar RANSAC en múltiples iteraciones
def detectar_varias_lineas_ransac(puntos_X, puntos_Y, min_puntos=5, max_iteraciones=50):
    lineas_ransac = []
    iteracion = 0

    while len(puntos_X) > min_puntos and iteracion < max_iteraciones:
        # Aplicar RANSAC para detectar una línea
        ransac = RANSACRegressor()
        ransac.fit(puntos_X, puntos_Y)
        
        # Obtener los inliers, es decir, los puntos que se ajustan bien a la línea
        inlier_mask = ransac.inlier_mask_
        
        # Extraer los puntos que corresponden a la línea detectada
        puntos_inliers_X = puntos_X[inlier_mask]
        puntos_inliers_Y = puntos_Y[inlier_mask]
        
        # Guardar los puntos de la línea detectada
        lineas_ransac.append((puntos_inliers_X, puntos_inliers_Y))
        
        # Eliminar los inliers de la lista de puntos (quitar la línea ya detectada)
        puntos_X = puntos_X[~inlier_mask]
        puntos_Y = puntos_Y[~inlier_mask]
        
        iteracion += 1

    return lineas_ransac

# Función para detectar vecindarios en rango de +50 y -50 (ya existente)
def detectar_vecindarios(image, rango=50):
    rows, cols = image.shape
    visitado = np.zeros((rows, cols), dtype=bool)
    vecindarios = []
    bordes_vecindarios = []

    def obtener_vecindario(r, c, pivote_valor, rango):
        vecindario = [(r, c)]
        borde = set()
        cola = [(r, c)]
        visitado[r, c] = True
        pivote_valor = int(pivote_valor)

        while cola:
            x, y = cola.pop(0)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visitado[nx, ny]:
                        if pivote_valor - rango <= int(image[nx, ny]) <= pivote_valor + rango:
                            visitado[nx, ny] = True
                            vecindario.append((nx, ny))
                            cola.append((nx, ny))
                        else:
                            borde.add((nx, ny))
        return vecindario, borde

    for r in range(rows):
        for c in range(cols):
            if not visitado[r, c]:
                pivote_valor = image[r, c]
                vecindario, borde = obtener_vecindario(r, c, pivote_valor, rango)
                vecindarios.append(vecindario)
                bordes_vecindarios.append(borde)

    return vecindarios, bordes_vecindarios

# Cargar imagen 
imagen_a_color = cv2.imread('imagen.png')
if imagen_a_color is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # Convertirla a escala de grises
    imagen = cv2.cvtColor(imagen_a_color, cv2.COLOR_BGR2GRAY)

    # Detectar vecindarios y bordes
    vecindarios, bordes_vecindarios = detectar_vecindarios(imagen)

    # Mostrar cuántos vecindarios se encontraron
    print(f"Número de vecindarios encontrados: {len(vecindarios)}")

    # Para visualizar los vecindarios en la imagen (opcional)
    output_image = np.zeros_like(imagen)
    for i, vecindario in enumerate(vecindarios):
        color_value = (255 - (i * 25)) % 256  # Asegurar que los valores estén entre 0 y 255
        for (r, c) in vecindario:
            output_image[r, c] = color_value  # Colorear cada vecindario de manera diferente
    # Crear imagen binaria para los bordes
    bordes_imagen = np.zeros_like(imagen)

    # Marcar los píxeles de borde en la imagen binaria
    for borde in bordes_vecindarios:
        for (r, c) in borde:
            bordes_imagen[r, c] = 255  # Borde en blanco

    # Aplicar la Transformada de Hough directamente sobre la imagen binarizada de bordes
    # threshold: Decide que el minimo de votos que debe tener para detectarse como linea recta
    # minLineLength: La minima cantidad de pixeles para poder considerarse linea recta
    # masLineGap: maxima separacion permitida entre dos puntos de una linea
    lineas = cv2.HoughLinesP(bordes_imagen, rho=1, theta=np.pi / 180, threshold=50, minLineLength=2, maxLineGap=10)

    solo_lineas = np.zeros_like(bordes_imagen)
    bordes_restantes = np.copy(bordes_imagen)

    if lineas is not None:
        for linea in lineas:
            for x1, y1, x2, y2 in linea:
                cv2.line(imagen_a_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibuja cada línea con color verde
                cv2.line(solo_lineas, (x1, y1), (x2, y2), (255), 2)  # Dibuja cada línea con color verde
                cv2.line(bordes_restantes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Quita las lineas rectas

    # Mostrar la imagen con las líneas detectadas
    cv2.imshow('Vecindarios', output_image)
    cv2.imshow("Bordes de la imagen", bordes_imagen)
    cv2.imshow('Lineas rectas sobrepuestas', imagen_a_color)
    cv2.imshow('Solo lineas rectas', solo_lineas)
    cv2.imshow("Bordes restantes de la imagen", bordes_restantes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar las imagenes
    cv2.imwrite('Imagen Vecindarios.png', output_image)
    cv2.imwrite("Imagen Bordes de la imagen.png", bordes_imagen)
    cv2.imwrite('Imagen Lineas rectas sobrepuestas.png', imagen_a_color)
    cv2.imwrite('Imagen Solo lineas rectas.png', solo_lineas)
    cv2.imwrite("Imagen Bordes restantes de la imagen.png", bordes_restantes)

    # Guardar la matriz de la imagen en escala de grises en un archivo CSV
    np.savetxt('Matriz imagen_gris.csv', imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'imagen.csv'.")

    # Guardar la matriz de la imagen vecindarios en un archivo CSV
    np.savetxt('Matriz Vecindarios.csv', output_image, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'imagen_vecindarios.csv'.")

    # Guardar la matriz de la imagen Bordes de la imagen en un archivo CSV
    np.savetxt('Matriz Bordes de la imagen.csv', bordes_imagen, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'imagen_vecindarios.csv'.")

    # Guardar la matriz de la imagen Solo lineas rectas en un archivo CSV
    np.savetxt('Matriz Solo lineas rectas.csv', solo_lineas, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'imagen_vecindarios.csv'.")

    # Guardar la matriz de la imagen Bordes restantes de la imagen en un archivo CSV
    np.savetxt('Matriz Bordes restantes de la imagen.csv', bordes_restantes, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'imagen_vecindarios.csv'.")