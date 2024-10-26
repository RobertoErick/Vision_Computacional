import cv2
import numpy as np

# Función para detectar vecindarios
def detectar_vecindarios(image, rango=50):
    # Obtener las dimensiones de la imagen
    rows, cols = image.shape
    visitado = np.zeros((rows, cols), dtype=bool)  # Para marcar píxeles ya revisados
    vecindarios = []  # Lista para almacenar los vecindarios encontrados
    bordes_vecindarios = []  # Lista para almacenar los bordes de vecindarios

    def obtener_vecindario(r, c, pivote_valor, rango):
        vecindario = [(r, c)]
        borde = set()
        cola = [(r, c)]
        visitado[r, c] = True
        
        # Convertir el pivote a int para evitar desbordamiento
        pivote_valor = int(pivote_valor)

        while cola:
            x, y = cola.pop(0)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visitado[nx, ny]:
                        # Convertir el valor actual del píxel a int antes de la comparación
                        if pivote_valor - rango <= int(image[nx, ny]) <= pivote_valor + rango:
                            visitado[nx, ny] = True
                            vecindario.append((nx, ny))
                            cola.append((nx, ny))
                        else:
                            borde.add((nx, ny))
        return vecindario, borde

    # Recorrer la imagen
    for r in range(rows):
        for c in range(cols):
            if not visitado[r, c]:
                pivote_valor = image[r, c]
                vecindario, borde = obtener_vecindario(r, c, pivote_valor, rango)
                vecindarios.append(vecindario)
                bordes_vecindarios.append(borde)

    return vecindarios, bordes_vecindarios

# Cargar imagen en escala de grises
imagen_a_color = cv2.imread('imagen recortada.png')
if imagen_a_color is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que el archivo 'Imagen a Color.png' está en el directorio correcto.")
else:
    imagen = cv2.cvtColor(imagen_a_color, cv2.COLOR_BGR2GRAY)

    # Detectar vecindarios y bordes
    vecindarios, bordes_vecindarios = detectar_vecindarios(imagen)

    # Mostrar cuántos vecindarios se encontraron
    print(f"Número de vecindarios encontrados: {len(vecindarios)}")

    # Mostrar los píxeles borde de cada vecindario
    #for i, borde in enumerate(bordes_vecindarios):
        #print(f"Vecindario {i + 1} tiene {len(borde)} píxeles borde.")

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

    cv2.imshow('Vecindarios', output_image)
    cv2.imwrite('Vecindarios.png', output_image) 
    cv2.imshow("Bordes de la imagen", bordes_imagen)
    cv2.imwrite("Bordes de la imagen.png", bordes_imagen)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()

