import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

def detectar_linea_conectada(bordes_restantes):
    # Crear una copia de la imagen para visualizar solo la línea encontrada
    linea_detectada = np.zeros_like(bordes_restantes)

    # Dimensiones de la imagen
    rows, cols = bordes_restantes.shape

    # Variable para marcar si se ha encontrado la primera línea
    linea_encontrada = False

    # Función para seguir una línea conectada usando una búsqueda en profundidad (DFS)
    def seguir_linea(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < cols and 0 <= cy < rows and bordes_restantes[cy, cx] == 255:
                # Marcar el píxel en la imagen de la línea detectada
                linea_detectada[cy, cx] = 255
                # Marcar el píxel como visitado en bordes_restantes para no repetir
                bordes_restantes[cy, cx] = 0
                # Agregar píxeles vecinos (4 direcciones) a la pila
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    stack.append((nx, ny))

    # Recorrer la imagen de izquierda a derecha y de arriba a abajo
    for y in range(rows):
        for x in range(cols):
            if bordes_restantes[y, x] == 255:
                seguir_linea(x, y)
                linea_encontrada = True
                break  # Detener después de encontrar la primera línea
        if linea_encontrada:
            break

    # Mostrar la imagen con la única línea detectada
    cv2.imshow("Primera línea conectada", linea_detectada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return linea_detectada

def verificar_centro_candidato(linea_detectada, centro_x, centro_y, imagen_prueba):
    # Dimensiones de la imagen
    rows, cols = linea_detectada.shape

    # Definir las 8 direcciones de los píxeles
    direcciones = [
        (-1, 0),   # arriba
        (1, 0),    # abajo
        (0, -1),   # izquierda
        (0, 1),    # derecha
        (-1, -1),  # diagonal arriba-izquierda
        (-1, 1),   # diagonal arriba-derecha
        (1, -1),   # diagonal abajo-izquierda
        (1, 1)     # diagonal abajo-derecha
    ]

    # Contador de líneas que tocan un borde
    contador_lineas_que_toquen_borde = 0

    # Crear una copia de la imagen para dibujar las líneas de prueba
    imagen_lineas = cv2.cvtColor(linea_detectada, cv2.COLOR_GRAY2BGR)

    # Verificar cada dirección y dibujar las líneas de prueba
    for dx, dy in direcciones:
        end_x, end_y = centro_x, centro_y

        # Extender la línea en la dirección actual hasta encontrar un borde o salir de la imagen
        while 0 <= end_x + dx < cols and 0 <= end_y + dy < rows:
            end_x += dx
            end_y += dy
            if linea_detectada[end_y, end_x] == 255:  # Si encontramos un borde
                contador_lineas_que_toquen_borde += 1
                # Dibujar la línea desde el centro hasta el borde encontrado
                cv2.line(imagen_lineas, (centro_x, centro_y), (end_x, end_y), (0, 0, 255), 1)
                break
            # Dibujar la línea de prueba mientras busca el borde
            cv2.circle(imagen_lineas, (end_x, end_y), 1, (255, 0, 0), -1)  # puntos de prueba en azul

    # Mostrar las líneas de prueba y el proceso de detección
    cv2.imshow("Líneas de prueba para verificación de centro", imagen_lineas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Comprobar si al menos 3 líneas tocan un borde
    return contador_lineas_que_toquen_borde >= 3

def encontrar_centro_con_rango_de_busqueda(linea_detectada, intervalo=10, rango=20):
    acumulador = np.zeros_like(linea_detectada, dtype=np.int32)
    rows, cols = linea_detectada.shape

    # Definir las 8 direcciones de los píxeles
    direcciones = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    contador_pixeles = 0

    for y in range(rows):
        for x in range(cols):
            if linea_detectada[y, x] == 255:
                contador_pixeles += 1
                if contador_pixeles % intervalo == 0:
                    longitudes_lineas = []
                    puntos_finales = []

                    for dx, dy in direcciones:
                        end_x, end_y = x, y
                        longitud = 0
                        while 0 <= end_x + dx < cols and 0 <= end_y + dy < rows:
                            end_x += dx
                            end_y += dy
                            longitud += 1
                            if linea_detectada[end_y, end_x] == 255:
                                break

                        longitudes_lineas.append(longitud)
                        puntos_finales.append((end_x, end_y))

                    max_longitud_idx = np.argmax(longitudes_lineas)
                    end_x, end_y = puntos_finales[max_longitud_idx]

                    mid_x = (x + end_x) // 2
                    mid_y = (y + end_y) // 2

                    acumulador[mid_y, mid_x] += 1

    contador_vecindario = np.zeros_like(acumulador)
    for y in range(rango, rows - rango):
        for x in range(rango, cols - rango):
            ventana = acumulador[y - rango:y + rango + 1, x - rango:x + rango + 1]
            contador_vecindario[y, x] = np.sum(ventana)

    centro_y, centro_x = np.unravel_index(np.argmax(contador_vecindario), contador_vecindario.shape)
    max_votos = contador_vecindario[centro_y, centro_x]

    # Crear una copia para la imagen de prueba de verificación
    imagen_prueba = cv2.cvtColor(linea_detectada, cv2.COLOR_GRAY2BGR)

    # Verificar el centro y dibujar las líneas de prueba
    if verificar_centro_candidato(linea_detectada, centro_x, centro_y, imagen_prueba):
        max_distancia = 0
        for dx, dy in direcciones:
            end_x, end_y = centro_x, centro_y
            while 0 <= end_x + dx < cols and 0 <= end_y + dy < rows:
                end_x += dx
                end_y += dy
                if linea_detectada[end_y, end_x] == 255:  # Si encontramos un borde
                    distancia = np.sqrt((end_x - centro_x) ** 2 + (end_y - centro_y) ** 2)
                    if distancia > max_distancia:
                        max_distancia = distancia
                    break

        # Verificar si el radio es mayor que cero antes de dibujar
        if max_distancia > 0:
            resultado = cv2.cvtColor(linea_detectada, cv2.COLOR_GRAY2BGR)
            cv2.circle(resultado, (centro_x, centro_y), int(max_distancia), (0, 255, 0), 2)
            cv2.circle(resultado, (centro_x, centro_y), 2, (0, 0, 255), 3)

            cv2.imshow("Acumulador de puntos medios", (acumulador / max_votos * 255).astype(np.uint8))
            cv2.imshow("Círculo detectado", resultado)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return (centro_x, centro_y), max_distancia
        else:
            print("El radio calculado es cero, no se dibuja el círculo.")
            return None, None
    else:
        print("No se detectó un centro válido.")
        return None, None
        
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

# Función para detectar vecindarios (ya existente)
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

# Cargar imagen en escala de grises
imagen_a_color = cv2.imread('cubo.png')
if imagen_a_color is None:
    print("Error: No se pudo cargar la imagen.")
else:
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
    lineas = cv2.HoughLinesP(bordes_imagen, rho=1, theta=np.pi / 180, threshold=75, minLineLength=3, maxLineGap=1)

    solo_lineas = np.zeros_like(bordes_imagen)
    bordes_restantes = np.copy(bordes_imagen)

    """
    if lineas is not None:
    # Preparar las listas de puntos X e Y
        puntos_X = []
        puntos_Y = []

        for linea in lineas:
            for x1, y1, x2, y2 in linea:
                puntos_X.append([x1])  # x como predictor
                puntos_X.append([x2])
                puntos_Y.append(y1)    # y como valor predicho
                puntos_Y.append(y2)

        # Convertir las listas a matrices numpy
        puntos_X = np.array(puntos_X)
        puntos_Y = np.array(puntos_Y)

        # Detectar varias líneas con RANSAC
        lineas_detectadas = detectar_varias_lineas_ransac(puntos_X, puntos_Y)

        # Dibujar las líneas detectadas en la imagen
        for linea in lineas_detectadas:
            puntos_inliers_X, puntos_inliers_Y = linea
            
            # Encontrar los puntos extremos de los inliers
            if len(puntos_inliers_X) > 1:  # Asegurar que haya al menos dos puntos para formar una línea
                x_start, x_end = int(puntos_inliers_X.min()), int(puntos_inliers_X.max())
                
                # Obtener los valores de y correspondientes a los puntos extremos en x
                y_start = int(puntos_inliers_Y[puntos_inliers_X.argmin()])
                y_end = int(puntos_inliers_Y[puntos_inliers_X.argmax()])
                
                # Dibujar la línea entre los puntos extremos
                cv2.line(imagen_a_color, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    """
    if lineas is not None:
        for linea in lineas:
            for x1, y1, x2, y2 in linea:
                cv2.line(imagen_a_color, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibuja cada línea con color verde
                cv2.line(solo_lineas, (x1, y1), (x2, y2), (255), 2)  # Dibuja cada línea con color verde
                cv2.line(bordes_restantes, (x1, y1), (x2, y2), (0), 2) # Quita las lineas rectas

    np.savetxt('Lineas rectas.csv', solo_lineas, delimiter=',', fmt='%d')
    np.savetxt('Bordes restantes.csv', bordes_restantes, delimiter=',', fmt='%d')

    # Ejemplo de uso con la imagen bordes_restantes
    linea_detectada = detectar_linea_conectada(bordes_restantes)

    # Aplicar la función sobre la imagen de la línea detectada
    centro, radio = encontrar_centro_con_rango_de_busqueda(linea_detectada)
    if centro:
        print(f"Centro del círculo: {centro}")
        print(f"Diámetro del círculo: {2 * radio}")
    else:
        print("No se encontró un círculo válido.")

    # Mostrar la imagen con las líneas detectadas
    cv2.imshow('Vecindarios', output_image)
    cv2.imshow("Bordes de la imagen", bordes_imagen)
    cv2.imshow('Lineas rectas sobrepuestas', imagen_a_color)
    cv2.imshow('Solo lineas rectas', solo_lineas)
    cv2.imshow("Bordes restantes de la imagen", bordes_restantes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
