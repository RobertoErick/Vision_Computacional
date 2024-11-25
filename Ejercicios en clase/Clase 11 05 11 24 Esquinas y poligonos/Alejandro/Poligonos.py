import cv2
import numpy as np
import matplotlib.pyplot as plt

def detectar_esquinas(imagen):
    """Detecta esquinas en una imagen usando el algoritmo de Harris."""
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen

    esquinas = cv2.cornerHarris(gris, blockSize=2, ksize=3, k=0.04)

    _, esquinas_binarias = cv2.threshold(esquinas, 0.01 * esquinas.max(), 255, 0)
    esquinas_binarias = np.uint8(esquinas_binarias)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(esquinas_binarias)

    esquinas_refinadas = []
    for centro in centroids:
        esquinas_refinadas.append((int(centro[0]), int(centro[1])))

    return esquinas_refinadas

def conectar_esquinas(esquinas):
    """Conecta esquinas para formar posibles polígonos."""
    conexiones = []
    usadas = set()
    for i, (x1, y1) in enumerate(esquinas):
        for j, (x2, y2) in enumerate(esquinas):
            if i != j and (i, j) not in usadas and (j, i) not in usadas:
                distancia = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distancia < 50:  
                    conexiones.append(((x1, y1), (x2, y2)))
                    usadas.add((i, j))
    return conexiones

def verificar_poligonos(conexiones):
    """Verifica si las conexiones forman polígonos válidos."""
    poligonos = []
    visitados = set()
    for inicio, fin in conexiones:
        if inicio not in visitados and fin not in visitados:
            poligono = [inicio, fin]
            actual = fin
            while actual != inicio:
                for (a, b) in conexiones:
                    if a == actual and b not in poligono:
                        poligono.append(b)
                        actual = b
                        break
            if len(poligono) >= 3:  
                poligonos.append(poligono)
                visitados.update(poligono)
    return poligonos

def mostrar_resultados(imagen, esquinas, conexiones, poligonos):
    """Muestra la imagen con esquinas, conexiones y polígonos resaltados."""
    imagen_resultado = imagen.copy()
    for esquina in esquinas:
        cv2.circle(imagen_resultado, esquina, 5, (0, 0, 255), -1) 

    for (p1, p2) in conexiones:
        cv2.line(imagen_resultado, p1, p2, (255, 0, 0), 2)  
    for poligono in poligonos:
        for i in range(len(poligono)):
            cv2.line(imagen_resultado, poligono[i], poligono[(i + 1) % len(poligono)], (0, 255, 0), 2)  

    plt.imshow(cv2.cvtColor(imagen_resultado, cv2.COLOR_BGR2RGB))
    plt.title("Detección de Polígonos")
    plt.axis("off")
    plt.show()

ruta_imagen = input("Ingresa la ruta de la imagen: ")
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
else:
    esquinas = detectar_esquinas(imagen)
    conexiones = conectar_esquinas(esquinas)
    poligonos = verificar_poligonos(conexiones)

    print(f"Esquinas detectadas: {len(esquinas)}")
    print(f"Conexiones realizadas: {len(conexiones)}")
    print(f"Polígonos formados: {len(poligonos)}")
    mostrar_resultados(imagen, esquinas, conexiones, poligonos)
