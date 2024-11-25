import cv2
import numpy as np
import matplotlib.pyplot as plt

def es_agujero(val1, val2):
    """Determina si dos valores adyacentes forman un agujero."""
    if val2 == 0: 
        return False
    division = val1 / val2
    return division >= 2 or division <= 0.5

def detectar_agujeros_en_imagen(imagen):
    """
    Detecta agujeros en una imagen.
    Los agujeros se definen como áreas con diferencias significativas según el criterio dado.
    """
    if len(imagen.shape) == 3:  
        imagen_grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_grises = imagen
    
    bordes = cv2.Canny(imagen_grises, 50, 150)

    contornos, _ = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    agujeros = []
    for contorno in contornos:
        for i in range(len(contorno)):
            p1 = contorno[i][0]
            p2 = contorno[(i + 1) % len(contorno)][0]  
            distancia1 = np.linalg.norm(p1)
            distancia2 = np.linalg.norm(p2)
            
            if es_agujero(distancia1, distancia2):
                agujeros.append((tuple(p1), tuple(p2)))
    
    return agujeros, bordes

def mostrar_resultados(imagen_original, bordes, agujeros):
    """Muestra la imagen original, los bordes detectados y los agujeros resaltados."""
    imagen_con_agujeros = imagen_original.copy()
    for p1, p2 in agujeros:
        cv2.line(imagen_con_agujeros, p1, p2, (0, 0, 255), 2)  

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Imagen Original")
    plt.imshow(cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Bordes Detectados")
    plt.imshow(bordes, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Agujeros Detectados")
    plt.imshow(cv2.cvtColor(imagen_con_agujeros, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

ruta_imagen = input("Ingresa la ruta de la imagen: ")
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo cargar la imagen. Verifica la ruta.")
else:
    agujeros, bordes = detectar_agujeros_en_imagen(imagen)
    print(f"Se detectaron {len(agujeros)} agujeros.")
    mostrar_resultados(imagen, bordes, agujeros)
