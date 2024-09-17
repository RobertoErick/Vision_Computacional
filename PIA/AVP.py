import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para calcular el centroide del contorno
def calcular_centroide(contorno):
    M = cv2.moments(contorno)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

# Función para generar el perfil AVP desde diferentes ángulos
def generar_perfil_avp(contorno, imagen):
    # Obtener el centroide del contorno
    centroide = calcular_centroide(contorno)
    
    if centroide is None:
        raise ValueError("No se pudo calcular el centroide del contorno.")
    
    # Inicializamos un perfil para cada ángulo
    perfil_arriba = []
    perfil_abajo = []
    perfil_izquierda = []
    perfil_derecha = []
    
    # Dimensiones de la imagen
    alto, ancho = imagen.shape[:2]
    
    # Iterar sobre los puntos del contorno para proyectarlos en diferentes ángulos
    for punto in contorno:
        x, y = punto[0]
        
        # Proyección en la dirección hacia arriba (distancia al borde superior)
        distancia_arriba = y - 0  # Distancia desde el punto al borde superior
        perfil_arriba.append(distancia_arriba)
        
        # Proyección en la dirección hacia abajo (distancia al borde inferior)
        distancia_abajo = alto - y  # Distancia desde el punto al borde inferior
        perfil_abajo.append(distancia_abajo)
        
        # Proyección en la dirección hacia la izquierda (distancia al borde izquierdo)
        distancia_izquierda = x - 0  # Distancia desde el punto al borde izquierdo
        perfil_izquierda.append(distancia_izquierda)
        
        # Proyección en la dirección hacia la derecha (distancia al borde derecho)
        distancia_derecha = ancho - x  # Distancia desde el punto al borde derecho
        perfil_derecha.append(distancia_derecha)
    
    # Convertir los perfiles a arrays numpy
    perfil_arriba = np.array(perfil_arriba)
    perfil_abajo = np.array(perfil_abajo)
    perfil_izquierda = np.array(perfil_izquierda)
    perfil_derecha = np.array(perfil_derecha)
    
    return perfil_arriba, perfil_abajo, perfil_izquierda, perfil_derecha

# Función principal para generar el perfil AVP de una hoja
def obtener_perfil_avp(ruta_imagen):
    # Cargar la imagen binaria
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    # Detectar los contornos
    contornos, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        raise ValueError("No se encontraron contornos en la imagen.")
    
    # Usar el contorno más grande (que se asume que es el de la hoja)
    contorno_hoja = max(contornos, key=cv2.contourArea)
    
    # Generar el perfil AVP
    perfil_arriba, perfil_abajo, perfil_izquierda, perfil_derecha = generar_perfil_avp(contorno_hoja, imagen)
    
    return perfil_arriba, perfil_abajo, perfil_izquierda, perfil_derecha

# Función para mostrar los histogramas como líneas
def mostrar_histogramas_lineas(perfil_arriba, perfil_abajo, perfil_izquierda, perfil_derecha):
    # Crear una figura con subgráficos
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Histograma de cada perfil como gráfico de línea
    axs[0, 0].plot(perfil_arriba, color='blue', lw=1)
    axs[0, 0].set_title('Perfil Arriba')
    
    axs[0, 1].plot(perfil_abajo, color='green', lw=1)
    axs[0, 1].set_title('Perfil Abajo')
    
    axs[1, 0].plot(perfil_izquierda, color='red', lw=1)
    axs[1, 0].set_title('Perfil Izquierda')
    
    axs[1, 1].plot(perfil_derecha, color='purple', lw=1)
    axs[1, 1].set_title('Perfil Derecha')
    
    # Mostrar los gráficos
    plt.tight_layout()
    plt.show()

# Función para mostrar el perfil AVP en una sola línea continua
def mostrar_perfil_continuo(perfil_arriba, perfil_abajo, perfil_izquierda, perfil_derecha):
    # Unir los perfiles en el orden: Arriba -> Izquierda -> Abajo -> Derecha
    perfil_continuo = np.concatenate((perfil_arriba, perfil_izquierda, perfil_abajo, perfil_derecha))
    
    # Crear un gráfico de línea del perfil AVP continuo
    plt.figure(figsize=(10, 4))
    plt.plot(perfil_continuo, color='blue', lw=1)
    plt.title("Perfil AVP Continuo de la Hoja")
    plt.xlabel("Puntos del contorno")
    plt.ylabel("Distancia al borde")
    plt.grid(True)
    plt.show()
    
    return perfil_continuo

# Ruta de la imagen binarizada de la hoja
ruta_imagen = 'imagen_preprocesada.png'  

# Obtener los perfiles AVP
perfil_arriba, perfil_abajo, perfil_izquierda, perfil_derecha = obtener_perfil_avp(ruta_imagen)

# Mostrar los histogramas como líneas para cada perfil
mostrar_histogramas_lineas(perfil_arriba, perfil_abajo, perfil_izquierda, perfil_derecha)

# Mostrar el perfil AVP como una sola línea continua
perfil_continuo = mostrar_perfil_continuo(perfil_arriba, perfil_abajo, perfil_izquierda, perfil_derecha)