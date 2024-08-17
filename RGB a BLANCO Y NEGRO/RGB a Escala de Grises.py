import matplotlib.pyplot as plt
import cv2
import numpy as np

# Mostrar gráficos en línea dentro del notebook
# %matplotlib inline

# Leer la imagen 'Lenna.png' con OpenCV y la función imread
image = cv2.imread('Lenna.png')

# Verifica si la imagen se cargó correctamente
if image is None:
    print("Error: No se pudo cargar la imagen. Asegúrate de que el archivo 'Lenna.png' está en el directorio correcto.")
else:
    # Imprimir la matriz de la imagen original (en color)
    print("Matriz de la imagen original (en color):")
    print(image)

    # Guardar la matriz de la imagen original en un archivo CSV
    np.savetxt('imagen_original.csv', image.reshape(-1, image.shape[2]), delimiter=',', fmt='%d')
    print("La matriz de la imagen original se ha guardado en 'imagen_original.csv'.")

    # Obtener información de la imagen usando dtype y shape
    print("\nDimensiones de la imagen original:", image.shape)
    print("Tipo de datos de la imagen original:", image.dtype)

    # Graficar la imagen en color
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Imagen en color (RGB)")
    plt.show()

    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Imprimir la matriz de la imagen en escala de grises
    print("\nMatriz de la imagen en escala de grises:")
    print(gray_image)

    # Guardar la matriz de la imagen en escala de grises en un archivo CSV
    np.savetxt('imagen_gris.csv', gray_image, delimiter=',', fmt='%d')
    print("La matriz de la imagen en escala de grises se ha guardado en 'imagen_gris.csv'.")

    # Obtener información de la imagen en escala de grises usando dtype y shape
    print('\nEsta imagen en escala de grises es de tipo:', gray_image.dtype, 'con dimensiones:', gray_image.shape)

    # Graficar la imagen en escala de grises
    plt.imshow(gray_image, cmap='gray')
    plt.title("Imagen en escala de grises")
    plt.show()
