import matplotlib.pyplot as plt
import cv2

#Lee la imagen en escala de grises
image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

#Revisar la imagen que se haya cargado correctamente
if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    #Coordenada en que se va a aclarar el pixel de la imagen
    x, y = 120, 100

    #Aplicar el aclarado en la coordenada correspondiente de la imagen
    intensidad = image[y, x]
    nueva_intensidad = min(intensidad + 50, 255)  
    image[y, x] = nueva_intensidad

    #Utilizamos plt para poder ver las coordenadas donde se aclara
    plt.imshow(image, cmap='gray')
    plt.title("Imagen aclarada en un Pixel")
    plt.show()

