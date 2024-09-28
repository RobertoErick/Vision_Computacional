import numpy as np
import matplotlib.pyplot as plt

# Datos proporcionados
data = [
    [162, 167, 148, 139, 231, 176, 108, 61, 230, 201, 242, 184, 5, 79],
    [11, 183, 24, 29, 228, 76, 50, 116, 187, 40, 188, 67, 138, 243],
    [71, 27, 93, 125, 151, 240, 58, 212, 199, 77, 34, 82, 55, 173],
    [83, 18, 192, 251, 222, 41, 32, 9, 49, 55, 88, 204, 139, 97],
    [169, 206, 181, 234, 132, 102, 223, 46, 164, 215, 39, 67, 41, 158],
    [236, 144, 34, 250, 10, 95, 24, 238, 207, 218, 242, 164, 208, 12],
    [9, 115, 169, 161, 252, 244, 55, 49, 199, 151, 98, 165, 134, 73],
    [157, 166, 186, 222, 116, 192, 178, 152, 20, 124, 204, 152, 87, 188],
    [43, 1, 83, 250, 68, 201, 62, 10, 132, 75, 116, 154, 122, 180],
    [204, 159, 81, 129, 182, 15, 14, 175, 6, 67, 214, 198, 68, 250],
    [81, 31, 157, 64, 243, 244, 165, 254, 39, 163, 102, 176, 38, 154],
    [15, 209, 160, 51, 185, 205, 51, 43, 242, 222, 109, 47, 165, 243],
    [229, 95, 150, 191, 226, 170, 61, 68, 230, 147, 151, 111, 68, 78]
]

# Convertimos los datos en una matriz NumPy
data_array = np.array(data)

# Creamos la imagen en escala de grises
plt.imshow(data_array, cmap='gray', vmin=0, vmax=255)

# Removemos los ejes
plt.axis('off')

# Guardamos y mostramos la imagen
image_path = '/mnt/data/gray_image.png'
plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
plt.show()

image_path
