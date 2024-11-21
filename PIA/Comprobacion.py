import numpy as np
resultados = np.load('resultados_avp_dct.npy', allow_pickle=True).item()
print(type(resultados))  # Debería mostrar <class 'dict'>
