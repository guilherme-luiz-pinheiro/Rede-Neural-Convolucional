import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Define o diretório de teste
test_dir = "C:/Treinamento IA/Test"

# Carrega o modelo salvo
model = tf.keras.models.load_model('C:/Treinamento IA/result.keras')

# Percorre todas as imagens na pasta de teste
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    
    try:
        # Carrega e processa a imagem
        test_image = image.load_img(img_path, target_size=(480, 640))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)  # Adiciona uma dimensão para batch
        
        # Faz a previsão
        result = model.predict(test_image)
        
        # Exibe o resultado
        label = 'Healthy' if result[0][0] >= 0.5 else 'Greening'
        print(f'Imagem: {img_name} -> {label}')
    
    except Exception as e:
        print(f'Erro ao processar {img_name}: {e}')
