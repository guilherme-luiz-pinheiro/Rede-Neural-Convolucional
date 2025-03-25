import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================
# 1. Configuração do Early Stopping
# ============================
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Reduzido para 5
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# ============================
# 2. Definir a Função de Scheduler
# ============================
def scheduler(epoch, lr):
    # Exemplo de decaimento exponencial da taxa de aprendizado
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)  # Decaimento exponencial

# ============================
# 3. Configuração dos Dados
# ============================
TRAIN_DIR = "C:/Treinamento IA/Train"

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,  # Adicionando rotação
    width_shift_range=0.2,  # Deslocamento horizontal
    height_shift_range=0.2,  # Deslocamento vertical
    brightness_range=[0.8, 1.2],  # Ajuste de brilho
    validation_split=0.2
)

train_set = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(480, 640), batch_size=32, class_mode='binary', subset='training'
)

val_set = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(480, 640), batch_size=32, class_mode='binary', subset='validation'
)

# ============================
# 4. Definição do Modelo
# ============================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='Same', activation='relu', input_shape=(480, 640, 3),
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Reduzido
    tf.keras.layers.Conv2D(32, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(128, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Conv2D(128, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),  # Reduzido de 0.6 para 0.4
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

# ============================
# 5. Compilação do Modelo
# ============================
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# ============================
# 6. Configuração do ReduceLROnPlateau
# ============================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitorar a perda de validação
    factor=0.5,  # Reduz a taxa de aprendizado pela metade
    patience=3,  # Espera 3 épocas sem melhoria
    verbose=1,  # Mostra uma mensagem ao reduzir a taxa de aprendizado
    mode='min',  # A redução ocorre quando a perda diminui
    min_lr=1e-6  # A taxa de aprendizado não pode ser menor que 1e-6
)

# ============================
# 7. Treinamento com EarlyStopping e ReduceLROnPlateau
# ============================
history = model.fit(
    train_set,
    epochs=100,
    validation_data=val_set,
    callbacks=[early_stopping, reduce_lr]  # Usando ReduceLROnPlateau
)

# ============================
# 8. Salvamento do Modelo
# ============================
model.save('C:/Treinamento IA/result.keras')

# ============================
# 9. Gráfico de Aprendizado
# ============================
plt.figure(figsize=(12, 6))

# Plot da Acurácia
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.title('Curva de Acurácia')

# Plot da Perda
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)
plt.title('Curva de Perda')

# Plot da AUC
plt.subplot(1, 3, 3)
plt.plot(history.history['auc'], label='AUC Treino')
plt.plot(history.history['val_auc'], label='AUC Validação')
plt.xlabel('Épocas')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.title('Curva de AUC')

plt.tight_layout()
plt.show()
