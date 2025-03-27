import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================
# 1. Configuração do Early Stopping
# ============================
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,  # Aumentando o patience para evitar interrupção precoce
    verbose=1,
    mode='max',
    restore_best_weights=True
)

# ============================
# 2. Configuração dos Dados
# ============================
TRAIN_DIR = "C:/Treinamento IA/Train"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_set = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(480, 640), batch_size=32, class_mode='binary', subset='training'
)

val_set = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(480, 640), batch_size=32, class_mode='binary', subset='validation'
)

# ============================
# 3. Definição do Modelo
# ============================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='Same', activation='relu', input_shape=(480, 640, 3),
                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding='Same', activation='relu',
                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), padding='Same', activation='relu',
                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='Same', activation='relu',
                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), padding='Same', activation='relu',
                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='Same', activation='relu',
                           kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.1),  # Dropout reduzido para 0.1

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

# ============================
# 4. Compilação do Modelo
# ============================
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Taxa de aprendizado mais baixa
              loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# ============================
# 5. Configuração do ReduceLROnPlateau
# ============================
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',   # Agora estamos monitorando a precisão de validação
    factor=0.5,               # Fator pelo qual a taxa de aprendizado será reduzida
    patience=5,               # Espera por 5 épocas sem melhoria antes de reduzir a taxa de aprendizado
    verbose=1,                # Exibe as informações durante o treinamento
    mode='max',               # Tentando maximizar a precisão (val_accuracy)
    min_lr=1e-6               # Limite inferior para a taxa de aprendizado
)


# ============================
# 6. Treinamento com EarlyStopping e ReduceLROnPlateau
# ============================
history = model.fit(
    train_set,
    epochs=100,
    validation_data=val_set,
    callbacks=[early_stopping, reduce_lr]
)

# ============================
# 7. Salvamento do Modelo
# ============================
model.save('C:/Treinamento IA/result.keras')

# ============================
# 8. Gráfico de Aprendizado
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
