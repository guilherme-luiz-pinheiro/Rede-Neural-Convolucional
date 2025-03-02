import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

TRAIN_DIR = "C:/Treinamento IA/Train"

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='Same', activation='relu', input_shape=(480, 640, 3),
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),  
    tf.keras.layers.Conv2D(32, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.4),  
    tf.keras.layers.Conv2D(128, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Conv2D(128, (3, 3), padding='Same', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.6),  
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  
              loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_set, epochs=100, validation_data=val_set)

model.save('C:/Treinamento IA/result.keras')
