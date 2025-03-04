import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

VAL_DIR = "C:/Treinamento IA/Validate"

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_set = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(480, 640), batch_size=32, class_mode='binary', shuffle=False
)

model = tf.keras.models.load_model('C:/Treinamento IA/result.keras')

test_loss, test_accuracy = model.evaluate(val_set)
print(f"Acurácia do modelo carregado: {test_accuracy * 100:.2f}%")

y_pred_prob = model.predict(val_set)
y_pred = (y_pred_prob > 0.5).astype(int)  

y_true = val_set.classes

conf_matrix = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

plot_confusion_matrix(conf_matrix, class_names=val_set.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=val_set.class_indices.keys())
print("Relatório de Classificação:")
print(report)
