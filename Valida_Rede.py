import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ============================
# 1. Carregar os dados de validação
# ============================
VAL_DIR = "C:/Treinamento IA/Validate"

val_datagen = ImageDataGenerator(rescale=1. / 255)
val_set = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(480, 640), batch_size=32, class_mode='binary', shuffle=False
)

# ============================
# 2. Carregar o modelo treinado
# ============================
model = tf.keras.models.load_model('C:/Treinamento IA/result.keras')

# ============================
# 3. Avaliação do modelo
# ============================
results = model.evaluate(val_set)
metrics_names = model.metrics_names

# Exibir todas as métricas retornadas
print("Resultados da avaliação:")
for name, value in zip(metrics_names, results):
    print(f"{name}: {value:.4f}")

# Se quiser apenas loss e accuracy
test_loss, test_accuracy = results[:2]
print(f"Acurácia do modelo carregado: {test_accuracy * 100:.2f}%")

# ============================
# 4. Predições no conjunto de validação
# ============================
y_pred_prob = model.predict(val_set)  # Probabilidades de previsão
y_pred = (y_pred_prob > 0.5).astype(int)  # Converte para 0 ou 1

y_true = val_set.classes  # Valores reais das classes

# ============================
# 5. Matriz de Confusão
# ============================
conf_matrix = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

plot_confusion_matrix(conf_matrix, class_names=val_set.class_indices.keys())

# ============================
# 6. Relatório de Classificação
# ============================
report = classification_report(y_true, y_pred, target_names=val_set.class_indices.keys())
print("Relatório de Classificação:")
print(report)
