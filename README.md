# Rede Neural Convolucional para Classificação de Imagens

Este projeto implementa uma Rede Neural Convolucional (CNN) utilizando TensorFlow e Keras para classificação binária de imagens. O modelo é treinado com imagens organizadas em diretórios, utilizando data augmentation.

## 📌 Características do Modelo
- Arquitetura baseada em camadas convolucionais e pooling.
- Regularização L2 para evitar overfitting.
- Uso de dropout para melhorar a generalização.
- Otimizador Adam com taxa de aprendizado de 0.0001.
- Função de perda `binary_crossentropy` para classificação binária.

## 📁 Estrutura do Projeto
```
C:/Treinamento IA/
│── Train/
│   ├── Classe_1/
│   ├── Classe_2/
│── result.keras  (Modelo treinado salvo)
│── script.py (Código do modelo)
```

## 🔧 Dependências
Este projeto requer as seguintes bibliotecas:
```bash
pip install numpy tensorflow matplotlib
```

## 🚀 Como Executar
1. Certifique-se de que suas imagens estão organizadas em subpastas dentro do diretório `Train/`.
2. Execute o script Python:
```bash
python script.py
```
3. O modelo treinado será salvo no arquivo `result.keras`.

## 📊 Resultados
Durante o treinamento, os resultados de acurácia e perda serão exibidos no terminal e podem ser analisados usando a biblioteca Matplotlib.

## 📌 Ajustes e Melhorias
- Ajuste a taxa de aprendizado e dropout para melhorar a precisão.
- Utilize data augmentation para aumentar a variabilidade do conjunto de treinamento.
- Teste diferentes arquiteturas de camadas convolucionais para otimizar o desempenho.

## 📜 Licença
Este projeto é de uso livre e pode ser modificado conforme necessário.

