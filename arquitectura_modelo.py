import tensorflow as tf
from tensorflow.keras import layers, models, Model

def crear_modelo_cnn(forma_entrada, num_clases):
    """
    Crear modelo de red neuronal convolucional para clasificación de imágenes
    
    :param forma_entrada: Forma de las imágenes de entrada
    :param num_clases: Número de clases a clasificar
    :return: Modelo de Keras compilado
    """
    # Usar Sequential API con Input layer
    modelo = models.Sequential([
        # Capa de entrada
        layers.Input(shape=forma_entrada),
        
        # Capas convolucionales
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Capa de dropout
        layers.Dense(num_clases, activation='softmax')
    ])

    # Compilar modelo
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo
