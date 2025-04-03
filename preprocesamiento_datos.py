import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image


class PreprocesadorDatos:
    def __init__(self, directorio_datos, altura_img=224, anchura_img=224, tamanio_lote=32):
        """
        Inicializar preprocesador de datos
        
        :param directorio_datos: Ruta al directorio con imágenes
        :param altura_img: Altura de redimensionamiento
        :param anchura_img: Anchura de redimensionamiento
        :param tamanio_lote: Tamaño del lote de entrenamiento
        """
        self.directorio_datos = directorio_datos
        self.altura_img = altura_img
        self.anchura_img = anchura_img
        self.tamanio_lote = tamanio_lote
        self.clases = None

    def crear_generadores_datos(self):
        """
        Crear generadores de datos para entrenamiento y validación
        
        :return: Generadores de datos y número de clases
        """
        # Generador de datos con aumento
        generador_datos = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20% para validación
        )

        # Generador de entrenamiento
        generador_entrenamiento = generador_datos.flow_from_directory(
            self.directorio_datos,
            target_size=(self.altura_img, self.anchura_img),
            batch_size=self.tamanio_lote,
            class_mode='categorical',
            subset='training'
        )

        # Generador de validación
        generador_validacion = generador_datos.flow_from_directory(
            self.directorio_datos,
            target_size=(self.altura_img, self.anchura_img),
            batch_size=self.tamanio_lote,
            class_mode='categorical',
            subset='validation'
        )

        # Obtener nombres de clases
        self.clases = list(generador_entrenamiento.class_indices.keys())
        num_clases = len(self.clases)

        return generador_entrenamiento, generador_validacion, num_clases

    @staticmethod
    def preprocesar_imagen(imagen, tamaño=(224, 224)):
        """
        Preprocesar imagen para predicción de modelo

        Args:
            imagen (PIL.Image): Imagen a preprocesar
            tamaño (tuple): Tamaño de redimensionamiento, por defecto (224, 224)

        Returns:
            numpy.ndarray: Imagen preprocesada lista para predicción
        """
        # Redimensionar imagen
        imagen_redimensionada = imagen.resize(tamaño)
        
        # Convertir a array numpy
        imagen_array = np.array(imagen_redimensionada)
        
        # Normalizar valores de pixel
        imagen_normalizada = imagen_array / 255.0
        
        # Añadir dimensión de batch
        imagen_procesada = np.expand_dims(imagen_normalizada, axis=0)
        
        return imagen_procesada