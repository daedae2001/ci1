import os
import logging
import tensorflow as tf
from preprocesamiento_datos import PreprocesadorDatos
from arquitectura_modelo import crear_modelo_cnn

# Configurar logging
logging.basicConfig(
    filename='entrenamiento.log', 
    level=logging.DEBUG,  # Cambiar a DEBUG para más información
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configurar TensorFlow para usar solo CPU con múltiples núcleos
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(15)
tf.config.threading.set_inter_op_parallelism_threads(30)

class ProgresoEntrenamientoCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_callback=None):
        super().__init__()
        self.log_callback = log_callback

    def on_train_batch_end(self, batch, logs=None):
        if self.log_callback and logs:
            progreso = f"{batch+1}/{self.params['steps']} - accuracy: {logs.get('accuracy', 0):.4f} - loss: {logs.get('loss', 0):.4f}"
            self.log_callback(progreso)

    def on_epoch_end(self, epoch, logs=None):
        if self.log_callback and logs:
            mensaje = f"Época {epoch+1}/{self.params['epochs']}"
            for metrica, valor in logs.items():
                mensaje += f" - {metrica}: {valor:.4f}"
            self.log_callback(mensaje)

class EntrenadorModelo:
    def __init__(self, directorio_datos, log_callback=None):
        """
        Inicializar entrenador de modelo
        
        :param directorio_datos: Ruta al directorio con imágenes de entrenamiento
        :param log_callback: Función de callback para enviar logs de progreso
        """
        # Validar directorio de datos
        if not os.path.exists(directorio_datos):
            raise ValueError(f"El directorio de entrenamiento no existe: {directorio_datos}")
        
        if not os.listdir(directorio_datos):
            raise ValueError(f"El directorio de entrenamiento está vacío: {directorio_datos}")

        self.directorio_datos = os.path.abspath(directorio_datos)
        self.modelo = None
        self.historial = None
        self.log_callback = log_callback
        self.clases = None
        logging.info(f"Inicializando entrenador con directorio: {self.directorio_datos}")

    def entrenar(self, epocas=25, altura_img=224, anchura_img=224, tamanio_lote=32):
        """
        Entrenar modelo de clasificación de imágenes
        
        :param epocas: Número de épocas de entrenamiento
        :return: Historial de entrenamiento
        """
        try:
            logging.info(f"Iniciando preprocesamiento de datos")
            # Preprocesar datos
            preprocesador = PreprocesadorDatos(
                self.directorio_datos, 
                altura_img=altura_img, 
                anchura_img=anchura_img, 
                tamanio_lote=tamanio_lote
            )
            generador_entrenamiento, generador_validacion, num_clases = preprocesador.crear_generadores_datos()

            logging.info(f"Creando modelo con {num_clases} clases")
            # Crear modelo
            forma_entrada = (altura_img, anchura_img, 3)
            self.modelo = crear_modelo_cnn(forma_entrada, num_clases)

            # Callbacks
            callbacks = [
                ProgresoEntrenamientoCallback(log_callback=self.log_callback),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=3, 
                    restore_best_weights=True
                )
            ]

            # Entrenar modelo
            logging.info(f"Iniciando entrenamiento por {epocas} épocas")
            self.historial = self.modelo.fit(
                generador_entrenamiento,
                epochs=epocas,
                validation_data=generador_validacion,
                callbacks=callbacks
            )

            # Intentar guardar modelo en varios directorios
            rutas_posibles = [
                os.path.join(os.getcwd(), f"modelo_clasificador_epocas_{epocas}.h5"),
                os.path.join(self.directorio_datos, f"modelo_clasificador_epocas_{epocas}.h5"),
                os.path.join(os.path.dirname(__file__), f"modelo_clasificador_epocas_{epocas}.h5")
            ]

            for ruta_modelo in rutas_posibles:
                try:
                    self.guardar_modelo(ruta_modelo, epocas)
                    break
                except Exception as e:
                    logging.warning(f"No se pudo guardar en {ruta_modelo}: {str(e)}")

            logging.info("Entrenamiento completado exitosamente")
            return self.historial

        except Exception as e:
            logging.error(f"Error durante el entrenamiento: {str(e)}")
            raise

    def guardar_modelo(self, ruta_guardado=None, epocas=5):
        """
        Guardar modelo entrenado con un nombre descriptivo de sus parámetros
        
        :param ruta_guardado: Ruta para guardar el modelo
        :param epocas: Número de épocas de entrenamiento
        """
        try:
            if self.modelo is None:
                raise ValueError("Primero debe entrenar el modelo antes de guardarlo")
            
            if ruta_guardado is None:
                ruta_guardado = f"modelo_clasificador_epocas_{epocas}.h5"
            
            # Asegurarse de que la ruta sea absoluta
            ruta_guardado = os.path.abspath(ruta_guardado)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
            
            # Guardar modelo
            self.modelo.save(ruta_guardado)
            
            # Guardar clases junto con el modelo
            ruta_clases = ruta_guardado.replace('.h5', '_clases.txt')
            with open(ruta_clases, 'w') as f:
                for clase in self.clases:
                    f.write(f"{clase}\n")
            
            logging.info(f"Modelo guardado en: {ruta_guardado}")
            logging.info(f"Clases guardadas en: {ruta_clases}")
            print(f"Modelo guardado en: {ruta_guardado}")
            print(f"Clases guardadas en: {ruta_clases}")
        
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {str(e)}")
            raise
