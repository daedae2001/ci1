import tensorflow as tf
import os

def configurar_gpu():
    """Configurar GPU de manera segura"""
    # Configurar variables de entorno para CUDA
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['PATH'] = f"{os.environ.get('CUDA_HOME', '')}/bin:{os.environ.get('PATH', '')}"
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('CUDA_HOME', '')}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

    # Intentar configurar GPU de manera segura
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Usar solo la primera GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("GPU configurada correctamente:", gpus)
        else:
            print("No se encontraron GPUs. Usando CPU.")
            # Forzar uso de CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    except Exception as e:
        print(f"Error al configurar GPU: {e}")
        print("Usando CPU por defecto.")
        # Forzar uso de CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Llamar a la función de configuración
configurar_gpu()
