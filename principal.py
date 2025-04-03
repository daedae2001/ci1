import os
import sys
from logger_config import setup_logger
from configurar_gpu import configurar_gpu
from interfaz_grafica import main

logger = setup_logger()

if __name__ == "__main__":
    logger.info("=== Iniciando clasificador de imágenes ===")
    try:
        configurar_gpu()
        logger.info("GPU configurada correctamente")
        main()
    except Exception as e:
        logger.critical(f"Error fatal: {str(e)}", exc_info=True)