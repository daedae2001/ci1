import os
import shutil

def normalizar_nombres_archivos(directorio_origen, directorio_destino):
    """
    Normalizar nombres de archivos para compatibilidad Windows
    """
    # Crear directorio de destino si no existe
    os.makedirs(directorio_destino, exist_ok=True)
    
    for root, dirs, files in os.walk(directorio_origen):
        for nombre in files:
            # Eliminar caracteres no permitidos
            nombre_normalizado = "".join(
                c for c in nombre 
                if c.isalnum() or c in ['-', '_', '.']
            )
            
            # Truncar si es muy largo
            nombre_normalizado = nombre_normalizado[:255]
            
            # Ruta completa origen y destino
            ruta_origen = os.path.join(root, nombre)
            ruta_destino = os.path.join(
                directorio_destino, 
                nombre_normalizado
            )
            
            # Copiar archivo
            try:
                shutil.copy2(ruta_origen, ruta_destino)
            except Exception as e:
                print(f"Error copiando {nombre}: {e}")

# Ejemplo de uso
normalizar_nombres_archivos(
    '/home/dae/CascadeProjects/clasificador_imagenes', 
    '/home/dae/CascadeProjects/clasificador_imagenes_windows'
)