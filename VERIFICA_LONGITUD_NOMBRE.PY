import os

def check_filename_lengths(directory):
    """
    Verificar longitud de nombres de archivos y directorios
    """
    max_length = 0
    longest_path = ""
    
    for root, dirs, files in os.walk(directory):
        for name in files + dirs:
            full_path = os.path.join(root, name)
            if len(full_path) > max_length:
                max_length = len(full_path)
                longest_path = full_path
    
    print(f"Longitud máxima de ruta: {max_length} caracteres")
    print(f"Ruta más larga: {longest_path}")
    
    # Verificar nombres que podrían causar problemas en Windows
    problematic_names = [
        name for name in files + dirs 
        if len(name) > 255 or any(char in name for char in '<>:"/\\|?*')
    ]
    
    if problematic_names:
        print("Nombres problemáticos:")
        for name in problematic_names:
            print(name)
    else:
        print("No se encontraron nombres problemáticos")

# Ejecutar verificación
check_filename_lengths('/home/dae/CascadeProjects/clasificador_imagenes')