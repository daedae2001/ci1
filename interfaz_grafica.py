import os
import sys
import logging
import cv2
import numpy as np
import tensorflow as tf
import customtkinter as ctk
import tkinter as tk
import threading
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from entrenamiento_modelo import EntrenadorModelo
from preprocesamiento_datos import PreprocesadorDatos
import time
import tkinter as tk
# Configurar tema moderno de customtkinter
ctk.set_appearance_mode("dark")  # Modo oscuro moderno
ctk.set_default_color_theme("blue")  # Tema de color

class DialogoPersonalizado(ctk.CTkToplevel):
    def __init__(self, master, titulo, mensaje, tipo="info"):
        super().__init__(master)
        self.title(titulo)
        self.geometry("400x200")
        self.resizable(False, False)

        # Configurar color según el tipo de diálogo
        color_fondo = {
            "info": "#2C3E50",
            "error": "#C0392B",
            "advertencia": "#F39C12"
        }.get(tipo, "#2C3E50")

        # Marco principal
        marco = ctk.CTkFrame(self, fg_color=color_fondo)
        marco.pack(padx=20, pady=20, fill="both", expand=True)

        # Ícono
        icono = {
            "info": "ℹ️",
            "error": "❌",
            "advertencia": "⚠️"
        }.get(tipo, "ℹ️")

        # Etiqueta de ícono y mensaje
        etiqueta_icono = ctk.CTkLabel(
            marco, 
            text=icono, 
            font=("Arial", 50)
        )
        etiqueta_icono.pack(pady=(20, 10))

        etiqueta_mensaje = ctk.CTkLabel(
            marco, 
            text=mensaje, 
            font=("Arial", 16), 
            wraplength=350
        )
        etiqueta_mensaje.pack(pady=(0, 20))

        # Botón de aceptar
        boton_aceptar = ctk.CTkButton(
            marco, 
            text="Aceptar", 
            command=self.destroy
        )
        boton_aceptar.pack(pady=10)

        # Centrar ventana
        self.after(10, self._center_window)

    def _center_window(self):
        """Centrar la ventana en la pantalla"""
        self.update_idletasks()
        ancho = self.winfo_width()
        altura = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.winfo_screenheight() // 2) - (altura // 2)
        self.geometry(f'{ancho}x{altura}+{x}+{y}')

class SelectorDirectorio(ctk.CTkToplevel):
    def __init__(self, master, titulo="Seleccionar Directorio", ruta_inicial=None):
        super().__init__(master)
        self.title(titulo)
        self.geometry("800x600")
        self.resizable(True, True)
        self.directorio_seleccionado = None
        
        # Configuración de ventana modal
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Ruta inicial por defecto
        self.ruta_inicial = ruta_inicial or os.path.expanduser("~")

        # Marco principal
        self.marco_principal = ctk.CTkFrame(self)
        self.marco_principal.pack(padx=20, pady=20, fill="both", expand=True)

        # Título
        ctk.CTkLabel(
            self.marco_principal, 
            text="Seleccionar Directorio de Entrenamiento", 
            font=("Arial", 18, "bold")
        ).pack(pady=(10, 20))

        # Entrada de ruta manual
        frame_ruta = ctk.CTkFrame(self.marco_principal, fg_color="transparent")
        frame_ruta.pack(padx=10, pady=(0, 10), fill="x")

        self.entrada_ruta = ctk.CTkEntry(
            frame_ruta, 
            placeholder_text="Ruta del directorio", 
            width=600
        )
        self.entrada_ruta.pack(side="left", padx=(0, 10), expand=True, fill="x")
        
        boton_explorar = ctk.CTkButton(
            frame_ruta, 
            text="🔍", 
            width=50, 
            command=self._explorar_sistema_archivos
        )
        boton_explorar.pack(side="right")

        # Frame de árbol de directorios
        self.frame_arbol = ctk.CTkFrame(self.marco_principal)
        self.frame_arbol.pack(padx=10, pady=10, fill="both", expand=True)

        # Estilo personalizado para el Treeview
        style = ttk.Style()
        style.theme_use('default')
        style.configure(
            "Custom.Treeview", 
            background="#2C3E50", 
            foreground="white", 
            fieldbackground="#2C3E50", 
            font=('Arial', 10)
        )
        style.map(
            "Custom.Treeview", 
            background=[('selected', '#3498DB')]
        )

        # Árbol de directorios
        self.arbol_directorios = self._crear_arbol_directorios(style)
        self.arbol_directorios.pack(padx=10, pady=10, fill="both", expand=True)

        # Eventos de selección y expansión
        self.arbol_directorios.bind('<<TreeviewSelect>>', self._actualizar_ruta_seleccionada)
        self.arbol_directorios.bind('<<TreeviewOpen>>', self._cargar_subdirectorios)

        # Botones
        frame_botones = ctk.CTkFrame(self.marco_principal, fg_color="transparent")
        frame_botones.pack(padx=10, pady=10, fill="x")

        ctk.CTkButton(
            frame_botones, 
            text="Seleccionar", 
            command=self._confirmar_directorio
        ).pack(side="right", padx=10)

        ctk.CTkButton(
            frame_botones, 
            text="Cancelar", 
            command=self.destroy,
            fg_color="gray"
        ).pack(side="right")

        # Cargar árbol desde ruta inicial
        self._cargar_arbol_desde_ruta(self.ruta_inicial)

        # Centrar ventana y esperar a que esté lista
        self.after(100, self._prepare_window)

    def _prepare_window(self):
        """Preparar la ventana para mostrar y bloquear"""
        self.deiconify()  # Asegurar que la ventana sea visible
        self.lift()  # Traer la ventana al frente
        self.focus_force()  # Forzar el foco

    def _on_close(self):
        """Manejar el cierre de la ventana"""
        self.directorio_seleccionado = None
        self.destroy()

    def _center_window(self):
        """Centrar la ventana en la pantalla"""
        self.update_idletasks()
        ancho = self.winfo_width()
        altura = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.winfo_screenheight() // 2) - (altura // 2)
        self.geometry(f'{ancho}x{altura}+{x}+{y}')

    def _crear_arbol_directorios(self, style):
        """Crear árbol de directorios con ttk.Treeview"""
        arbol = ttk.Treeview(
            self.frame_arbol, 
            columns=("Tamaño", "Archivos"), 
            show="tree headings",
            style="Custom.Treeview"
        )
        
        # Configurar columnas
        arbol.heading("#0", text="Directorio")
        arbol.heading("Tamaño", text="Tamaño")
        arbol.heading("Archivos", text="Archivos")
        
        # Estilo de árbol
        arbol.column("#0", width=300)
        arbol.column("Tamaño", width=100, anchor="center")
        arbol.column("Archivos", width=100, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.frame_arbol, orient="vertical", command=arbol.yview)
        arbol.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        return arbol

    def _cargar_arbol_desde_ruta(self, ruta_base):
        """Cargar árbol de directorios desde una ruta base"""
        # Limpiar árbol existente
        for item in self.arbol_directorios.get_children():
            self.arbol_directorios.delete(item)

        # Poblar árbol
        self._poblar_arbol(self.arbol_directorios, ruta_base)

    def _poblar_arbol(self, arbol, ruta_base, padre=""):
        """Poblar árbol de directorios recursivamente"""
        try:
            for entrada in os.scandir(ruta_base):
                if entrada.is_dir() and not entrada.name.startswith('.'):
                    # Obtener información del directorio
                    try:
                        num_archivos = len([f for f in os.listdir(entrada.path) if os.path.isfile(os.path.join(entrada.path, f))])
                        tamanio = sum(os.path.getsize(os.path.join(entrada.path, f)) for f in os.listdir(entrada.path) if os.path.isfile(os.path.join(entrada.path, f))) / 1024 / 1024
                    except:
                        num_archivos = 0
                        tamanio = 0

                    # Insertar en el árbol
                    nodo = arbol.insert(
                        padre, 
                        "end", 
                        text=entrada.name, 
                        values=(f"{tamanio:.2f} MB", f"{num_archivos} archivos"),
                        open=False,
                        tags=('directorio',)
                    )

                    # Agregar un elemento ficticio para permitir expansión
                    arbol.insert(nodo, "end", text="Cargando...", tags=('cargando',))

        except PermissionError:
            pass

    def _cargar_subdirectorios(self, evento):
        """Cargar subdirectorios al expandir un nodo"""
        nodo_seleccionado = self.arbol_directorios.focus()
        if nodo_seleccionado:
            # Eliminar el elemento "Cargando..."
            for hijo in self.arbol_directorios.get_children(nodo_seleccionado):
                if self.arbol_directorios.item(hijo, 'tags')[0] == 'cargando':
                    self.arbol_directorios.delete(hijo)
                    break

            # Obtener ruta completa del directorio
            ruta_directorio = self._obtener_ruta_completa(nodo_seleccionado)

            # Poblar subdirectorios
            self._poblar_arbol(self.arbol_directorios, ruta_directorio, nodo_seleccionado)

    def _actualizar_ruta_seleccionada(self, evento):
        """Actualizar entrada de ruta al seleccionar un directorio"""
        seleccion = self.arbol_directorios.selection()
        if seleccion:
            ruta_completa = self._obtener_ruta_completa(seleccion[0])
            self.entrada_ruta.delete(0, ctk.END)
            self.entrada_ruta.insert(0, ruta_completa)

    def _obtener_ruta_completa(self, nodo):
        """Obtener ruta completa de un nodo del árbol"""
        ruta_componentes = []
        while nodo:
            ruta_componentes.insert(0, self.arbol_directorios.item(nodo, "text"))
            nodo = self.arbol_directorios.parent(nodo)
        return os.path.join(self.ruta_inicial, *ruta_componentes)

    def _explorar_sistema_archivos(self):
        """Abrir diálogo de selección de directorio del sistema"""
        directorio = filedialog.askdirectory(
            title="Seleccionar Directorio de Entrenamiento",
            initialdir=self.ruta_inicial
        )
        if directorio:
            self.entrada_ruta.delete(0, ctk.END)
            self.entrada_ruta.insert(0, directorio)
            self._cargar_arbol_desde_ruta(directorio)

    def _confirmar_directorio(self):
        """Confirmar selección de directorio"""
        ruta = self.entrada_ruta.get()
        if os.path.isdir(ruta):
            self.directorio_seleccionado = ruta
            self.destroy()
        else:
            ctk.CTkMessagebox(
                title="Error", 
                message="La ruta seleccionada no es un directorio válido", 
                icon="cancel"
            )

class AplicacionClasificador:
    def __init__(self, raiz):
        """
        Inicializar aplicación de clasificación de imágenes
        
        :param raiz: Ventana principal de Tkinter
        """
        self.raiz = raiz
        self.raiz.title("Clasificador de Imágenes")
        self.raiz.geometry("800x900")  # Ajustar tamaño para más controles

        # Variables
        self.directorio_datos = None
        self.modelo = None
        self.log_queue = None
        self.estado_modelo = {
            'entrenado': False,
            'epocas': 0,
            'clases': 0,
            'ruta_modelo': None
        }

        # Configurar interfaz
        self.crear_interfaz()
    def procesar_directorio(self, directorio_imagenes):
        """
        Procesar todas las imágenes en un directorio para predicción
        
        Args:
            directorio_imagenes (str): Ruta al directorio con imágenes
        """
        logger = logging.getLogger('ImageClassifier')
        try:
            # Limpiar área de log
            self.area_log.delete("1.0", tk.END)
            
            # Obtener lista de archivos de imagen
            archivos_imagen = [
                os.path.join(directorio_imagenes, f) 
                for f in os.listdir(directorio_imagenes) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]
            
            if not archivos_imagen:
                self.mostrar_mensaje(
                    "Error", 
                    "No se encontraron imágenes en el directorio seleccionado", 
                    "error"
                )
                return
            
            # Deshabilitar botones durante predicción
            self.boton_predecir.configure(state="disabled")
            self.boton_predecir_directorio.configure(state="disabled")
            
            # Procesar cada imagen
            resultados = []
            for imagen_path in archivos_imagen:
                try:
                    # Cargar y preprocesar imagen
                    imagen = Image.open(imagen_path)
                    imagen_procesada = PreprocesadorDatos.preprocesar_imagen(
                        imagen, 
                        tamaño=(224, 224)
                    )
                    
                    # Realizar predicción
                    prediccion = self.modelo.predict(imagen_procesada)
                    clase_predicha = self.clases[np.argmax(prediccion)]
                    
                    # Registrar resultado
                    resultado = f"Imagen: {os.path.basename(imagen_path)} - Clase: {clase_predicha}"
                    resultados.append(resultado)
                    
                    # Mostrar en log
                    self.log_queue = resultado
                    time.sleep(0.1)  # Pequeña pausa para actualizar log
                
                except Exception as e_imagen:
                    logger.error(f"Error procesando imagen {imagen_path}: {e_imagen}")
            
            # Mostrar resumen final
            resumen = "\n".join(resultados)
            self.log_queue = f"\n=== Resultados de Predicción ===\n{resumen}"
            
            # Mostrar mensaje de finalización
            self.mostrar_mensaje(
                "Predicción Completada", 
                f"Procesadas {len(archivos_imagen)} imágenes", 
                "info"
            )
        
        except Exception as e:
            logger.error(f"Error en procesamiento de directorio: {e}", exc_info=True)
            self.mostrar_mensaje(
                "Error", 
                f"Error procesando directorio: {str(e)}", 
                "error"
            )
        finally:
            # Re-habilitar botones
            self.boton_predecir.configure(state="normal")
            self.boton_predecir_directorio.configure(state="normal")
    def actualizar_log(self):
        """
        Método para actualizar el área de log de forma asíncrona
        Maneja la actualización de mensajes en la cola de logs
        """
        while True:
            try:
                # Verificar si hay un mensaje en la cola de logs
                if hasattr(self, 'log_queue') and self.log_queue:
                    # Insertar mensaje en el área de log
                    self.area_log.insert(tk.END, self.log_queue + "\n")
                    
                    # Hacer scroll automático al final
                    self.area_log.see(tk.END)
                    
                    # Limpiar la cola de logs
                    self.log_queue = None
                
                # Pequeña pausa para evitar sobrecarga de CPU
                time.sleep(0.1)
            
            except Exception as e:
                # Manejar cualquier error sin detener el thread
                logging.error(f"Error en actualizar_log: {e}")
                time.sleep(1)
    def predecir_directorio(self):
        """Predecir clases para todas las imágenes en un directorio"""
        # Verificar si hay modelo entrenado
        if not self.modelo or not self.estado_modelo.get('entrenado', False):
            self.mostrar_mensaje(
                "Error", 
                "Debe entrenar o cargar un modelo antes de hacer predicciones", 
                "error"
            )
            return

        # Verificar que haya clases
        if not hasattr(self, 'clases') or not self.clases:
            self.mostrar_mensaje(
                "Error", 
                "No se han definido clases para el modelo", 
                "error"
            )
            return

        # Abrir selector de directorio
        directorio_imagenes = filedialog.askdirectory(
            title="Seleccionar Directorio de Imágenes para Clasificar"
        )

        if not directorio_imagenes:
            return

        # Iniciar predicción en thread separado
        threading.Thread(
            target=self.procesar_directorio, 
            args=(directorio_imagenes,), 
            daemon=True
        ).start()
    def iniciar_entrenamiento(self):
        """Iniciar proceso de entrenamiento"""
        if not self.directorio_datos:
            self.mostrar_mensaje(
                "Error", 
                "Seleccione un directorio con imágenes de entrenamiento", 
                "error"
            )
            return

        # Limpiar área de log
        self.area_log.delete("1.0", "end")
        
        # Deshabilitar botones durante entrenamiento
        self.boton_entrenar.configure(state="disabled")
        self.boton_seleccionar_directorio.configure(state="disabled")
        
        # Iniciar entrenamiento en thread separado
        threading.Thread(target=self.ejecutar_entrenamiento, daemon=True).start()

    def ejecutar_entrenamiento(self):
        """Ejecutar entrenamiento del modelo"""
        try:
            # Función callback para logs
            def actualizar_log(mensaje):
                self.log_queue = mensaje
            
            # Crear y entrenar modelo
            entrenador = EntrenadorModelo(
                self.directorio_datos,
                log_callback=actualizar_log
            )
            
            self.historial = entrenador.entrenar()
            self.modelo = entrenador.modelo
            
            # Actualizar estado
            self.estado_modelo = {
                'entrenado': True,
                'epocas': 25,
                'clases': len(os.listdir(self.directorio_datos)),
                'ruta_modelo': "modelo_entrenado.h5"
            }
            
            # Habilitar botones de predicción
            self.boton_predecir.configure(state="normal")
            self.boton_predecir_directorio.configure(state="normal")
            
            # Mostrar mensaje de éxito
            self.mostrar_mensaje("Éxito", "Entrenamiento completado", "info")
            
        except Exception as e:
            logging.error(f"Error en entrenamiento: {str(e)}", exc_info=True)
            self.mostrar_mensaje("Error", f"Error en entrenamiento: {str(e)}", "error")
        finally:
            # Re-habilitar botones
            self.boton_entrenar.configure(state="normal")
            self.boton_seleccionar_directorio.configure(state="normal")

    def mostrar_mensaje(self, titulo, mensaje, tipo="info"):
        """Mostrar diálogo personalizado"""
        DialogoPersonalizado(self.raiz, titulo, mensaje, tipo)

    def seleccionar_directorio(self):
        """Seleccionar directorio de entrenamiento con diálogo personalizado"""
        selector = SelectorDirectorio(self.raiz)
        self.raiz.wait_window(selector)

        if selector.directorio_seleccionado:
            self.directorio_datos = selector.directorio_seleccionado
            self.entrada_directorio.delete(0, ctk.END)
            self.entrada_directorio.insert(0, self.directorio_datos)

    def crear_interfaz(self):
        """Crear elementos de la interfaz gráfica"""
        # Marco principal
        marco = ctk.CTkFrame(self.raiz)
        marco.pack(padx=20, pady=20, fill="both", expand=True)

        # Título
        titulo = ctk.CTkLabel(marco, text="Clasificador de Imágenes", font=("Helvetica", 20))
        titulo.pack(pady=10)

        # Sección de carga de modelo
        seccion_modelo = ctk.CTkFrame(marco)
        seccion_modelo.pack(padx=20, pady=10, fill="x")

        # Botón para cargar modelo existente
        boton_cargar_modelo = ctk.CTkButton(
            seccion_modelo, 
            text="Cargar Modelo", 
            command=self.cargar_modelo
        )
        boton_cargar_modelo.pack(side="left", padx=10)

        # Etiqueta de estado del modelo
        self.etiqueta_estado_modelo = ctk.CTkLabel(
            seccion_modelo, 
            text="Ningún modelo cargado", 
            text_color="orange"
        )
        self.etiqueta_estado_modelo.pack(side="left", padx=10)

        # Sección de entrenamiento
        seccion_entrenamiento = ctk.CTkFrame(marco)
        seccion_entrenamiento.pack(padx=20, pady=10, fill="x")

        # Entrada de directorio de datos
        ctk.CTkLabel(seccion_entrenamiento, text="Directorio de Entrenamiento:").pack(side="left", padx=(0, 10))
        self.entrada_directorio = ctk.CTkEntry(seccion_entrenamiento, width=400)
        self.entrada_directorio.pack(side="left", expand=True, fill="x", padx=(0, 10))

        boton_seleccionar_directorio = ctk.CTkButton(
            seccion_entrenamiento, 
            text="🔍", 
            width=50, 
            command=self.seleccionar_directorio
        )
        boton_seleccionar_directorio.pack(side="left")

        # Botón de entrenamiento
        boton_entrenar = ctk.CTkButton(
            marco, 
            text="Entrenar Modelo", 
            command=self.iniciar_entrenamiento
        )
        boton_entrenar.pack(pady=10)

        # Sección de predicción
        seccion_prediccion = ctk.CTkFrame(marco)
        seccion_prediccion.pack(padx=20, pady=10, fill="x")

        # Botón para predecir imagen individual
        boton_predecir = ctk.CTkButton(
            seccion_prediccion, 
            text="Predecir Imagen", 
            command=self.predecir_imagen,
            state="disabled"  # Inicialmente desactivado
        )
        boton_predecir.pack(side="left", padx=10)
        self.boton_predecir = boton_predecir

        # Botón para predicción masiva
        boton_predecir_directorio = ctk.CTkButton(
            seccion_prediccion, 
            text="Predecir Directorio", 
            command=self.predecir_directorio,
            state="disabled"  # Inicialmente desactivado
        )
        boton_predecir_directorio.pack(side="left", padx=10)
        self.boton_predecir_directorio = boton_predecir_directorio

        # Área de resultados de predicción
        self.area_resultados = ctk.CTkTextbox(marco, height=300)
        self.area_resultados.pack(pady=20, padx=20, fill="x")

        # Área de log en tiempo real
        self.area_log = ctk.CTkTextbox(marco, height=200)
        self.area_log.pack(padx=20, pady=10, fill="both", expand=True)

        # Iniciar thread para actualizar log
        self.log_thread = threading.Thread(target=self.actualizar_log, daemon=True)
        self.log_thread.start()

    def cargar_modelo(self, ruta_modelo=None):
        logger = logging.getLogger('ImageClassifier')
        try:
            logger.info("Iniciando carga del modelo...")
            if not ruta_modelo:
                ruta_modelo = filedialog.askopenfilename(
                    title="Seleccionar Modelo Entrenado",
                    initialdir=os.path.dirname(os.path.abspath(__file__)),
                    filetypes=[("Modelos Keras", "*.h5"), ("Todos los archivos", "*.*")]
                )
    
            if not ruta_modelo:
                logger.warning("No se seleccionó ningún modelo")
                return False
    
            logger.info(f"Cargando modelo desde: {ruta_modelo}")
            self.modelo = tf.keras.models.load_model(ruta_modelo)
            
            # Obtener número de clases usando units de la última capa
            ultima_capa = self.modelo.layers[-1]
            num_clases = ultima_capa.units
            
            # Intentar cargar clases desde archivo
            ruta_clases = ruta_modelo.replace('.h5', '_clases.txt')
            if os.path.exists(ruta_clases):
                with open(ruta_clases, 'r') as f:
                    self.clases = [linea.strip() for linea in f.readlines()]
            else:
                # Generar clases genéricas
                self.clases = [f'Clase {i}' for i in range(num_clases)]
            
            # Actualizar estado del modelo
            nombre_modelo = os.path.basename(ruta_modelo)
            self.estado_modelo = {
                'entrenado': True,
                'clases': num_clases,
                'ruta_modelo': ruta_modelo
            }
            
            # Actualizar interfaz
            self.etiqueta_estado_modelo.configure(text=f"Modelo Cargado: {nombre_modelo}")
            self.boton_predecir.configure(state="normal")
            self.boton_predecir_directorio.configure(state="normal")
            
            logger.info(f"Modelo cargado exitosamente: {nombre_modelo}")
            self.mostrar_mensaje("Éxito", f"Modelo cargado correctamente\nClases: {num_clases}", "info")
            return True
    
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}", exc_info=True)
            self.mostrar_mensaje("Error", f"Error al cargar el modelo: {str(e)}", "error")
            
            # Restablecer estado
            self.modelo = None
            self.clases = []
            self.estado_modelo = {'entrenado': False}
            self.boton_predecir.configure(state="disabled")
            self.boton_predecir_directorio.configure(state="disabled")
            
            return False

    def seleccionar_modelo(self):
        """Seleccionar un modelo existente para cargar"""
        ruta_modelo = filedialog.askopenfilename(
            title="Seleccionar Modelo Entrenado", 
            initialdir="/home/dae/CascadeProjects/clasificador_imagenes",
            filetypes=[("Modelos de Keras", "*.h5")]
        )

        if not ruta_modelo:
            return

        try:
            # Cargar modelo
            self.cargar_modelo(ruta_modelo)
            
            # Habilitar botón de predicción
            self.boton_predecir.configure(state="normal")
            self.boton_predecir_directorio.configure(state="normal")
            
        except Exception as e:
            # Mostrar mensaje de error en una sola ventana
            self.mostrar_mensaje(
                "Error al Cargar Modelo", 
                f"No se pudo cargar el modelo:\n{str(e)}", 
                "error"
            )

    def predecir_imagen(self):
        """Predecir clase de una imagen"""
        # Verificar si hay modelo entrenado o cargado
        if not self.modelo or not self.estado_modelo.get('entrenado', False):
            self.mostrar_mensaje("Error", "Debe entrenar o cargar un modelo antes de hacer predicciones", "error")
            return

        # Verificar que haya clases
        if not hasattr(self, 'clases') or not self.clases:
            self.mostrar_mensaje("Error", "No se han definido clases para el modelo", "error")
            return

        # Abrir selector de imagen
        ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar Imagen para Clasificar", 
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not ruta_imagen:
            return

        try:
            # Cargar y preprocesar imagen
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")
            
            imagen_original = imagen.copy()  # Guardar imagen original para mostrar
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen = cv2.resize(imagen, (224, 224))
            imagen = imagen / 255.0  # Normalizar
            imagen = np.expand_dims(imagen, axis=0)

            # Predecir
            predicciones = self.modelo.predict(imagen)[0]
            
            # Manejar caso de pocas clases
            if len(predicciones) < len(self.clases):
                logging.warning(f"Discrepancia entre predicciones ({len(predicciones)}) y clases ({len(self.clases)})")
                # Ajustar clases si es necesario
                self.clases = self.clases[:len(predicciones)]

            indice_clase = np.argmax(predicciones)
            clase_predicha = self.clases[indice_clase]
            probabilidad = predicciones[indice_clase] * 100

            # Convertir imagen para mostrar
            imagen_mostrar = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)

            # Crear ventana con imagen y predicción
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(imagen_mostrar)
            plt.title("Imagen Original")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.bar(self.clases, predicciones * 100)
            plt.title(f"Predicciones ({clase_predicha})")
            plt.xlabel("Clases")
            plt.ylabel("Probabilidad (%)")
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            plt.show()

            # Limpiar área de resultados
            self.area_resultados.delete("1.0", "end")
            self.area_resultados.insert("1.0", 
                f"Predicción para {os.path.basename(ruta_imagen)}:\n"
                f"Clase: {clase_predicha}\n"
                f"Probabilidad: {probabilidad:.2f}%\n"
                f"Ruta: {ruta_imagen}"
            )

            logging.info(f"Predicción realizada: {clase_predicha} ({probabilidad:.2f}%)")

        except Exception as e:
            logging.error(f"Error en predicción: {e}", exc_info=True)
            self.mostrar_mensaje(
                "Error de Predicción", 
                f"No se pudo realizar la predicción: {e}", 
                "error"
            )
def iniciar_entrenamiento(self):
    """Iniciar proceso de entrenamiento"""
    if not self.directorio_datos:
        self.mostrar_mensaje(
            "Error", 
            "Seleccione un directorio con imágenes de entrenamiento", 
            "error"
        )
        return

    # Limpiar área de log
    self.area_log.delete("1.0", "end")
    
    # Deshabilitar botones durante entrenamiento
    self.boton_entrenar.configure(state="disabled")
    self.boton_seleccionar_directorio.configure(state="disabled")
    
    # Iniciar entrenamiento en thread separado
    threading.Thread(target=self.ejecutar_entrenamiento, daemon=True).start()

def ejecutar_entrenamiento(self):
    """Ejecutar entrenamiento del modelo"""
    try:
        # Función callback para logs
        def actualizar_log(mensaje):
            self.log_queue = mensaje
        
        # Crear y entrenar modelo
        entrenador = EntrenadorModelo(
            self.directorio_datos,
            log_callback=actualizar_log
        )
        
        self.historial = entrenador.entrenar()
        self.modelo = entrenador.modelo
        
        # Actualizar estado
        self.estado_modelo = {
            'entrenado': True,
            'epocas': 25,
            'clases': len(os.listdir(self.directorio_datos)),
            'ruta_modelo': "modelo_entrenado.h5"
        }
        
        # Habilitar botones de predicción
        self.boton_predecir.configure(state="normal")
        self.boton_predecir_directorio.configure(state="normal")
        
        # Mostrar mensaje de éxito
        self.mostrar_mensaje("Éxito", "Entrenamiento completado", "info")
        
    except Exception as e:
        logging.error(f"Error en entrenamiento: {str(e)}", exc_info=True)
        self.mostrar_mensaje("Error", f"Error en entrenamiento: {str(e)}", "error")
    finally:
        # Re-habilitar botones
        self.boton_entrenar.configure(state="normal")
        self.boton_seleccionar_directorio.configure(state="normal")
def procesar_directorio(self):
    logger = logging.getLogger('ImageClassifier')
    try:
        directorio = filedialog.askdirectory(title="Seleccionar Directorio de Imágenes")
        if not directorio:
            return

        logger.info(f"Procesando directorio: {directorio}")
        self.area_resultados.delete("1.0", "end")
        
        for archivo in os.listdir(directorio):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_completa = os.path.join(directorio, archivo)
                try:
                    logger.info(f"Procesando imagen: {archivo}")
                    # Tu código de procesamiento de imagen aquí
                    resultado = self.clasificar_imagen(ruta_completa)
                    self.area_resultados.insert("end", f"{archivo}: {resultado}\n")
                    self.area_resultados.see("end")
                except Exception as e:
                    logger.error(f"Error procesando {archivo}: {str(e)}")
                    self.area_resultados.insert("end", f"Error en {archivo}: {str(e)}\n")

        logger.info("Procesamiento de directorio completado")
        self.log_queue = "Procesamiento de directorio completado"

    except Exception as e:
        logger.error(f"Error en procesamiento por lotes: {str(e)}", exc_info=True)
        self.mostrar_mensaje("Error", str(e), "error")
    def iniciar_entrenamiento(self):
        """Iniciar entrenamiento"""
        # Validar directorio
        if not self.directorio_datos:
            self.mostrar_mensaje(
                "Error", 
                "Por favor, seleccione un directorio de entrenamiento", 
                "error"
            )
            return

        # Limpiar log anterior
        self.area_log.delete("1.0", "end")

        # Iniciar entrenamiento en thread
        threading.Thread(target=self.ejecutar_entrenamiento, daemon=True).start()

    def ejecutar_entrenamiento(self):
        """Ejecutar entrenamiento"""
        try:
            # Función para agregar logs a la cola
            def log_callback(mensaje):
                self.log_queue = mensaje

            # Inicializar entrenador con callback de log
            entrenador = EntrenadorModelo(self.directorio_datos, log_callback=log_callback)
            
            # Entrenar modelo
            historial = entrenador.entrenar()
            
            # Guardar modelo
            entrenador.guardar_modelo()
            self.modelo = entrenador.modelo

            # Actualizar estado del modelo
            self.estado_modelo = {
                'entrenado': True, 
                'epocas': 5,
                'clases': len(os.listdir(self.directorio_datos)),
                'ruta_modelo': f"modelo_clasificador_epocas_5.h5"
            }

            # Mostrar mensaje final
            self.log_queue = "Entrenamiento completado exitosamente"
        except Exception as e:
            self.log_queue = f"Error de entrenamiento: {str(e)}"
            self.mostrar_mensaje(
                "Error de Entrenamiento", 
                str(e), 
                "error"
            )

    def actualizar_log(self):
        """Actualizar área de log en tiempo real"""
        while True:
            try:
                if self.log_queue:
                    msg = str(self.log_queue)
                    self.area_log.insert("end", msg + "\n")
                    self.area_log.see("end")
                    self.log_queue = None
            except Exception:
                pass

def main():
    raiz = ctk.CTk()
    app = AplicacionClasificador(raiz)
    raiz.mainloop()

if __name__ == "__main__":
    main()
