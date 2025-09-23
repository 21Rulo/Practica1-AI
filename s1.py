import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image
import cv2
import numpy as np
import os
from math import log2
from scipy.stats import skew

# Importaciones de Matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 1. Funciones de Procesamiento ---

def separar_canales(imagen_pil):
    """Función pura que solo separa los canales y los devuelve."""
    # Asegura que la imagen esté en modo RGB para poder separarla
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    return imagen_pil.split()

def dibujar_canales_rgb(fig, r, g, b):
    """Dibuja los tres canales en una figura de Matplotlib."""
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3) # Forma actual de crear subplots

    ax1.imshow(r, cmap='Reds')
    ax1.set_title("Componente R")
    ax1.axis('off')

    ax2.imshow(g, cmap='Greens')
    ax2.set_title("Componente G")
    ax2.axis('off')

    ax3.imshow(b, cmap='Blues')
    ax3.set_title("Componente B")
    ax3.axis('off')
    
    fig.tight_layout() # Ajusta los subplots para que no se superpongan

def convertir_a_grises_en_ax(imagen_pil, ax):
    """Convierte la imagen a grises y la dibuja en un eje de Matplotlib."""
    ax.clear()
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    gris_cv = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    ax.imshow(gris_cv, cmap='gray')
    ax.set_title("Imagen en escala de grises")
    ax.axis('off')
    return gris_cv

def binarizar_imagen_en_ax(imagen_gris_cv, ax, umbral=128):
    """Binariza la imagen y la dibuja en un eje de Matplotlib."""
    ax.clear()
    _, binaria = cv2.threshold(imagen_gris_cv, umbral, 255, cv2.THRESH_BINARY)
    ax.imshow(binaria, cmap='gray')
    ax.set_title(f"Imagen binarizada (umbral = {umbral})")
    ax.axis('off')
    return binaria

def calcular_histograma_rgb(imagen_pil):
    """Calcula histogramas para los canales RGB de una imagen PIL."""
    imagen_np = np.array(imagen_pil)
    if len(imagen_np.shape) == 3:  # Imagen a color
        hist_r = cv2.calcHist([imagen_np], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([imagen_np], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([imagen_np], [2], None, [256], [0, 256])
        return hist_r, hist_g, hist_b
    else:  # Imagen en escala de grises
        hist = cv2.calcHist([imagen_np], [0], None, [256], [0, 256])
        return hist

def dibujar_histograma_imagen_original(fig, imagen_pil):
    """Dibuja la imagen original y su histograma RGB."""
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    
    # Mostrar imagen
    ax_img.imshow(imagen_pil)
    ax_img.set_title("Imagen Original")
    ax_img.axis('off')
    
    # Calcular y mostrar histograma RGB
    hist_r, hist_g, hist_b = calcular_histograma_rgb(imagen_pil)
    
    ax_hist.plot(hist_r, color='red', alpha=0.7, label='Canal R')
    ax_hist.plot(hist_g, color='green', alpha=0.7, label='Canal G')
    ax_hist.plot(hist_b, color='blue', alpha=0.7, label='Canal B')
    ax_hist.set_title("Histograma RGB")
    ax_hist.set_xlabel("Intensidad de Pixel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    fig.tight_layout()

def dibujar_histograma_canales_rgb(fig, r, g, b):
    """Dibuja los canales RGB y sus histogramas individuales."""
    fig.clear()
    # Crear una cuadrícula de 2x3 (canales arriba, histogramas abajo)
    gs = fig.add_gridspec(2, 3, hspace=0.3)
    
    # Canales RGB en la fila superior
    ax_r = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    
    ax_r.imshow(r, cmap='Reds')
    ax_r.set_title("Canal R")
    ax_r.axis('off')
    
    ax_g.imshow(g, cmap='Greens')
    ax_g.set_title("Canal G")
    ax_g.axis('off')
    
    ax_b.imshow(b, cmap='Blues')
    ax_b.set_title("Canal B")
    ax_b.axis('off')
    
    # Histogramas en la fila inferior
    ax_hist_r = fig.add_subplot(gs[1, 0])
    ax_hist_g = fig.add_subplot(gs[1, 1])
    ax_hist_b = fig.add_subplot(gs[1, 2])
    
    # Calcular histogramas
    hist_r = cv2.calcHist([np.array(r)], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([np.array(g)], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([np.array(b)], [0], None, [256], [0, 256])
    
    ax_hist_r.plot(hist_r, color='red')
    ax_hist_r.set_title("Histograma R")
    ax_hist_r.grid(True, alpha=0.3)
    
    ax_hist_g.plot(hist_g, color='green')
    ax_hist_g.set_title("Histograma G")
    ax_hist_g.grid(True, alpha=0.3)
    
    ax_hist_b.plot(hist_b, color='blue')
    ax_hist_b.set_title("Histograma B")
    ax_hist_b.grid(True, alpha=0.3)

def dibujar_histograma_escala_grises(fig, imagen_gris_cv):
    """Dibuja la imagen en escala de grises y su histograma."""
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    
    # Mostrar imagen
    ax_img.imshow(imagen_gris_cv, cmap='gray')
    ax_img.set_title("Imagen en Escala de Grises")
    ax_img.axis('off')
    
    # Calcular y mostrar histograma
    hist = cv2.calcHist([imagen_gris_cv], [0], None, [256], [0, 256])
    ax_hist.plot(hist, color='black')
    ax_hist.set_title("Histograma Escala de Grises")
    ax_hist.set_xlabel("Intensidad de Pixel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.grid(True, alpha=0.3)
    
    fig.tight_layout()

def dibujar_histograma_binaria(fig, imagen_binaria_cv):
    """Dibuja la imagen binarizada y su histograma."""
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    
    # Mostrar imagen
    ax_img.imshow(imagen_binaria_cv, cmap='gray')
    ax_img.set_title("Imagen Binarizada")
    ax_img.axis('off')
    
    # Calcular y mostrar histograma
    hist = cv2.calcHist([imagen_binaria_cv], [0], None, [256], [0, 256])
    ax_hist.bar([0, 255], [hist[0].item(), hist[255].item()], color='black', width=50)
    ax_hist.set_title("Histograma Binario")
    ax_hist.set_xlabel("Intensidad de Pixel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.set_xlim(-25, 280)
    ax_hist.grid(True, alpha=0.3)
    
    fig.tight_layout()

# --- FUNCIONES PARA CALCULAR CARACTERÍSTICAS ESTADÍSTICAS ---
def calcular_caracteristicas_estadisticas(imagen_pil):
    """Calcula las características estadísticas de cada canal RGB."""
    imagen_rgb = np.array(imagen_pil)
    
    # Si la imagen no está en RGB, convertirla
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
        imagen_rgb = np.array(imagen_pil)
    
    resultados = {}
    
    for i, canal in enumerate(['Red', 'Green', 'Blue']):
        datos = imagen_rgb[:, :, i].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
        
        # Evitar división por cero
        if histograma.sum() == 0:
            continue
            
        prob = histograma / histograma.sum()
        
        # Cálculos estadísticos
        energia = np.sum(prob ** 2)
        entropia = -np.sum([p * log2(p) for p in prob if p > 0])
        asimetria = skew(datos)
        media = np.mean(datos)
        varianza = np.var(datos)
        
        resultados[canal] = {
            'Energía': energia,
            'Entropía': entropia,
            'Asimetría': asimetria,
            'Media': media,
            'Varianza': varianza
        }
    
    return resultados

def calcular_caracteristicas_escala_grises(imagen_gris_cv):
    """Calcula las características estadísticas de una imagen en escala de grises."""
    datos = imagen_gris_cv.flatten()
    histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
    
    # Evitar división por cero
    if histograma.sum() == 0:
        return None
        
    prob = histograma / histograma.sum()
    
    # Cálculos estadísticos
    energia = np.sum(prob ** 2)
    entropia = -np.sum([p * log2(p) for p in prob if p > 0])
    asimetria = skew(datos)
    media = np.mean(datos)
    varianza = np.var(datos)
    
    return {
        'Energía': energia,
        'Entropía': entropia,
        'Asimetría': asimetria,
        'Media': media,
        'Varianza': varianza
    }

# --- 2. Clase de la Aplicación GUI ---

class MatplotlibApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes")

        self.original_image = None
        self.grayscale_image_cv = None
        self.binary_image_cv = None
        self.current_rgb_channels = None
        self.current_state = "original"  # Estados: "original", "rgb_channels", "grayscale", "binary"

        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Botones existentes
        ttk.Button(control_frame, text="Cargar Imagen", command=self.cargar_imagen).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Separar Canales RGB", command=self.aplicar_separacion_rgb).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Escala de Grises", command=self.aplicar_escala_de_grises).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Binarizar Imagen", command=self.aplicar_binarizacion).pack(side=tk.LEFT, padx=5)
        
        # --- BOTONES MEJORADOS ---
        style = ttk.Style()
        style.configure("Red.TButton", foreground="black", background="red")
        style.configure("Blue.TButton", foreground="black", background="blue")
        style.configure("Orange.TButton", foreground="black", background="orange")
        style.configure("Green.TButton", foreground="black", background="green")

        ttk.Button(control_frame, text="Histograma", command=self.mostrar_histograma, style="Orange.TButton").pack(side=tk.LEFT, padx=5)
        
        # BOTÓN PARA CARACTERÍSTICAS ESTADÍSTICAS
        ttk.Button(control_frame, text="Características", command=self.mostrar_caracteristicas, style="Green.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Guardar Actual", command=self.guardar_imagen_actual, style="Blue.TButton").pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Revertir", command=self.revertir_a_original, style="Red.TButton").pack(side=tk.RIGHT, padx=5)
        
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _preparar_lienzo_unico(self):
        self.fig.clear()
        return self.fig.add_subplot(111)

    def _mostrar_imagen_original(self):
        """Función auxiliar para dibujar la imagen original en el lienzo."""
        ax = self._preparar_lienzo_unico()
        ax.imshow(self.original_image)
        ax.set_title("Imagen Original")
        ax.axis('off')
        self.grayscale_image_cv = None
        self.binary_image_cv = None
        self.current_rgb_channels = None
        self.current_state = "original"
        self.canvas.draw()

    def cargar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(filetypes=[("Archivos de Imagen", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not ruta_imagen:
            return
        self.original_image = Image.open(ruta_imagen)
        self._mostrar_imagen_original()

    def aplicar_separacion_rgb(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        r, g, b = separar_canales(self.original_image)
        self.current_rgb_channels = (r, g, b)
        dibujar_canales_rgb(self.fig, r, g, b)
        self.current_state = "rgb_channels"
        self.canvas.draw()

    def aplicar_escala_de_grises(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        ax = self._preparar_lienzo_unico()
        self.grayscale_image_cv = convertir_a_grises_en_ax(self.original_image, ax)
        self.current_state = "grayscale"
        self.canvas.draw()

    def aplicar_binarizacion(self):
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Primero convierte la imagen a escala de grises.")
            return
        ax = self._preparar_lienzo_unico()
        self.binary_image_cv = binarizar_imagen_en_ax(self.grayscale_image_cv, ax)
        self.current_state = "binary"
        self.canvas.draw()
        
    # --- FUNCIÓN DE HISTOGRAMA ---
    
    def mostrar_histograma(self):
        """Muestra el histograma según el estado actual de la imagen."""
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return

        if self.current_state == "original":
            dibujar_histograma_imagen_original(self.fig, self.original_image)
        elif self.current_state == "rgb_channels":
            if self.current_rgb_channels is None:
                messagebox.showerror("Error", "No hay canales RGB disponibles.")
                return
            r, g, b = self.current_rgb_channels
            dibujar_histograma_canales_rgb(self.fig, r, g, b)
        elif self.current_state == "grayscale":
            if self.grayscale_image_cv is None:
                messagebox.showerror("Error", "No hay imagen en escala de grises disponible.")
                return
            dibujar_histograma_escala_grises(self.fig, self.grayscale_image_cv)
        elif self.current_state == "binary":
            if self.binary_image_cv is None:
                messagebox.showerror("Error", "No hay imagen binaria disponible.")
                return
            dibujar_histograma_binaria(self.fig, self.binary_image_cv)
        
        self.canvas.draw()
    
    # --- MOSTRAR CARACTERÍSTICAS ESTADÍSTICAS ---
    
    def mostrar_caracteristicas(self):
        """Calcula y muestra las características estadísticas según el estado actual de la imagen."""
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        
        try:
            caracteristicas = None
            tipo_imagen = ""
            
            # Determinar qué características calcular según el estado actual
            if self.current_state == "original":
                caracteristicas = calcular_caracteristicas_estadisticas(self.original_image)
                tipo_imagen = "Imagen Original (RGB)"
            elif self.current_state == "grayscale" and self.grayscale_image_cv is not None:
                caracteristicas_grises = calcular_caracteristicas_escala_grises(self.grayscale_image_cv)
                if caracteristicas_grises:
                    caracteristicas = {"Escala de Grises": caracteristicas_grises}
                tipo_imagen = "Imagen en Escala de Grises"
            elif self.current_state == "binary" and self.binary_image_cv is not None:
                caracteristicas_binaria = calcular_caracteristicas_escala_grises(self.binary_image_cv)
                if caracteristicas_binaria:
                    caracteristicas = {"Imagen Binaria": caracteristicas_binaria}
                tipo_imagen = "Imagen Binarizada"
            elif self.current_state == "rgb_channels":
                # Para canales RGB separados, usar la imagen original
                caracteristicas = calcular_caracteristicas_estadisticas(self.original_image)
                tipo_imagen = "Canales RGB Separados"
            else:
                # Fallback a imagen original
                caracteristicas = calcular_caracteristicas_estadisticas(self.original_image)
                tipo_imagen = "Imagen Original (RGB)"
            
            if not caracteristicas:
                messagebox.showerror("Error", "No se pudieron calcular las características estadísticas.")
                return
            
            # Crear ventana emergente para mostrar resultados
            ventana_stats = tk.Toplevel(self.root)
            ventana_stats.title("Características Estadísticas de la Imagen")
            ventana_stats.geometry("650x450")
            
            # Crear frame con scrollbar
            main_frame = ttk.Frame(ventana_stats)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Título
            ttk.Label(main_frame, text=f"Características Estadísticas", 
                     font=('Arial', 14, 'bold')).pack(pady=(0, 5))
            ttk.Label(main_frame, text=f"Tipo: {tipo_imagen}", 
                     font=('Arial', 11, 'italic')).pack(pady=(0, 10))
            
            # Crear texto con scroll
            text_frame = ttk.Frame(main_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            texto_scroll = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=texto_scroll.yview)
            texto_scroll.configure(yscrollcommand=scrollbar.set)
            
            # Formatear y mostrar resultados
            resultado_texto = ""
            for canal, stats in caracteristicas.items():
                if canal == "Escala de Grises":
                    resultado_texto += f"\n═══ IMAGEN EN ESCALA DE GRISES ═══\n"
                elif canal == "Imagen Binaria":
                    resultado_texto += f"\n═══ IMAGEN BINARIZADA ═══\n"
                else:
                    resultado_texto += f"\n═══ CANAL {canal.upper()} ═══\n"
                resultado_texto += f"Energía:    {stats['Energía']:.6f}\n"
                resultado_texto += f"Entropía:   {stats['Entropía']:.6f}\n"
                resultado_texto += f"Asimetría:  {stats['Asimetría']:.6f}\n"
                resultado_texto += f"Media:      {stats['Media']:.2f}\n"
                resultado_texto += f"Varianza:   {stats['Varianza']:.2f}\n"
                resultado_texto += "─" * 35 + "\n"
            
            # Interpretación básica para escala de grises
            if self.current_state in ["grayscale", "binary"]:
                resultado_texto += "\n📊 INTERPRETACIÓN:\n"
                resultado_texto += "─" * 35 + "\n"
                stats = list(caracteristicas.values())[0]
                
                if stats['Entropía'] > 6:
                    resultado_texto += "• Alta entropía: Imagen con mucha información/textura\n"
                elif stats['Entropía'] < 4:
                    resultado_texto += "• Baja entropía: Imagen con poca variación/más uniforme\n"
                else:
                    resultado_texto += "• Entropía media: Imagen con variación moderada\n"
                    
                if abs(stats['Asimetría']) < 0.5:
                    resultado_texto += "• Distribución simétrica de intensidades\n"
                elif stats['Asimetría'] > 0.5:
                    resultado_texto += "• Distribución sesgada hacia valores bajos (imagen oscura)\n"
                else:
                    resultado_texto += "• Distribución sesgada hacia valores altos (imagen clara)\n"
            
            texto_scroll.insert(tk.END, resultado_texto)
            texto_scroll.config(state=tk.DISABLED)  # Solo lectura
            
            texto_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Frame para botones
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=10)
            
            # Botón para exportar datos
            ttk.Button(button_frame, text="Exportar a archivo", 
                      command=lambda: self.exportar_caracteristicas(caracteristicas, tipo_imagen)).pack(side=tk.LEFT, padx=5)
            
            # Botón para cerrar
            ttk.Button(button_frame, text="Cerrar", 
                      command=ventana_stats.destroy).pack(side=tk.LEFT, padx=5)
                      
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular características: {str(e)}")
    
    def exportar_caracteristicas(self, caracteristicas, tipo_imagen):
        """Exporta las características estadísticas a un archivo de texto."""
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Exportar características estadísticas como..."
        )
        
        if ruta_archivo:
            try:
                with open(ruta_archivo, 'w', encoding='utf-8') as f:
                    f.write("CARACTERÍSTICAS ESTADÍSTICAS DE LA IMAGEN\n")
                    f.write("=" * 45 + "\n")
                    f.write(f"Tipo de imagen: {tipo_imagen}\n\n")
                    
                    for canal, stats in caracteristicas.items():
                        if canal == "Escala de Grises":
                            f.write("IMAGEN EN ESCALA DE GRISES:\n")
                        elif canal == "Imagen Binaria":
                            f.write("IMAGEN BINARIZADA:\n")
                        else:
                            f.write(f"CANAL {canal.upper()}:\n")
                        f.write(f"  Energía:    {stats['Energía']:.6f}\n")
                        f.write(f"  Entropía:   {stats['Entropía']:.6f}\n")
                        f.write(f"  Asimetría:  {stats['Asimetría']:.6f}\n")
                        f.write(f"  Media:      {stats['Media']:.2f}\n")
                        f.write(f"  Varianza:   {stats['Varianza']:.2f}\n")
                        f.write("-" * 30 + "\n")
                
                messagebox.showinfo("Éxito", f"Características exportadas a: {os.path.basename(ruta_archivo)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al exportar: {str(e)}")
        
    # --- FUNCIONES DE GUARDADO  ---
    
    def revertir_a_original(self):
        """Vuelve a mostrar la imagen original en el lienzo."""
        if self.original_image is None:
            messagebox.showerror("Error", "No hay una imagen original cargada.")
            return
        self._mostrar_imagen_original()
        messagebox.showinfo("Revertido", "Los cambios han sido revertidos a la imagen original.")

    
    def guardar_imagen_actual(self):
        """Guarda la imagen según el estado actual de procesamiento."""
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return

        try:
            if self.current_state == "original":
                self._guardar_original()
            elif self.current_state == "rgb_channels":
                self._guardar_canales_individuales()
            elif self.current_state == "grayscale":
                self._guardar_escala_grises()
            elif self.current_state == "binary":
                self._guardar_imagen_binaria()
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"Ocurrió un error: {e}")

    def _guardar_original(self):
        """Guarda la imagen original."""
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar imagen original como..."
        )
        if ruta_archivo:
            self.original_image.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Imagen original guardada como: {os.path.basename(ruta_archivo)}")

    def _guardar_canales_individuales(self):
        """Guarda los canales RGB separados con sus colores característicos."""
        if self.current_rgb_channels is None:
            messagebox.showerror("Error", "No hay canales RGB disponibles.")
            return
            
        ruta_base = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Guardar canales RGB como..."
        )
        if not ruta_base:
            return
            
        directorio, nombre_completo = os.path.split(ruta_base)
        nombre_base, extension = os.path.splitext(nombre_completo)

        r, g, b = self.current_rgb_channels
        
        # Crear imágenes RGB con los colores característicos
        # Canal Rojo: mantener el canal R, poner G y B en 0
        imagen_roja = Image.merge('RGB', (r, Image.new('L', r.size, 0), Image.new('L', r.size, 0)))
        
        # Canal Verde: mantener el canal G, poner R y B en 0
        imagen_verde = Image.merge('RGB', (Image.new('L', g.size, 0), g, Image.new('L', g.size, 0)))
        
        # Canal Azul: mantener el canal B, poner R y G en 0
        imagen_azul = Image.merge('RGB', (Image.new('L', b.size, 0), Image.new('L', b.size, 0), b))
        
        # Guardar las imágenes con colores
        imagen_roja.save(os.path.join(directorio, f"{nombre_base}_R.png"))
        imagen_verde.save(os.path.join(directorio, f"{nombre_base}_G.png"))
        imagen_azul.save(os.path.join(directorio, f"{nombre_base}_B.png"))
        
        messagebox.showinfo("Éxito", f"Canales RGB con colores guardados en:\n{directorio}")

    def _guardar_escala_grises(self):
        """Guarda la imagen en escala de grises."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "No hay imagen en escala de grises disponible.")
            return
            
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar imagen en escala de grises como..."
        )
        if ruta_archivo:
            # Convertir de OpenCV a PIL para guardar
            imagen_pil = Image.fromarray(self.grayscale_image_cv)
            imagen_pil.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Imagen en escala de grises guardada como: {os.path.basename(ruta_archivo)}")

    def _guardar_imagen_binaria(self):
        """Guarda la imagen binarizada."""
        if self.binary_image_cv is None:
            messagebox.showerror("Error", "No hay imagen binaria disponible.")
            return
            
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Guardar imagen binarizada como..."
        )
        if ruta_archivo:
            # Convertir de OpenCV a PIL para guardar
            imagen_pil = Image.fromarray(self.binary_image_cv)
            imagen_pil.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Imagen binarizada guardada como: {os.path.basename(ruta_archivo)}")

# --- 3. Bloque Principal de Ejecución  ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MatplotlibApp(root)
    root.mainloop()