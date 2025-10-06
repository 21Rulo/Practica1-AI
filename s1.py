# -----------------------------------------------------------------------------
# PROYECTO 1: PROCESADOR DE IMÁGENES DIGITALES
#
# Este script crea una aplicación de escritorio con Tkinter para realizar
# diversas operaciones de procesamiento de imágenes, como la separación de
# canales en modelos RGB, CMY, HSV, YIQ y HSI, conversión a escala de grises,
# binarización y visualización de histogramas.
# -----------------------------------------------------------------------------

# --- Importación de Librerías ---
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from PIL import Image
import cv2
import numpy as np
import os
import colorsys
import math

# Librerías para integrar los gráficos de Matplotlib en Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================================================================
# --- SECCIÓN 1: FUNCIONES DE PROCESAMIENTO DE IMÁGENES ---
# =============================================================================

# (Se omiten las funciones de separación de canales que no cambian para brevedad)
# --- Funciones para el Modelo RGB ---
def separar_canales_rgb(imagen_pil):
    if imagen_pil.mode != 'RGB': imagen_pil = imagen_pil.convert('RGB')
    return imagen_pil.split()

def dibujar_canales_rgb(fig, r, g, b):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    zero_channel = Image.new('L', r.size, 0)
    img_r = Image.merge('RGB', (r, zero_channel, zero_channel))
    img_g = Image.merge('RGB', (zero_channel, g, zero_channel))
    img_b = Image.merge('RGB', (zero_channel, zero_channel, b))
    ax1.imshow(img_r); ax1.set_title("Componente R"); ax1.axis('off')
    ax2.imshow(img_g); ax2.set_title("Componente G"); ax2.axis('off')
    ax3.imshow(img_b); ax3.set_title("Componente B"); ax3.axis('off')
    fig.tight_layout()

# --- Funciones para el Modelo CMY ---
def separar_canales_cmy(imagen_pil):
    if imagen_pil.mode != 'RGB': imagen_pil = imagen_pil.convert('RGB')
    imagen_rgb_np = np.array(imagen_pil)
    imagen_cmy_np = 255 - imagen_rgb_np
    c = Image.fromarray(imagen_cmy_np[:, :, 0])
    m = Image.fromarray(imagen_cmy_np[:, :, 1])
    y = Image.fromarray(imagen_cmy_np[:, :, 2])
    return c, m, y

def dibujar_canales_cmy(fig, c, m, y):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    c_np, m_np, y_np = np.array(c), np.array(m), np.array(y)
    h, w = c_np.shape
    white = np.full((h, w), 255, dtype=np.uint8)
    img_c = np.stack([255 - c_np, white, white], axis=-1)
    img_m = np.stack([white, 255 - m_np, white], axis=-1)
    img_y = np.stack([white, white, 255 - y_np], axis=-1)
    ax1.imshow(img_c); ax1.set_title("Componente C (Cyan)"); ax1.axis('off')
    ax2.imshow(img_m); ax2.set_title("Componente M (Magenta)"); ax2.axis('off')
    ax3.imshow(img_y); ax3.set_title("Componente Y (Yellow)"); ax3.axis('off')
    fig.tight_layout()

# --- Funciones para el Modelo HSV ---
def separar_canales_hsv(imagen_pil):
    if imagen_pil.mode != 'RGB': imagen_pil = imagen_pil.convert('RGB')
    imagen_rgb_np = np.array(imagen_pil)
    imagen_bgr_np = cv2.cvtColor(imagen_rgb_np, cv2.COLOR_RGB2BGR)
    imagen_hsv_np = cv2.cvtColor(imagen_bgr_np, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imagen_hsv_np)
    return h, s, v

def dibujar_canales_hsv(fig, h, s, v):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(h, cmap='hsv'); ax1.set_title("Componente H (Hue)"); ax1.axis('off')
    ax2.imshow(s, cmap='plasma'); ax2.set_title("Componente S (Saturation)"); ax2.axis('off')
    ax3.imshow(v, cmap='gray'); ax3.set_title("Componente V (Value)"); ax3.axis('off')
    fig.tight_layout()

# --- Funciones para el Modelo YIQ ---
def separar_canales_yiq(imagen_pil):
    if imagen_pil.mode != 'RGB': imagen_pil = imagen_pil.convert('RGB')
    rgb_array = np.array(imagen_pil) / 255.0
    yiq_array = np.zeros_like(rgb_array)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            r, g, b = rgb_array[i, j]
            yiq_array[i, j] = colorsys.rgb_to_yiq(r, g, b)
    y = (yiq_array[:, :, 0] * 255).astype(np.uint8)
    i_normalized = ((yiq_array[:, :, 1] - yiq_array[:, :, 1].min()) / 
                    (yiq_array[:, :, 1].max() - yiq_array[:, :, 1].min() + 1e-6) * 255).astype(np.uint8)
    q_normalized = ((yiq_array[:, :, 2] - yiq_array[:, :, 2].min()) / 
                    (yiq_array[:, :, 2].max() - yiq_array[:, :, 2].min() + 1e-6) * 255).astype(np.uint8)
    return y, i_normalized, q_normalized

def dibujar_canales_yiq(fig, y, i, q):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(y, cmap='gray'); ax1.set_title("Componente Y (Luminancia)"); ax1.axis('off')
    ax2.imshow(i, cmap='RdBu'); ax2.set_title("Componente I (Crominancia)"); ax2.axis('off')
    ax3.imshow(q, cmap='PiYG'); ax3.set_title("Componente Q (Crominancia)"); ax3.axis('off')
    fig.tight_layout()

# --- Funciones para el Modelo HSI ---
def separar_canales_hsi(imagen_pil):
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.array(imagen_pil) / 255.0; r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        i = (r + g + b) / 3.0; min_rgb = np.minimum(np.minimum(r, g), b)
        s = 1 - (3 / (r + g + b + 1e-6)) * min_rgb
        num = 0.5 * ((r - g) + (r - b)); den = np.sqrt((r - g)**2 + (r - b) * (g - b))
        theta = np.arccos(np.clip(num / (den + 1e-6), -1, 1)); h = np.copy(theta)
        h[b > g] = (2 * np.pi) - h[b > g]; h = h / (2 * np.pi)
    h_out = (h * 255).astype(np.uint8); s_out = (s * 255).astype(np.uint8); i_out = (i * 255).astype(np.uint8)
    return h_out, s_out, i_out

def dibujar_canales_hsi(fig, h, s, i):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(h, cmap='hsv'); ax1.set_title("Componente H (Hue)"); ax1.axis('off')
    ax2.imshow(s, cmap='plasma'); ax2.set_title("Componente S (Saturation)"); ax2.axis('off')
    ax3.imshow(i, cmap='gray'); ax3.set_title("Componente I (Intensity)"); ax3.axis('off')
    fig.tight_layout()

# --- Funciones para Escala de Grises y Binarización ---
def convertir_a_grises_en_ax(imagen_pil, ax):
    ax.clear(); imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    gris_cv = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    ax.imshow(gris_cv, cmap='gray'); ax.set_title("Imagen en Escala de Grises"); ax.axis('off')
    return gris_cv

def binarizar_imagen_en_ax(imagen_gris_cv, ax, umbral=128):
    ax.clear(); _, binaria = cv2.threshold(imagen_gris_cv, umbral, 255, cv2.THRESH_BINARY)
    ax.imshow(binaria, cmap='gray'); ax.set_title(f"Imagen Binarizada (Umbral = {umbral})"); ax.axis('off')
    return binaria

# --- Funciones para el Dibujo de Histogramas ---
def dibujar_histograma_canales_rgb(fig, r, g, b):
    # (Código de esta función sin cambios)
    fig.clear(); gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    (ax_r, ax_g, ax_b) = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]))
    zero_channel = Image.new('L', r.size, 0)
    ax_r.imshow(Image.merge('RGB', (r, zero_channel, zero_channel))); ax_r.set_title("Canal R"); ax_r.axis('off')
    ax_g.imshow(Image.merge('RGB', (zero_channel, g, zero_channel))); ax_g.set_title("Canal G"); ax_g.axis('off')
    ax_b.imshow(Image.merge('RGB', (zero_channel, zero_channel, b))); ax_b.set_title("Canal B"); ax_b.axis('off')
    (ax_hr, ax_hg, ax_hb) = (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]))
    hr = cv2.calcHist([np.array(r)], [0], None, [256], [0, 256]); hg = cv2.calcHist([np.array(g)], [0], None, [256], [0, 256]); hb = cv2.calcHist([np.array(b)], [0], None, [256], [0, 256])
    ax_hr.plot(hr, color='red'); ax_hr.set_title("Histograma R"); ax_hr.grid(True, alpha=0.3)
    ax_hg.plot(hg, color='green'); ax_hg.set_title("Histograma G"); ax_hg.grid(True, alpha=0.3)
    ax_hb.plot(hb, color='blue'); ax_hb.set_title("Histograma B"); ax_hb.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)

def dibujar_histograma_canales_cmy(fig, c, m, y):
    # (Código de esta función sin cambios)
    fig.clear(); gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    (ax_c, ax_m, ax_y) = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]))
    c_np, m_np, y_np = np.array(c), np.array(m), np.array(y); h, w = c_np.shape; white = np.full((h, w), 255, dtype=np.uint8)
    ax_c.imshow(np.stack([255 - c_np, white, white], axis=-1)); ax_c.set_title("Canal C"); ax_c.axis('off')
    ax_m.imshow(np.stack([white, 255 - m_np, white], axis=-1)); ax_m.set_title("Canal M"); ax_m.axis('off')
    ax_y.imshow(np.stack([white, white, 255 - y_np], axis=-1)); ax_y.set_title("Canal Y"); ax_y.axis('off')
    (ax_hc, ax_hm, ax_hy) = (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]))
    hc = cv2.calcHist([c_np], [0], None, [256], [0, 256]); hm = cv2.calcHist([m_np], [0], None, [256], [0, 256]); hy = cv2.calcHist([y_np], [0], None, [256], [0, 256])
    ax_hc.plot(hc, color='cyan'); ax_hc.set_title("Histograma C"); ax_hc.grid(True, alpha=0.3)
    ax_hm.plot(hm, color='magenta'); ax_hm.set_title("Histograma M"); ax_hm.grid(True, alpha=0.3)
    ax_hy.plot(hy, color='orange'); ax_hy.set_title("Histograma Y"); ax_hy.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)

def dibujar_histograma_canales_hsv(fig, h, s, v):
    # (Código de esta función sin cambios)
    fig.clear(); gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    (ax_h, ax_s, ax_v) = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]))
    ax_h.imshow(h, cmap='hsv'); ax_h.set_title("Canal H"); ax_h.axis('off')
    ax_s.imshow(s, cmap='plasma'); ax_s.set_title("Canal S"); ax_s.axis('off')
    ax_v.imshow(v, cmap='gray'); ax_v.set_title("Canal V"); ax_v.axis('off')
    (ax_hh, ax_hs, ax_hv) = (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]))
    hh = cv2.calcHist([h], [0], None, [180], [0, 180]); hs = cv2.calcHist([s], [0], None, [256], [0, 256]); hv = cv2.calcHist([v], [0], None, [256], [0, 256])
    ax_hh.plot(hh, color='red'); ax_hh.set_title("Histograma H"); ax_hh.grid(True, alpha=0.3)
    ax_hs.plot(hs, color='gray'); ax_hs.set_title("Histograma S"); ax_hs.grid(True, alpha=0.3)
    ax_hv.plot(hv, color='black'); ax_hv.set_title("Histograma V"); ax_hv.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)
    
def dibujar_histograma_canales_yiq(fig, y, i, q):
    # (Código de esta función sin cambios)
    fig.clear(); gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    (ax_y, ax_i, ax_q) = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]))
    ax_y.imshow(y, cmap='gray'); ax_y.set_title("Canal Y"); ax_y.axis('off')
    ax_i.imshow(i, cmap='RdBu'); ax_i.set_title("Canal I"); ax_i.axis('off')
    ax_q.imshow(q, cmap='PiYG'); ax_q.set_title("Canal Q"); ax_q.axis('off')
    (ax_hy, ax_hi, ax_hq) = (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]))
    hy = cv2.calcHist([y], [0], None, [256], [0, 256]); hi = cv2.calcHist([i], [0], None, [256], [0, 256]); hq = cv2.calcHist([q], [0], None, [256], [0, 256])
    ax_hy.plot(hy, color='black'); ax_hy.set_title("Histograma Y"); ax_hy.grid(True, alpha=0.3)
    ax_hi.plot(hi, color='blue'); ax_hi.set_title("Histograma I"); ax_hi.grid(True, alpha=0.3)
    ax_hq.plot(hq, color='purple'); ax_hq.set_title("Histograma Q"); ax_hq.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)

def dibujar_histograma_canales_hsi(fig, h, s, i):
    # (Código de esta función sin cambios)
    fig.clear(); gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    (ax_h, ax_s, ax_i) = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]))
    ax_h.imshow(h, cmap='hsv'); ax_h.set_title("Canal H"); ax_h.axis('off')
    ax_s.imshow(s, cmap='plasma'); ax_s.set_title("Canal S"); ax_s.axis('off')
    ax_i.imshow(i, cmap='gray'); ax_i.set_title("Canal I"); ax_i.axis('off')
    (ax_hh, ax_hs, ax_hi) = (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]))
    hh = cv2.calcHist([h], [0], None, [256], [0, 256]); hs = cv2.calcHist([s], [0], None, [256], [0, 256]); hi = cv2.calcHist([i], [0], None, [256], [0, 256])
    ax_hh.plot(hh, color='red'); ax_hh.set_title("Histograma H"); ax_hh.grid(True, alpha=0.3)
    ax_hs.plot(hs, color='gray'); ax_hs.set_title("Histograma S"); ax_hs.grid(True, alpha=0.3)
    ax_hi.plot(hi, color='black'); ax_hi.set_title("Histograma I"); ax_hi.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)

# --- NUEVAS FUNCIONES DE HISTOGRAMA PARA GRISES Y BINARIO ---
def dibujar_histograma_escala_grises(fig, imagen_gris_cv):
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    ax_img.imshow(imagen_gris_cv, cmap='gray'); ax_img.set_title("Imagen en Escala de Grises"); ax_img.axis('off')
    hist = cv2.calcHist([imagen_gris_cv], [0], None, [256], [0, 256])
    ax_hist.plot(hist, color='black'); ax_hist.set_title("Histograma de Grises"); ax_hist.grid(True, alpha=0.3)
    ax_hist.set_xlabel("Intensidad"); ax_hist.set_ylabel("Frecuencia")
    fig.tight_layout()

def dibujar_histograma_binaria(fig, imagen_binaria_cv):
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    ax_img.imshow(imagen_binaria_cv, cmap='gray'); ax_img.set_title("Imagen Binarizada"); ax_img.axis('off')
    hist = cv2.calcHist([imagen_binaria_cv], [0], None, [256], [0, 256])
    ax_hist.bar([0, 255], [hist[0].item(), hist[255].item()], color='black', width=20)
    ax_hist.set_title("Histograma Binario"); ax_hist.set_xticks([0, 255])
    ax_hist.set_xlabel("Intensidad (0=Negro, 255=Blanco)"); ax_hist.set_ylabel("Cantidad de Píxeles")
    ax_hist.grid(True, alpha=0.3); fig.tight_layout()

# =============================================================================
# --- SECCIÓN 2: CLASE DE LA APLICACIÓN (INTERFAZ GRÁFICA) ---
# =============================================================================
class MatplotlibApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes Digitales")
        self.original_image = None; self.grayscale_image_cv = None; self.binary_image_cv = None
        self.current_rgb_channels = None; self.current_cmy_channels = None; self.current_hsv_channels = None
        self.current_yiq_channels = None; self.current_hsi_channels = None; self.current_state = "original"
        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        color_model_frame = ttk.LabelFrame(control_frame, text="Modelos de Color", padding=5)
        color_model_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(color_model_frame, text="Cargar Imagen", command=self.cargar_imagen).pack(fill=tk.X, pady=2)
        ttk.Button(color_model_frame, text="Separar Canales RGB", command=self.aplicar_separacion_rgb).pack(fill=tk.X, pady=2)
        ttk.Button(color_model_frame, text="Separar Canales CMY", command=self.aplicar_separacion_cmy).pack(fill=tk.X, pady=2)
        ttk.Button(color_model_frame, text="Separar Canales HSV", command=self.aplicar_separacion_hsv).pack(fill=tk.X, pady=2)
        ttk.Button(color_model_frame, text="Separar Canales YIQ", command=self.aplicar_separacion_yiq).pack(fill=tk.X, pady=2)
        ttk.Button(color_model_frame, text="Separar Canales HSI", command=self.aplicar_separacion_hsi).pack(fill=tk.X, pady=2)
        single_channel_frame = ttk.LabelFrame(control_frame, text="Procesos", padding=5)
        single_channel_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(single_channel_frame, text="Escala de Grises", command=self.aplicar_escala_de_grises).pack(fill=tk.X, pady=2)
        ttk.Button(single_channel_frame, text="Binarizar Imagen", command=self.aplicar_binarizacion).pack(fill=tk.X, pady=2)
        analysis_frame = ttk.LabelFrame(control_frame, text="Análisis", padding=5)
        analysis_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        ttk.Button(analysis_frame, text="Histograma", command=self.mostrar_histograma).pack(fill=tk.X, pady=2)
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Button(action_frame, text="Guardar Vista", command=self.guardar_figura).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Revertir", command=self.revertir_a_original).pack(fill=tk.X, pady=2)
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _reset_states(self):
        self.grayscale_image_cv = None; self.binary_image_cv = None
        self.current_rgb_channels = None; self.current_cmy_channels = None; self.current_hsv_channels = None
        self.current_yiq_channels = None; self.current_hsi_channels = None; self.current_state = "original"

    def cargar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(title="Selecciona un archivo de imagen", filetypes=[("Archivos de Imagen", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not ruta_imagen: return
        self.original_image = Image.open(ruta_imagen).convert('RGB')
        self.fig.clear(); ax = self.fig.add_subplot(111)
        ax.imshow(self.original_image); ax.set_title("Imagen Original"); ax.axis('off')
        self._reset_states(); self.canvas.draw()

    def aplicar_separacion_rgb(self):
        if self.original_image is None: messagebox.showerror("Error", "Primero debes cargar una imagen."); return
        self._reset_states(); r, g, b = separar_canales_rgb(self.original_image); self.current_rgb_channels = (r, g, b)
        dibujar_canales_rgb(self.fig, r, g, b); self.current_state = "rgb_channels"; self.canvas.draw()

    def aplicar_separacion_cmy(self):
        if self.original_image is None: messagebox.showerror("Error", "Primero debes cargar una imagen."); return
        self._reset_states(); c, m, y = separar_canales_cmy(self.original_image); self.current_cmy_channels = (c, m, y)
        dibujar_canales_cmy(self.fig, c, m, y); self.current_state = "cmy_channels"; self.canvas.draw()

    def aplicar_separacion_hsv(self):
        if self.original_image is None: messagebox.showerror("Error", "Primero debes cargar una imagen."); return
        self._reset_states(); h, s, v = separar_canales_hsv(self.original_image); self.current_hsv_channels = (h, s, v)
        dibujar_canales_hsv(self.fig, h, s, v); self.current_state = "hsv_channels"; self.canvas.draw()

    def aplicar_separacion_yiq(self):
        if self.original_image is None: messagebox.showerror("Error", "Primero debes cargar una imagen."); return
        self._reset_states(); y, i, q = separar_canales_yiq(self.original_image); self.current_yiq_channels = (y, i, q)
        dibujar_canales_yiq(self.fig, y, i, q); self.current_state = "yiq_channels"; self.canvas.draw()
        
    def aplicar_separacion_hsi(self):
        if self.original_image is None: messagebox.showerror("Error", "Primero debes cargar una imagen."); return
        self._reset_states(); h, s, i = separar_canales_hsi(self.original_image); self.current_hsi_channels = (h, s, i)
        dibujar_canales_hsi(self.fig, h, s, i); self.current_state = "hsi_channels"; self.canvas.draw()

    def aplicar_escala_de_grises(self):
        if self.original_image is None: messagebox.showerror("Error", "Primero debes cargar una imagen."); return
        self._reset_states(); self.fig.clear(); ax = self.fig.add_subplot(111)
        self.grayscale_image_cv = convertir_a_grises_en_ax(self.original_image, ax)
        self.current_state = "grayscale"; self.canvas.draw()

    def aplicar_binarizacion(self):
        if self.grayscale_image_cv is None:
            if messagebox.askyesno("Confirmación", "Se requiere una imagen en escala de grises. ¿Deseas convertirla ahora?"):
                self.aplicar_escala_de_grises(); self.root.update_idletasks() 
            else: return
        umbral_elegido = simpledialog.askinteger("Umbral de Binarización", "Ingresa el valor del umbral (0-255):", parent=self.root, minvalue=0, maxvalue=255, initialvalue=128)
        if umbral_elegido is None: return
        self.fig.clear(); ax = self.fig.add_subplot(111)
        imagen_a_binarizar = self.grayscale_image_cv if self.grayscale_image_cv is not None else cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
        self.binary_image_cv = binarizar_imagen_en_ax(imagen_a_binarizar, ax, umbral=umbral_elegido)
        self.current_state = "binary"; self.canvas.draw()
        
    def mostrar_histograma(self):
        if self.original_image is None: messagebox.showerror("Error", "Primero debes cargar una imagen."); return
        
        if self.current_state == "rgb_channels": dibujar_histograma_canales_rgb(self.fig, *self.current_rgb_channels)
        elif self.current_state == "cmy_channels": dibujar_histograma_canales_cmy(self.fig, *self.current_cmy_channels)
        elif self.current_state == "hsv_channels": dibujar_histograma_canales_hsv(self.fig, *self.current_hsv_channels)
        elif self.current_state == "yiq_channels": dibujar_histograma_canales_yiq(self.fig, *self.current_yiq_channels)
        elif self.current_state == "hsi_channels": dibujar_histograma_canales_hsi(self.fig, *self.current_hsi_channels)
        elif self.current_state == "grayscale":
            if self.grayscale_image_cv is not None: dibujar_histograma_escala_grises(self.fig, self.grayscale_image_cv)
        elif self.current_state == "binary":
            if self.binary_image_cv is not None: dibujar_histograma_binaria(self.fig, self.binary_image_cv)
        else:
            messagebox.showinfo("Aviso", "Mostrando histograma RGB de la imagen original.")
            self.fig.clear(); ax_img, ax_hist = self.fig.subplots(1, 2)
            ax_img.imshow(self.original_image); ax_img.set_title("Imagen Original"); ax_img.axis('off')
            imagen_np = np.array(self.original_image)
            hist_r = cv2.calcHist([imagen_np], [0], None, [256], [0, 256]); hist_g = cv2.calcHist([imagen_np], [1], None, [256], [0, 256]); hist_b = cv2.calcHist([imagen_np], [2], None, [256], [0, 256])
            ax_hist.plot(hist_r, color='red', alpha=0.7, label='R'); ax_hist.plot(hist_g, color='green', alpha=0.7, label='G'); ax_hist.plot(hist_b, color='blue', alpha=0.7, label='B')
            ax_hist.set_title("Histograma RGB"); ax_hist.set_xlabel("Intensidad"); ax_hist.set_ylabel("Frecuencia"); ax_hist.legend(); ax_hist.grid(True, alpha=0.3)
            self.fig.tight_layout()
        self.canvas.draw()
    
    def revertir_a_original(self):
        if self.original_image is None: messagebox.showerror("Error", "No hay imagen cargada."); return
        self.fig.clear(); ax = self.fig.add_subplot(111)
        ax.imshow(self.original_image); ax.set_title("Imagen Original"); ax.axis('off')
        self._reset_states(); self.canvas.draw(); messagebox.showinfo("Revertido", "Mostrando la imagen original.")

    def guardar_figura(self):
        if self.original_image is None: messagebox.showerror("Error", "No hay nada que guardar."); return
        ruta = filedialog.asksaveasfilename(title="Guardar vista actual como...", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("PDF", "*.pdf")])
        if ruta: self.fig.savefig(ruta); messagebox.showinfo("Éxito", "La vista actual ha sido guardada.")

# =============================================================================
# --- SECCIÓN 3: PUNTO DE ENTRADA PRINCIPAL ---
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = MatplotlibApp(root)
    root.mainloop()