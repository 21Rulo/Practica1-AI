# -----------------------------------------------------------------------------
# PROYECTO 1: PROCESADOR DE IMÁGENES DIGITALES - VERSIÓN CORREGIDA
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
from scipy.stats import skew
from scipy import ndimage
import random

# Librerías para integrar los gráficos de Matplotlib en Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap 

# --- FUNCIÓN LOG2 FALTANTE ---
def log2(x):
    """Calcula logaritmo base 2."""
    if x <= 0:
        return 0
    return math.log2(x)

# --- 1. Funciones de Procesamiento Básicas ---
def separar_canales(imagen_pil):
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    return imagen_pil.split()

def separar_canales_rgb(imagen_pil):
    """Separa canales RGB de una imagen."""
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    return imagen_pil.split()

def separar_canales_cmy(imagen_pil):
    """Separa canales CMY de una imagen."""
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    
    r, g, b = imagen_pil.split()
    r_arr, g_arr, b_arr = np.array(r), np.array(g), np.array(b)
    
    c = 255 - r_arr
    m = 255 - g_arr
    y = 255 - b_arr
    
    return Image.fromarray(c), Image.fromarray(m), Image.fromarray(y)

def separar_canales_hsv(imagen_pil):
    """Separa canales HSV de una imagen."""
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    return Image.fromarray(h), Image.fromarray(s), Image.fromarray(v)

def dibujar_canales_rgb(fig, r, g, b):
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3)
    ax1.imshow(r, cmap='Reds')
    ax1.set_title("Componente R")
    ax1.axis('off')
    ax2.imshow(g, cmap='Greens')
    ax2.set_title("Componente G")
    ax2.axis('off')
    ax3.imshow(b, cmap='Blues')
    ax3.set_title("Componente B")
    ax3.axis('off')
    fig.tight_layout()

def dibujar_canales_cmy(fig, c, m, y):
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3)
    ax1.imshow(c, cmap='cyan')
    ax1.set_title("Componente C")
    ax1.axis('off')
    ax2.imshow(m, cmap='magma')
    ax2.set_title("Componente M")
    ax2.axis('off')
    ax3.imshow(y, cmap='YlOrBr')
    ax3.set_title("Componente Y")
    ax3.axis('off')
    fig.tight_layout()

def dibujar_canales_hsv(fig, h, s, v):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(h, cmap='hsv')
    ax1.set_title("Componente H (Hue)")
    ax1.axis('off')
    ax2.imshow(s, cmap='plasma')
    ax2.set_title("Componente S (Saturation)")
    ax2.axis('off')
    ax3.imshow(v, cmap='gray')
    ax3.set_title("Componente V (Value)")
    ax3.axis('off')
    fig.tight_layout()

# --- Funciones para el Modelo YIQ ---
def separar_canales_yiq(imagen_pil):
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    rgb_array = np.array(imagen_pil) / 255.0
    yiq_array = np.zeros_like(rgb_array)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            yiq_array[i, j] = colorsys.rgb_to_yiq(*rgb_array[i, j])
    y = (yiq_array[:, :, 0] * 255).astype(np.uint8)
    i_norm = ((yiq_array[:, :, 1] - yiq_array[:, :, 1].min()) / (yiq_array[:, :, 1].max() - yiq_array[:, :, 1].min() + 1e-6) * 255).astype(np.uint8)
    q_norm = ((yiq_array[:, :, 2] - yiq_array[:, :, 2].min()) / (yiq_array[:, :, 2].max() - yiq_array[:, :, 2].min() + 1e-6) * 255).astype(np.uint8)
    return y, i_norm, q_norm

def dibujar_canales_yiq(fig, y, i, q):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(y, cmap='gray')
    ax1.set_title("Componente Y (Luminancia)")
    ax1.axis('off')
    ax2.imshow(i, cmap='RdBu')
    ax2.set_title("Componente I (Crominancia)")
    ax2.axis('off')
    ax3.imshow(q, cmap='PiYG')
    ax3.set_title("Componente Q (Crominancia)")
    ax3.axis('off')
    fig.tight_layout()

# --- Funciones para el Modelo HSI ---
def separar_canales_hsi(imagen_pil):
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.array(imagen_pil) / 255.0
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        i = (r + g + b) / 3.0
        min_rgb = np.minimum(np.minimum(r, g), b)
        s = 1 - (3 / (r + g + b + 1e-6)) * min_rgb
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g)**2 + (r - b) * (g - b))
        theta = np.arccos(np.clip(num / (den + 1e-6), -1, 1))
        h = np.copy(theta)
        h[b > g] = (2 * np.pi) - h[b > g]
        h = h / (2 * np.pi)
    h_out = (h * 255).astype(np.uint8)
    s_out = (s * 255).astype(np.uint8)
    i_out = (i * 255).astype(np.uint8)
    return h_out, s_out, i_out

def dibujar_canales_hsi(fig, h, s, i):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(h, cmap='hsv')
    ax1.set_title("Componente H (Hue)")
    ax1.axis('off')
    ax2.imshow(s, cmap='plasma')
    ax2.set_title("Componente S (Saturation)")
    ax2.axis('off')
    ax3.imshow(i, cmap='gray')
    ax3.set_title("Componente I (Intensity)")
    ax3.axis('off')
    fig.tight_layout()

# --- Funciones para Escala de Grises y Binarización ---
def convertir_a_grises_en_ax(imagen_pil, ax):
    ax.clear()
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    gris_cv = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    ax.imshow(gris_cv, cmap='gray')
    ax.set_title("Imagen en Escala de Grises")
    ax.axis('off')
    return gris_cv

def binarizar_imagen_en_ax(imagen_gris_cv, ax, umbral=128):
    ax.clear()
    _, binaria = cv2.threshold(imagen_gris_cv, umbral, 255, cv2.THRESH_BINARY)
    ax.imshow(binaria, cmap='gray')
    ax.set_title(f"Imagen binarizada (umbral = {umbral})")
    ax.axis('off')
    return binaria

def calcular_histograma_rgb(imagen_pil):
    imagen_np = np.array(imagen_pil)
    if len(imagen_np.shape) == 3:
        hist_r = cv2.calcHist([imagen_np], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([imagen_np], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([imagen_np], [2], None, [256], [0, 256])
        return hist_r, hist_g, hist_b
    else:
        hist = cv2.calcHist([imagen_np], [0], None, [256], [0, 256])
        return hist

def dibujar_histograma_imagen_original(fig, imagen_pil):
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    ax_img.imshow(imagen_pil)
    ax_img.set_title("Imagen Original")
    ax_img.axis('off')
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

# --- Funciones para el Dibujo de Histogramas ---
def dibujar_histograma_canales_rgb(fig, r, g, b):
    fig.clear()
    gs = fig.add_gridspec(2, 3, hspace=0.3)
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
    ax_hist_r = fig.add_subplot(gs[1, 0])
    ax_hist_g = fig.add_subplot(gs[1, 1])
    ax_hist_b = fig.add_subplot(gs[1, 2])
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
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    ax_img.imshow(imagen_gris_cv, cmap='gray')
    ax_img.set_title("Imagen en Escala de Grises")
    ax_img.axis('off')
    hist = cv2.calcHist([imagen_gris_cv], [0], None, [256], [0, 256])
    ax_hist.plot(hist, color='black')
    ax_hist.set_title("Histograma Escala de Grises")
    ax_hist.set_xlabel("Intensidad de Pixel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.grid(True, alpha=0.3)
    fig.tight_layout()

def dibujar_histograma_binaria(fig, imagen_binaria_cv):
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    ax_img.imshow(imagen_binaria_cv, cmap='gray')
    ax_img.set_title("Imagen Binarizada")
    ax_img.axis('off')
    hist = cv2.calcHist([imagen_binaria_cv], [0], None, [256], [0, 256])
    ax_hist.bar([0, 255], [hist[0].item(), hist[255].item()], color='black', width=50)
    ax_hist.set_title("Histograma Binario")
    ax_hist.set_xlabel("Intensidad de Pixel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.set_xlim(-25, 280)
    ax_hist.grid(True, alpha=0.3)
    fig.tight_layout()

def calcular_caracteristicas_estadisticas(imagen_pil):
    imagen_rgb = np.array(imagen_pil)
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
        imagen_rgb = np.array(imagen_pil)
    resultados = {}
    for i, canal in enumerate(['Red', 'Green', 'Blue']):
        datos = imagen_rgb[:, :, i].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
        if histograma.sum() == 0:
            continue
        prob = histograma / histograma.sum()
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
    datos = imagen_gris_cv.flatten()
    histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
    if histograma.sum() == 0:
        return None
    prob = histograma / histograma.sum()
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

# --- 2. FUNCIONES DE MORFOLOGÍA MATEMÁTICA ---

def crear_kernel(tamano, forma='rect'):
    """Crea un elemento estructurante (kernel)."""
    if forma == 'rect':
        return np.ones((tamano, tamano), np.uint8)
    elif forma == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano, tamano))
    elif forma == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (tamano, tamano))
    else:
        return np.ones((tamano, tamano), np.uint8)

# Operaciones básicas
def erosion_morfologica(imagen, kernel, iteraciones=1):
    return cv2.erode(imagen, kernel, iterations=iteraciones)

def dilatacion_morfologica(imagen, kernel, iteraciones=1):
    return cv2.dilate(imagen, kernel, iterations=iteraciones)

def apertura_tradicional(imagen, kernel, iteraciones=1):
    erosionada = cv2.erode(imagen, kernel, iterations=iteraciones)
    return cv2.dilate(erosionada, kernel, iterations=iteraciones)

def apertura_opencv(imagen, kernel, iteraciones=1):
    return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel, iterations=iteraciones)

def cierre_tradicional(imagen, kernel, iteraciones=1):
    dilatada = cv2.dilate(imagen, kernel, iterations=iteraciones)
    return cv2.erode(dilatada, kernel, iterations=iteraciones)

def cierre_opencv(imagen, kernel, iteraciones=1):
    return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)

# Operaciones avanzadas para Morfología Binaria
def obtener_frontera(imagen, kernel):
    """Frontera: Original - Erosión."""
    erosionada = cv2.erode(imagen, kernel, iterations=1)
    return cv2.subtract(imagen, erosionada)

def adelgazamiento(imagen, kernel, iteraciones=1):
    """Adelgazamiento (Thinning) morfológico."""
    resultado = imagen.copy()
    for _ in range(iteraciones):
        erosionada = cv2.erode(resultado, kernel, iterations=1)
        temp = cv2.dilate(erosionada, kernel, iterations=1)
        temp = cv2.subtract(resultado, temp)
        resultado = cv2.bitwise_or(erosionada, temp)
    return resultado

def hit_or_miss(imagen, kernel1, kernel2):
    """Transformada Hit-or-Miss."""
    hit = cv2.erode(imagen, kernel1, iterations=1)
    miss = cv2.erode(cv2.bitwise_not(imagen), kernel2, iterations=1)
    return cv2.bitwise_and(hit, miss)

def esqueleto_morfologico(imagen, kernel):
    """Calcula el esqueleto morfológico de una imagen binaria."""
    esqueleto = np.zeros(imagen.shape, dtype=np.uint8)
    temp = imagen.copy()
    
    while True:
        erosionada = cv2.erode(temp, kernel, iterations=1)
        temp_abierta = cv2.morphologyEx(erosionada, cv2.MORPH_OPEN, kernel)
        subset = cv2.subtract(erosionada, temp_abierta)
        esqueleto = cv2.bitwise_or(esqueleto, subset)
        temp = erosionada.copy()
        
        if cv2.countNonZero(temp) == 0:
            break
    
    return esqueleto

# Operaciones para Morfología en Laticces (Escala de Grises)
def gradiente_morfologico_simetrico(imagen, kernel):
    """Gradiente simétrico: Dilatación - Erosión."""
    dilatada = cv2.dilate(imagen, kernel, iterations=1)
    erosionada = cv2.erode(imagen, kernel, iterations=1)
    return cv2.subtract(dilatada, erosionada)

def gradiente_por_erosion(imagen, kernel):
    """Gradiente por erosión: Original - Erosión."""
    erosionada = cv2.erode(imagen, kernel, iterations=1)
    return cv2.subtract(imagen, erosionada)

def gradiente_por_dilatacion(imagen, kernel):
    """Gradiente por dilatación: Dilatación - Original."""
    dilatada = cv2.dilate(imagen, kernel, iterations=1)
    return cv2.subtract(dilatada, imagen)

def top_hat(imagen, kernel):
    """Top Hat: Original - Apertura (detecta puntos brillantes)."""
    return cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)

def bottom_hat(imagen, kernel):
    """Bottom Hat: Cierre - Original (detecta puntos oscuros)."""
    return cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)

def filtro_suavizado_apertura(imagen, kernel):
    """Filtro de suavizado mediante apertura."""
    return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)

def filtro_suavizado_cierre(imagen, kernel):
    """Filtro de suavizado mediante cierre."""
    return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)

def filtro_suavizado_apertura_cierre(imagen, kernel):
    """Suavizado: Apertura seguida de Cierre."""
    apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)

def filtro_suavizado_cierre_apertura(imagen, kernel):
    """Suavizado: Cierre seguida de Apertura."""
    cierre = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(cierre, cv2.MORPH_OPEN, kernel)


# --- NUEVAS FUNCIONES: OPERACIONES ARITMÉTICAS Y LÓGICAS ---

def sumar_escalar(imagen, escalar):
    """Suma un escalar a la imagen."""
    matriz_escalar = np.full(imagen.shape, int(escalar), dtype=np.uint8)
    return cv2.add(imagen, matriz_escalar)

def restar_escalar(imagen, escalar):
    """Resta un escalar de la imagen."""
    matriz_escalar = np.full(imagen.shape, int(escalar), dtype=np.uint8)
    return cv2.subtract(imagen, matriz_escalar)

def multiplicar_escalar(imagen, escalar):
    """Multiplica la imagen por un escalar."""
    return cv2.multiply(imagen, float(escalar))

def sumar_imagenes(img1, img2):
    """Suma dos imágenes (ajustando tamaños)."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.add(img1_resized, img2_resized)

def restar_imagenes(img1, img2):
    """Resta dos imágenes (ajustando tamaños)."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.subtract(img1_resized, img2_resized)

def operacion_and(img1, img2):
    """AND lógico entre dos imágenes."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.bitwise_and(img1_resized, img2_resized)

def operacion_or(img1, img2):
    """OR lógico entre dos imágenes."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.bitwise_or(img1_resized, img2_resized)

def operacion_xor(img1, img2):
    """XOR lógico entre dos imágenes."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.bitwise_xor(img1_resized, img2_resized)

def operacion_not(imagen):
    """NOT lógico (inversión)."""
    return cv2.bitwise_not(imagen)

def operacion_relacional(imagen, umbral, operador):
    """Aplica operación relacional (>, <, ==)."""
    if operador == '>':
        mask = imagen > umbral
    elif operador == '<':
        mask = imagen < umbral
    elif operador == '==':
        mask = imagen == umbral
    else:
        return imagen
    return np.uint8(mask) * 255

def agregar_ruido_sal_pimienta(imagen, porcentaje, modo='mixto'):
    """Agrega ruido sal y pimienta."""
    img_ruidosa = imagen.copy()
    alto, ancho = img_ruidosa.shape
    num_pixeles = int((porcentaje / 100) * alto * ancho)

    for _ in range(num_pixeles):
        y = random.randint(0, alto - 1)
        x = random.randint(0, ancho - 1)
        
        if modo == 'mixto':
            if random.random() < 0.5:
                img_ruidosa[y, x] = 255
            else:
                img_ruidosa[y, x] = 0
        elif modo == 'sal':
            img_ruidosa[y, x] = 255
        elif modo == 'pimienta':
            img_ruidosa[y, x] = 0
                
    return img_ruidosa

def agregar_ruido_gaussiano(imagen, media=0, sigma=25):
    """Agrega ruido gaussiano."""
    img_ruidosa = imagen.copy().astype(np.float64)
    ruido = np.random.normal(media, sigma, imagen.shape)
    img_ruidosa += ruido
    img_ruidosa = np.clip(img_ruidosa, 0, 255)
    return img_ruidosa.astype(np.uint8)

def etiquetar_componentes(imagen, vecindad=8):
    """Etiqueta componentes conexas."""
    if vecindad == 4:
        estructura = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    else:
        estructura = np.ones((3, 3), dtype=int)
    
    etiquetas, num_objetos = ndimage.label(imagen, structure=estructura)
    return etiquetas, num_objetos


# --- 3. Clase de la Aplicación GUI ---

class MatplotlibApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes - Morfología Avanzada")
        self.root.geometry("1200x800")

        self.original_image = None
        self.grayscale_image_cv = None
        self.binary_image_cv = None
        self.current_rgb_channels = None
        self.current_cmy_channels = None
        self.current_hsv_channels = None
        self.current_yiq_channels = None
        self.current_hsi_channels = None
        self.current_state = "original"
        self.morphology_result = None
        
        # Sistema de historial de operaciones
        self.imagen_trabajo_actual = None
        self.historial_operaciones = []
        
        # Frame principal dividido
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo con SCROLLBAR
        left_container = ttk.Frame(main_container, width=280)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_container.pack_propagate(False)
        
        bottom_bar_left = ttk.Frame(left_container)
        bottom_bar_left.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        ttk.Separator(bottom_bar_left).pack(fill=tk.X, pady=(0, 5))
        
        revert_frame = ttk.Frame(bottom_bar_left)
        revert_frame.pack(fill=tk.X, padx=5)
        
        ttk.Button(revert_frame, text="🔄 Revertir", 
                   command=self.revertir_ultima_operacion).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1, pady=2)
        ttk.Button(revert_frame, text="🔙 Original", 
                   command=self.revertir_a_original).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1, pady=2)
        
        ttk.Label(bottom_bar_left, text="GUARDAR", 
                  font=('Arial', 10, 'bold')).pack(pady=(8, 5))
        ttk.Button(bottom_bar_left, text="💾 Guardar Resultado", 
                   command=self.guardar_imagen_actual).pack(fill=tk.X, padx=5, pady=2)

        # Canvas para permitir scroll
        canvas_left = tk.Canvas(left_container, width=260, highlightthickness=0)
        scrollbar_left = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=canvas_left.yview)
        
        # Frame scrollable dentro del canvas
        left_panel = ttk.Frame(canvas_left)
        
        # Configurar scroll
        left_panel.bind(
            "<Configure>",
            lambda e: canvas_left.configure(scrollregion=canvas_left.bbox("all"))
        )
        
        canvas_left.create_window((0, 0), window=left_panel, anchor="nw")
        canvas_left.configure(yscrollcommand=scrollbar_left.set)
        
        # Empaquetar canvas y scrollbar
        canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_left.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Habilitar scroll con rueda del mouse
        def _on_mousewheel(event):
            canvas_left.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_left.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Panel derecho: Visualización
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- CONTROLES EN PANEL IZQUIERDO (con scroll) ---
        
        # Sección: Operaciones Básicas
        ttk.Label(left_panel, text="OPERACIONES BÁSICAS", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
        
        ttk.Button(left_panel, text="📁 Cargar Imagen", command=self.cargar_imagen).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🎨 Separar Canales RGB", command=self.aplicar_separacion_rgb).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="⚫ Escala de Grises", command=self.aplicar_escala_de_grises).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="◾ Binarizar", command=self.aplicar_binarizacion).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sección: Morfología
        ttk.Label(left_panel, text="MORFOLOGÍA", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
        
        ttk.Button(left_panel, text="🔷 Morfología Básica", command=self.abrir_menu_morfologia_basica).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🔶 Morfología Binaria", command=self.abrir_menu_morfologia_binaria).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🔸 Morfología Laticces", command=self.abrir_menu_morfologia_laticces).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sección: Operaciones Avanzadas
        ttk.Label(left_panel, text="OPERACIONES AVANZADAS", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
        
        ttk.Button(left_panel, text="➕ Operaciones Aritméticas", command=self.abrir_menu_aritmeticas).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🔀 Operaciones Lógicas", command=self.abrir_menu_logicas).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="⚖️ Operaciones Relacionales", command=self.abrir_menu_relacionales).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🎲 Agregar Ruido", command=self.abrir_menu_ruido).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🏷️ Etiquetado Componentes", command=self.abrir_menu_etiquetado).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sección: Análisis
        ttk.Label(left_panel, text="ANÁLISIS", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
        
        ttk.Button(left_panel, text="📊 Pseudocolor", command=self.abrir_opciones_pseudocolor).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="📈 Histograma", command=self.mostrar_histograma).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="📋 Características", command=self.mostrar_caracteristicas).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sección de Historial
        ttk.Label(left_panel, text="HISTORIAL", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
        
        # Frame con scrollbar para historial
        historial_frame = ttk.Frame(left_panel)
        historial_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.historial_text = tk.Text(historial_frame, height=8, width=30, wrap=tk.WORD, font=('Courier', 8))
        historial_scroll = ttk.Scrollbar(historial_frame, orient=tk.VERTICAL, command=self.historial_text.yview)
        self.historial_text.configure(yscrollcommand=historial_scroll.set)
        
        self.historial_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        historial_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.historial_text.config(state=tk.DISABLED)
        
        # --- CANVAS DE VISUALIZACIÓN ---
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _reset_states(self):
        """Resetea los estados de procesamiento."""
        self.grayscale_image_cv = None
        self.binary_image_cv = None
        self.current_rgb_channels = None
        self.current_cmy_channels = None
        self.current_hsv_channels = None
        self.current_yiq_channels = None
        self.current_hsi_channels = None
        self.morphology_result = None
        self.imagen_trabajo_actual = None
        self.historial_operaciones = []
        self.actualizar_texto_historial()

    def agregar_a_historial(self, operacion):
        """Agrega una operación al historial."""
        self.historial_operaciones.append(operacion)
        self.actualizar_texto_historial()
    
    def actualizar_texto_historial(self):
        """Actualiza el texto del historial en la interfaz."""
        self.historial_text.config(state=tk.NORMAL)
        self.historial_text.delete(1.0, tk.END)
        
        if not self.historial_operaciones:
            self.historial_text.insert(tk.END, "Sin operaciones aplicadas")
        else:
            for i, op in enumerate(self.historial_operaciones, 1):
                self.historial_text.insert(tk.END, f"{i}. {op}\n")
        
        self.historial_text.config(state=tk.DISABLED)
        self.historial_text.see(tk.END)
    
    def revertir_ultima_operacion(self):
        """Revierte la última operación morfológica aplicada."""
        if not self.historial_operaciones:
            messagebox.showinfo("Info", "No hay operaciones para revertir.")
            return
        
        # Remover última operación
        self.historial_operaciones.pop()
        
        if not self.historial_operaciones:
            # Si no quedan operaciones, volver a la imagen base
            if self.binary_image_cv is not None:
                self.imagen_trabajo_actual = self.binary_image_cv.copy()
            elif self.grayscale_image_cv is not None:
                self.imagen_trabajo_actual = self.grayscale_image_cv.copy()
            messagebox.showinfo("Revertido", "Se revirtió a la imagen base.")
        else:
            # Volver a aplicar todas las operaciones menos la última
            messagebox.showinfo("Info", "Para revertir múltiples pasos, usa 'Revertir a Original' y reaplica las operaciones deseadas.")
            return
        
        self.actualizar_texto_historial()
        self._mostrar_imagen_actual()

    def _preparar_lienzo_unico(self):
        self.fig.clear()
        return self.fig.add_subplot(111)

    def _mostrar_imagen_original(self):
        ax = self._preparar_lienzo_unico()
        ax.imshow(self.original_image)
        ax.set_title("Imagen Original")
        ax.axis('off')
        self._reset_states()
        self.current_state = "original"
        self.canvas.draw()
    
    def _mostrar_imagen_actual(self):
        """Muestra la imagen de trabajo actual."""
        if self.imagen_trabajo_actual is not None:
            ax = self._preparar_lienzo_unico()
            ax.imshow(self.imagen_trabajo_actual, cmap='gray')
            ax.set_title(f"Imagen Procesada ({len(self.historial_operaciones)} operaciones)")
            ax.axis('off')
            self.canvas.draw()

    def cargar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(
            title="Selecciona un archivo de imagen", 
            filetypes=[("Archivos de Imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not ruta_imagen:
            return
        
        try:
            self.original_image = Image.open(ruta_imagen).convert('RGB')
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(self.original_image)
            ax.set_title("Imagen Original")
            ax.axis('off')
            self._reset_states()
            self.current_state = "original"
            self.canvas.draw()
            messagebox.showinfo("Éxito", "Imagen cargada correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen: {e}")

    def aplicar_separacion_rgb(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        r, g, b = separar_canales_rgb(self.original_image)
        self.current_rgb_channels = (r, g, b)
        dibujar_canales_rgb(self.fig, r, g, b)
        self.current_state = "rgb_channels"
        self.canvas.draw()

    def aplicar_separacion_cmy(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        c, m, y = separar_canales_cmy(self.original_image)
        self.current_cmy_channels = (c, m, y)
        dibujar_canales_cmy(self.fig, c, m, y)
        self.current_state = "cmy_channels"
        self.canvas.draw()

    def aplicar_separacion_hsv(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        h, s, v = separar_canales_hsv(self.original_image)
        self.current_hsv_channels = (h, s, v)
        dibujar_canales_hsv(self.fig, h, s, v)
        self.current_state = "hsv_channels"
        self.canvas.draw()

    def aplicar_separacion_yiq(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        y, i, q = separar_canales_yiq(self.original_image)
        self.current_yiq_channels = (y, i, q)
        dibujar_canales_yiq(self.fig, y, i, q)
        self.current_state = "yiq_channels"
        self.canvas.draw()
        
    def aplicar_separacion_hsi(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        h, s, i = separar_canales_hsi(self.original_image)
        self.current_hsi_channels = (h, s, i)
        dibujar_canales_hsi(self.fig, h, s, i)
        self.current_state = "hsi_channels"
        self.canvas.draw()

    def aplicar_escala_de_grises(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.grayscale_image_cv = convertir_a_grises_en_ax(self.original_image, ax)
        self.imagen_trabajo_actual = self.grayscale_image_cv.copy()
        self.current_state = "grayscale"
        self.historial_operaciones = []
        self.actualizar_texto_historial()
        self.canvas.draw()

    def aplicar_binarizacion(self):
        if self.imagen_trabajo_actual is None and self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Primero convierte la imagen a escala de grises.")
            return
        self.abrir_dialogo_binarizacion()

    # --- MENÚS DE MORFOLOGÍA ---
    
    def abrir_menu_morfologia_basica(self):
        """Menú para operaciones morfológicas básicas."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.\nPresiona 'Escala de Grises' primero.")
            return
        
        # Establecer imagen de trabajo
        if self.imagen_trabajo_actual is None:
            if self.binary_image_cv is not None:
                self.imagen_trabajo_actual = self.binary_image_cv.copy()
            else:
                self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Morfología Básica")
        dialog.geometry("350x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Operaciones Básicas", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="Erosión", 
                   command=lambda: self.aplicar_operacion_secuencial('erosion', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Dilatación", 
                   command=lambda: self.aplicar_operacion_secuencial('dilatacion', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Apertura", 
                   command=lambda: self.aplicar_operacion_secuencial('apertura', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Cierre", 
                   command=lambda: self.aplicar_operacion_secuencial('cierre', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    # --- NUEVOS MENÚS: OPERACIONES AVANZADAS ---
    
    def abrir_menu_aritmeticas(self):
        """Menú para operaciones aritméticas."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Operaciones Aritméticas")
        dialog.geometry("350x280")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Operaciones Aritméticas", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Con escalar
        ttk.Label(main_frame, text="Con Escalar:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(5, 2))
        ttk.Button(main_frame, text="Sumar Escalar", 
                   command=lambda: self.aplicar_operacion_secuencial('suma_escalar', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Restar Escalar", 
                   command=lambda: self.aplicar_operacion_secuencial('resta_escalar', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Multiplicar Escalar", 
                   command=lambda: self.aplicar_operacion_secuencial('mult_escalar', dialog)).pack(fill=tk.X, pady=2)
        
        # Con otra imagen
        ttk.Label(main_frame, text="Con otra Imagen:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(8, 2))
        ttk.Button(main_frame, text="Sumar Imágenes", 
                   command=lambda: self.aplicar_operacion_con_imagen('suma_img', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Restar Imágenes", 
                   command=lambda: self.aplicar_operacion_con_imagen('resta_img', dialog)).pack(fill=tk.X, pady=2)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_logicas(self):
        """Menú para operaciones lógicas."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Operaciones Lógicas")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Operaciones Lógicas", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="AND (con otra imagen)", 
                   command=lambda: self.aplicar_operacion_con_imagen('and_img', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="OR (con otra imagen)", 
                   command=lambda: self.aplicar_operacion_con_imagen('or_img', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="XOR (con otra imagen)", 
                   command=lambda: self.aplicar_operacion_con_imagen('xor_img', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="NOT (inversión)", 
                   command=lambda: self.aplicar_operacion_secuencial('not_img', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_relacionales(self):
        """Menú para operaciones relacionales."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Operaciones Relacionales")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Operaciones Relacionales", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="Mayor que (>)", 
                   command=lambda: self.aplicar_operacion_secuencial('mayor', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Menor que (<)", 
                   command=lambda: self.aplicar_operacion_secuencial('menor', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Igual a (==)", 
                   command=lambda: self.aplicar_operacion_secuencial('igual', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_ruido(self):
        """Menú para agregar ruido."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Agregar Ruido")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Agregar Ruido", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="Ruido Sal y Pimienta", 
                   command=lambda: self.aplicar_operacion_secuencial('ruido_sp', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Ruido Sal", 
                   command=lambda: self.aplicar_operacion_secuencial('ruido_sal', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Ruido Pimienta", 
                   command=lambda: self.aplicar_operacion_secuencial('ruido_pimienta', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Ruido Gaussiano", 
                   command=lambda: self.aplicar_operacion_secuencial('ruido_gauss', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_etiquetado(self):
        """Menú para etiquetado de componentes."""
        if self.binary_image_cv is None and self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "Requiere imagen binaria.\nAplica primero: Escala de Grises → Binarizar")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.binary_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Etiquetado de Componentes")
        dialog.geometry("350x220") 
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Etiquetado de Componentes", font=('Arial', 12, 'bold')).pack(pady=10)
        
        invertir_var = tk.BooleanVar(value=False)
        chk_invertir = ttk.Checkbutton(main_frame, 
                                        text="Invertir imagen (Objetos negros a blancos)", 
                                        variable=invertir_var)
        chk_invertir.pack(fill=tk.X, pady=5)

        ttk.Button(main_frame, text="Vecindad 4", 
                   command=lambda: self.aplicar_etiquetado_directo(4, invertir_var.get(), dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Vecindad 8", 
                   command=lambda: self.aplicar_etiquetado_directo(8, invertir_var.get(), dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()
    
    def aplicar_etiquetado_directo(self, vecindad, invertir, dialog):
        """Aplica etiquetado y muestra resultado."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_para_label = self.imagen_trabajo_actual.copy()
            descripcion_inversion = ""
            if invertir:
                imagen_para_label = cv2.bitwise_not(imagen_para_label)
                descripcion_inversion = " (Invertido)"

            etiquetas, num_objetos = etiquetar_componentes(imagen_para_label, vecindad)
            
            # Crear imagen coloreada
            if num_objetos == 0:
                img_color = np.zeros((etiquetas.shape[0], etiquetas.shape[1], 3), dtype=np.uint8)
            else:
                etiquetas_norm = np.uint8(255 * etiquetas / num_objetos)
                img_color = cv2.applyColorMap(etiquetas_norm, cv2.COLORMAP_JET)
                img_color[etiquetas == 0] = [0, 0, 0]
            
            # Convertir a escala de grises para mantener compatibilidad
            img_resultado = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            # Actualizar imagen de trabajo
            self.imagen_trabajo_actual = img_resultado.copy()
            self.morphology_result = img_resultado
            self.current_state = "morphology"
            
            descripcion = f"Etiquetado V{vecindad}{descripcion_inversion} ({num_objetos} objetos)"
            self.agregar_a_historial(descripcion)
            
            # Mostrar con imagen coloreada
            self.fig.clear()
            axs = self.fig.subplots(1, 2)
            
            axs[0].imshow(imagen_para_label, cmap='gray')
            axs[0].set_title(f'Antes{descripcion_inversion}')
            axs[0].axis('off')
            
            # Etiquetado en color
            img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            axs[1].imshow(img_color_rgb)
            axs[1].set_title(f'Etiquetado V{vecindad}: {num_objetos} objetos')
            axs[1].axis('off')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            messagebox.showinfo("Resultado", f"Se encontraron {num_objetos} objetos con vecindad {vecindad}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en etiquetado: {e}")
    
    def aplicar_operacion_con_imagen(self, operacion, dialog):
        """Aplica operación que requiere cargar otra imagen."""
        dialog.destroy()
        
        # Cargar segunda imagen
        ruta = filedialog.askopenfilename(
            title="Seleccionar segunda imagen",
            filetypes=[("Archivos de Imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not ruta:
            return
        
        try:
            img_color = cv2.imread(ruta)
            img2 = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            # Ejecutar operación
            if operacion == 'suma_img':
                resultado = sumar_imagenes(imagen_entrada, img2)
                descripcion = "Suma con imagen"
            elif operacion == 'resta_img':
                resultado = restar_imagenes(imagen_entrada, img2)
                descripcion = "Resta con imagen"
            elif operacion == 'and_img':
                resultado = operacion_and(imagen_entrada, img2)
                descripcion = "AND con imagen"
            elif operacion == 'or_img':
                resultado = operacion_or(imagen_entrada, img2)
                descripcion = "OR con imagen"
            elif operacion == 'xor_img':
                resultado = operacion_xor(imagen_entrada, img2)
                descripcion = "XOR con imagen"
            else:
                messagebox.showerror("Error", "Operación no reconocida.")
                return
            
            # Actualizar
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def abrir_menu_morfologia_binaria(self):
        """Menú para operaciones morfológicas binarias avanzadas."""
        if self.binary_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen BINARIA.\nPresiona 'Binarizar' primero.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.binary_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Morfología Binaria Avanzada")
        dialog.geometry("350x280")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Morfología Binaria", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="Frontera", 
                   command=lambda: self.aplicar_operacion_secuencial('frontera', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Adelgazamiento (Thinning)", 
                   command=lambda: self.aplicar_operacion_secuencial('adelgazamiento', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Hit-or-Miss", 
                   command=lambda: self.aplicar_operacion_secuencial('hitmiss', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Esqueleto Morfológico", 
                   command=lambda: self.aplicar_operacion_secuencial('esqueleto', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_morfologia_laticces(self):
        """Menú para operaciones morfológicas en laticces (escala de grises)."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.\nPresiona 'Escala de Grises' primero.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Morfología en Laticces")
        dialog.geometry("400x420")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Morfología en Laticces", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Gradientes
        ttk.Label(main_frame, text="Gradientes:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(5, 2))
        ttk.Button(main_frame, text="Gradiente Simétrico", 
                   command=lambda: self.aplicar_operacion_secuencial('gradiente_simetrico', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Gradiente por Erosión", 
                   command=lambda: self.aplicar_operacion_secuencial('gradiente_erosion', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Gradiente por Dilatación", 
                   command=lambda: self.aplicar_operacion_secuencial('gradiente_dilatacion', dialog)).pack(fill=tk.X, pady=2)
        
        # Transformadas
        ttk.Label(main_frame, text="Transformadas:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(8, 2))
        ttk.Button(main_frame, text="Top Hat (Brillantes)", 
                   command=lambda: self.aplicar_operacion_secuencial('tophat', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Bottom Hat (Oscuros)", 
                   command=lambda: self.aplicar_operacion_secuencial('bottomhat', dialog)).pack(fill=tk.X, pady=2)
        
        # Filtros
        ttk.Label(main_frame, text="Filtros de Suavizado:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(8, 2))
        ttk.Button(main_frame, text="Suavizado por Apertura", 
                   command=lambda: self.aplicar_operacion_secuencial('suavizado_apertura', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Suavizado por Cierre", 
                   command=lambda: self.aplicar_operacion_secuencial('suavizado_cierre', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Suavizado Apertura+Cierre", 
                   command=lambda: self.aplicar_operacion_secuencial('suavizado_ac', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Suavizado Cierre+Apertura", 
                   command=lambda: self.aplicar_operacion_secuencial('suavizado_ca', dialog)).pack(fill=tk.X, pady=2)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def aplicar_operacion_secuencial(self, operacion, dialog):
        """Aplica operación morfológica de forma secuencial sobre la imagen de trabajo."""
        dialog.destroy()
        self.abrir_dialogo_configuracion_morfologia(operacion)

    def abrir_dialogo_configuracion_morfologia(self, operacion):
        """Diálogo para configurar parámetros del kernel y operaciones."""
        
        # Operaciones que necesitan parámetros especiales
        if operacion in ['suma_escalar', 'resta_escalar', 'mult_escalar']:
            self.dialogo_operacion_escalar(operacion)
            return
        elif operacion in ['mayor', 'menor', 'igual']:
            self.dialogo_operacion_relacional(operacion)
            return
        elif operacion in ['ruido_sp', 'ruido_sal', 'ruido_pimienta']:
            self.dialogo_ruido_sal_pimienta(operacion)
            return
        elif operacion == 'ruido_gauss':
            self.dialogo_ruido_gaussiano()
            return
        elif operacion == 'not_img':
            # NOT no necesita configuración
            self.ejecutar_operacion_simple(operacion)
            return
        
        # Diálogo estándar para morfología
        config_dialog = tk.Toplevel(self.root)
        config_dialog.title(f"Configuración - {operacion.capitalize()}")
        config_dialog.geometry("420x350")
        config_dialog.transient(self.root)
        config_dialog.grab_set()
        config_dialog.resizable(False, False)
        
        main_frame = ttk.Frame(config_dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text=f"Configurar {operacion.capitalize()}", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        # Tamaño del kernel
        ttk.Label(main_frame, text="Tamaño del Kernel:").pack(anchor='w', pady=(5, 0))
        tamano_var = tk.IntVar(value=5)
        tamano_frame = ttk.Frame(main_frame)
        tamano_frame.pack(fill=tk.X, pady=5)
        for tam in [3, 5, 7, 9, 11]:
            ttk.Radiobutton(tamano_frame, text=f"{tam}x{tam}", 
                            variable=tamano_var, value=tam).pack(side=tk.LEFT, padx=5)

        # Forma del kernel
        ttk.Label(main_frame, text="Forma del Kernel:").pack(anchor='w', pady=(10, 0))
        forma_var = tk.StringVar(value='rect')
        forma_frame = ttk.Frame(main_frame)
        forma_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(forma_frame, text="Rectangular", 
                        variable=forma_var, value='rect').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(forma_frame, text="Elíptico", 
                        variable=forma_var, value='ellipse').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(forma_frame, text="Cruz", 
                        variable=forma_var, value='cross').pack(side=tk.LEFT, padx=5)

        # Iteraciones
        ttk.Label(main_frame, text="Iteraciones:").pack(anchor='w', pady=(10, 0))
        iteraciones_var = tk.IntVar(value=1)
        iteraciones_spinbox = ttk.Spinbox(main_frame, from_=1, to=10, 
                                          textvariable=iteraciones_var, width=10)
        iteraciones_spinbox.pack(anchor='w', pady=5)

        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_morfologia_secuencial(operacion, 
                                                           tamano_var.get(), 
                                                           forma_var.get(), 
                                                           iteraciones_var.get(), 
                                                           config_dialog)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Cancelar", 
                   command=config_dialog.destroy).pack(side=tk.LEFT, padx=5)

    # --- DIÁLOGOS ESPECIALES PARA NUEVAS OPERACIONES ---
    
    def dialogo_operacion_escalar(self, operacion):
        """Diálogo para operaciones con escalar."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Configuración - {operacion}")
        dialog.geometry("320x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text=f"{operacion.replace('_', ' ').title()}", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Valor escalar:").pack(anchor='w', pady=(5, 0))
        escalar_var = tk.DoubleVar(value=50)
        escalar_entry = ttk.Entry(main_frame, textvariable=escalar_var, width=15)
        escalar_entry.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_operacion_escalar(operacion, escalar_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def dialogo_operacion_relacional(self, operacion):
        """Diálogo para operaciones relacionales."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Configuración - {operacion}")
        dialog.geometry("320x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ops = {'mayor': '>', 'menor': '<', 'igual': '=='}
        ttk.Label(main_frame, text=f"Operación: {ops[operacion]}", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Umbral (0-255):").pack(anchor='w', pady=(5, 0))
        umbral_var = tk.IntVar(value=128)
        umbral_spinbox = ttk.Spinbox(main_frame, from_=0, to=255, 
                                      textvariable=umbral_var, width=15)
        umbral_spinbox.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_operacion_relacional(operacion, umbral_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def dialogo_ruido_sal_pimienta(self, operacion):
        """Diálogo para ruido sal y pimienta."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración - Ruido")
        dialog.geometry("320x240")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text=f"Ruido {operacion.split('_')[1].title()}", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Porcentaje (0-100):").pack(anchor='w', pady=(5, 0))
        porcentaje_var = tk.IntVar(value=10)
        porcentaje_scale = tk.Scale(main_frame, from_=0, to=100, orient="horizontal", 
                                     variable=porcentaje_var, length=200)
        porcentaje_scale.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_ruido_sp(operacion, porcentaje_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def dialogo_ruido_gaussiano(self):
        """Diálogo para ruido gaussiano."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración - Ruido Gaussiano")
        dialog.geometry("320x240")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Ruido Gaussiano", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Media:").pack(anchor='w', pady=(5, 0))
        media_var = tk.DoubleVar(value=0)
        media_entry = ttk.Entry(main_frame, textvariable=media_var, width=15)
        media_entry.pack(anchor='w', pady=5)

        ttk.Label(main_frame, text="Desviación estándar (sigma):").pack(anchor='w', pady=(10, 0))
        sigma_var = tk.DoubleVar(value=25)
        sigma_entry = ttk.Entry(main_frame, textvariable=sigma_var, width=15)
        sigma_entry.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_ruido_gaussiano(media_var.get(), sigma_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def abrir_dialogo_binarizacion(self):
        """Diálogo para seleccionar umbral de binarización."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración - Binarizar")
        dialog.geometry("320x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Ajustar Umbral", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Umbral (0-255):").pack(anchor='w', pady=(5, 0))
        
        umbral_var = tk.IntVar(value=128)
        
        umbral_spinbox = ttk.Spinbox(main_frame, from_=0, to=255, 
                                      textvariable=umbral_var, width=15)
        umbral_spinbox.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_binarizacion(umbral_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def ejecutar_binarizacion(self, umbral, dialog):
        """Aplica la binarización con el umbral dado."""
        dialog.destroy()
        
        imagen_a_binarizar = None
        if self.imagen_trabajo_actual is not None:
            imagen_a_binarizar = self.imagen_trabajo_actual
        elif self.grayscale_image_cv is not None:
            imagen_a_binarizar = self.grayscale_image_cv
        else:
            messagebox.showerror("Error", "No hay imagen en escala de grises para binarizar.")
            return

        if imagen_a_binarizar is None:
            messagebox.showerror("Error", "Se perdió la imagen en escala de grises.")
            return
        
        try:
            ax = self._preparar_lienzo_unico()
            
            self.binary_image_cv = binarizar_imagen_en_ax(imagen_a_binarizar, ax, umbral)
            
            self.imagen_trabajo_actual = self.binary_image_cv.copy()
            self.current_state = "binary"
            
            self.agregar_a_historial(f"Binarizar (Umbral={umbral})")
            
            self.actualizar_texto_historial()
            self.canvas.draw()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al binarizar: {e}")

    # --- EJECUTORES DE OPERACIONES ESPECIALES ---
    
    def ejecutar_operacion_simple(self, operacion):
        """Ejecuta operación que no necesita parámetros."""
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            if operacion == 'not_img':
                resultado = operacion_not(imagen_entrada)
                descripcion = "NOT (inversión)"
            else:
                return
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_operacion_escalar(self, operacion, escalar, dialog):
        """Ejecuta operación aritmética con escalar."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            if operacion == 'suma_escalar':
                resultado = sumar_escalar(imagen_entrada, escalar)
                descripcion = f"Suma escalar {escalar}"
            elif operacion == 'resta_escalar':
                resultado = restar_escalar(imagen_entrada, escalar)
                descripcion = f"Resta escalar {escalar}"
            elif operacion == 'mult_escalar':
                resultado = multiplicar_escalar(imagen_entrada, escalar)
                descripcion = f"Multiplicar {escalar}"
            else:
                return
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_operacion_relacional(self, operacion, umbral, dialog):
        """Ejecuta operación relacional."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            ops = {'mayor': '>', 'menor': '<', 'igual': '=='}
            operador = ops[operacion]
            
            resultado = operacion_relacional(imagen_entrada, umbral, operador)
            descripcion = f"Relacional {operador} {umbral}"
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_ruido_sp(self, operacion, porcentaje, dialog):
        """Ejecuta ruido sal y pimienta."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            modos = {'ruido_sp': 'mixto', 'ruido_sal': 'sal', 'ruido_pimienta': 'pimienta'}
            modo = modos[operacion]
            
            resultado = agregar_ruido_sal_pimienta(imagen_entrada, porcentaje, modo)
            descripcion = f"Ruido {modo} {porcentaje}%"
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_ruido_gaussiano(self, media, sigma, dialog):
        """Ejecuta ruido gaussiano."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            resultado = agregar_ruido_gaussiano(imagen_entrada, media, sigma)
            descripcion = f"Ruido Gaussiano σ={sigma}"
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_morfologia_secuencial(self, operacion, tamano, forma, iteraciones, dialog):
        """Ejecuta operación morfológica sobre la imagen de trabajo actual."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo disponible.")
            return

        try:
            kernel = crear_kernel(tamano, forma)
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            # Ejecutar operación
            if operacion == 'erosion':
                resultado = erosion_morfologica(imagen_entrada, kernel, iteraciones)
                descripcion = f"Erosión {tamano}x{tamano} {forma} (iter:{iteraciones})"
                
            elif operacion == 'dilatacion':
                resultado = dilatacion_morfologica(imagen_entrada, kernel, iteraciones)
                descripcion = f"Dilatación {tamano}x{tamano} {forma} (iter:{iteraciones})"
                
            elif operacion == 'apertura':
                resultado = apertura_opencv(imagen_entrada, kernel, iteraciones)
                descripcion = f"Apertura {tamano}x{tamano} {forma} (iter:{iteraciones})"
                
            elif operacion == 'cierre':
                resultado = cierre_opencv(imagen_entrada, kernel, iteraciones)
                descripcion = f"Cierre {tamano}x{tamano} {forma} (iter:{iteraciones})"
            
            # Morfología Binaria
            elif operacion == 'frontera':
                resultado = obtener_frontera(imagen_entrada, kernel)
                descripcion = f"Frontera {tamano}x{tamano} {forma}"
                
            elif operacion == 'adelgazamiento':
                resultado = adelgazamiento(imagen_entrada, kernel, iteraciones)
                descripcion = f"Adelgazamiento {tamano}x{tamano} {forma} (iter:{iteraciones})"
                
            elif operacion == 'hitmiss':
                kernel2 = crear_kernel(tamano, forma)
                resultado = hit_or_miss(imagen_entrada, kernel, kernel2)
                descripcion = f"Hit-or-Miss {tamano}x{tamano} {forma}"
                
            elif operacion == 'esqueleto':
                resultado = esqueleto_morfologico(imagen_entrada, kernel)
                descripcion = f"Esqueleto {tamano}x{tamano} {forma}"
            
            # Morfología en Laticces
            elif operacion == 'gradiente_simetrico':
                resultado = gradiente_morfologico_simetrico(imagen_entrada, kernel)
                descripcion = f"Grad. Simétrico {tamano}x{tamano} {forma}"
                
            elif operacion == 'gradiente_erosion':
                resultado = gradiente_por_erosion(imagen_entrada, kernel)
                descripcion = f"Grad. Erosión {tamano}x{tamano} {forma}"
                
            elif operacion == 'gradiente_dilatacion':
                resultado = gradiente_por_dilatacion(imagen_entrada, kernel)
                descripcion = f"Grad. Dilatación {tamano}x{tamano} {forma}"
                
            elif operacion == 'tophat':
                resultado = top_hat(imagen_entrada, kernel)
                descripcion = f"Top Hat {tamano}x{tamano} {forma}"
                
            elif operacion == 'bottomhat':
                resultado = bottom_hat(imagen_entrada, kernel)
                descripcion = f"Bottom Hat {tamano}x{tamano} {forma}"
                
            elif operacion == 'suavizado_apertura':
                resultado = filtro_suavizado_apertura(imagen_entrada, kernel)
                descripcion = f"Suavizado Apertura {tamano}x{tamano} {forma}"
                
            elif operacion == 'suavizado_cierre':
                resultado = filtro_suavizado_cierre(imagen_entrada, kernel)
                descripcion = f"Suavizado Cierre {tamano}x{tamano} {forma}"
                
            elif operacion == 'suavizado_ac':
                resultado = filtro_suavizado_apertura_cierre(imagen_entrada, kernel)
                descripcion = f"Suavizado A+C {tamano}x{tamano} {forma}"
                
            elif operacion == 'suavizado_ca':
                resultado = filtro_suavizado_cierre_apertura(imagen_entrada, kernel)
                descripcion = f"Suavizado C+A {tamano}x{tamano} {forma}"
            
            else:
                messagebox.showerror("Error", f"Operación '{operacion}' no reconocida.")
                return
            
            # Actualizar imagen de trabajo
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            
            # Agregar al historial
            self.agregar_a_historial(descripcion)
            
            # Mostrar resultado
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error en Morfología", f"Ocurrió un error: {e}")

    def _mostrar_comparacion_antes_despues(self, antes, despues, titulo):
        """Muestra comparación lado a lado: antes y después."""
        self.fig.clear()
        axs = self.fig.subplots(1, 2)
        
        axs[0].imshow(antes, cmap='gray')
        axs[0].set_title('Antes')
        axs[0].axis('off')
        
        axs[1].imshow(despues, cmap='gray')
        axs[1].set_title(f'Después: {titulo}')
        axs[1].axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()

    # --- FUNCIONES DE PSEUDOCOLOR ---
    def abrir_opciones_pseudocolor(self):
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Opciones de Pseudocolor")
        dialog.geometry("350x220") 
        dialog.transient(self.root) 
        dialog.grab_set() 
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Seleccione mapas de color:", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Button(main_frame, text="JET, HOT, OCEAN", 
                   command=lambda: self.aplicar_pseudocolor_cv2(['JET', 'HOT', 'OCEAN'], dialog)).pack(fill=tk.X, pady=5)
        ttk.Button(main_frame, text="BONE, PINK, AUTUMN", 
                   command=lambda: self.aplicar_pseudocolor_cv2(['BONE', 'PINK', 'AUTUMN'], dialog)).pack(fill=tk.X, pady=5)
        ttk.Button(main_frame, text="Personalizado", 
                   command=lambda: self.aplicar_pseudocolor_pastel(dialog)).pack(fill=tk.X, pady=5)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack(side=tk.RIGHT)

    def aplicar_pseudocolor_cv2(self, map_names, dialog):
        if self.grayscale_image_cv is None: 
            messagebox.showerror("Error", "Se perdió la imagen.")
            dialog.destroy()
            return

        CV2_COLORMAPS = {
            'JET': cv2.COLORMAP_JET, 'HOT': cv2.COLORMAP_HOT, 'OCEAN': cv2.COLORMAP_OCEAN,
            'BONE': cv2.COLORMAP_BONE, 'PINK': cv2.COLORMAP_PINK, 'AUTUMN': cv2.COLORMAP_AUTUMN,
        }
        
        try:
            imagen_gris = self.grayscale_image_cv
            
            img_map1 = cv2.applyColorMap(imagen_gris, CV2_COLORMAPS[map_names[0]])
            img_map2 = cv2.applyColorMap(imagen_gris, CV2_COLORMAPS[map_names[1]])
            img_map3 = cv2.applyColorMap(imagen_gris, CV2_COLORMAPS[map_names[2]])
            
            img_map1_rgb = cv2.cvtColor(img_map1, cv2.COLOR_BGR2RGB)
            img_map2_rgb = cv2.cvtColor(img_map2, cv2.COLOR_BGR2RGB)
            img_map3_rgb = cv2.cvtColor(img_map3, cv2.COLOR_BGR2RGB)
            
            self.fig.clear()
            axs = self.fig.subplots(2, 2) 
            
            axs[0, 0].imshow(imagen_gris, cmap='gray')
            axs[0, 0].set_title('Escala de Grises')
            axs[0, 1].imshow(img_map1_rgb)
            axs[0, 1].set_title(f'{map_names[0]}')
            axs[1, 0].imshow(img_map2_rgb)
            axs[1, 0].set_title(f'{map_names[1]}')
            axs[1, 1].imshow(img_map3_rgb)
            axs[1, 1].set_title(f'{map_names[2]}')
            
            for ax in axs.flat:
                ax.axis('off')
            
            self.fig.tight_layout()
            self.current_state = "pseudocolor"
            self.canvas.draw()
            dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")
            dialog.destroy()

    def aplicar_pseudocolor_pastel(self, dialog):
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Se perdió la imagen.")
            dialog.destroy()
            return

        try:
            imagen_gris = self.grayscale_image_cv
            colores_pastel = [(1.0, 0.8, 0.9), (0.8, 1.0, 0.8), (0.8, 0.9, 1.0), (1.0, 1.0, 0.8), (0.9, 0.8, 1.0)]
            mapa_pastel = LinearSegmentedColormap.from_list("PastelMap", colores_pastel, N=256)
                                                            
            self.fig.clear()
            axs = self.fig.subplots(1, 2)
            
            axs[0].imshow(imagen_gris, cmap='gray')
            axs[0].set_title('Escala de grises')
            axs[0].axis('off')
            
            axs[1].imshow(imagen_gris, cmap=mapa_pastel)
            axs[1].set_title('Pseudocolor Pastel')
            axs[1].axis('off')
            
            self.fig.tight_layout()
            self.current_state = "pseudocolor"
            self.canvas.draw()
            dialog.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")
            dialog.destroy()

    # --- FUNCIÓN DE HISTOGRAMA ---
    def mostrar_histograma(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return

        if self.current_state == "original":
            dibujar_histograma_imagen_original(self.fig, self.original_image)
        elif self.current_state == "rgb_channels":
            if self.current_rgb_channels is None:
                messagebox.showerror("Error", "No hay canales RGB.")
                return
            r, g, b = self.current_rgb_channels
            dibujar_histograma_canales_rgb(self.fig, r, g, b)
        elif self.current_state in ["grayscale", "pseudocolor", "morphology"]:
            # Usar imagen de trabajo si existe
            img_mostrar = self.imagen_trabajo_actual if self.imagen_trabajo_actual is not None else self.grayscale_image_cv
            if img_mostrar is None:
                messagebox.showerror("Error", "No hay imagen disponible.")
                return
            dibujar_histograma_escala_grises(self.fig, img_mostrar)
        elif self.current_state == "binary":
            if self.binary_image_cv is None:
                messagebox.showerror("Error", "No hay imagen binaria.")
                return
            dibujar_histograma_binaria(self.fig, self.binary_image_cv)
        
        self.canvas.draw()
    
    # --- CARACTERÍSTICAS ESTADÍSTICAS ---
    def mostrar_caracteristicas(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        
        try:
            caracteristicas = None
            tipo_imagen = ""
            
            if self.current_state == "original":
                caracteristicas = calcular_caracteristicas_estadisticas(self.original_image)
                tipo_imagen = "Imagen Original (RGB)"
            elif self.current_state in ["grayscale", "pseudocolor", "morphology"]:
                img_analizar = self.imagen_trabajo_actual if self.imagen_trabajo_actual is not None else self.grayscale_image_cv
                if img_analizar is not None:
                    caracteristicas_grises = calcular_caracteristicas_escala_grises(img_analizar)
                    if caracteristicas_grises:
                        caracteristicas = {"Escala de Grises": caracteristicas_grises}
                    tipo_imagen = "Imagen Procesada" if self.imagen_trabajo_actual is not None else "Imagen en Escala de Grises"
            elif self.current_state == "binary" and self.binary_image_cv is not None:
                tipo_imagen = "Imagen Binarizada"; stats = calcular_caracteristicas_escala_grises(self.binary_image_cv)
                if stats: caracteristicas = {"Binaria": stats}
            elif self.current_state == "rgb_channels":
                caracteristicas = calcular_caracteristicas_estadisticas(self.original_image)
                tipo_imagen = "Canales RGB"
            
            if not caracteristicas:
                messagebox.showerror("Error", "No se pudieron calcular características.")
                return
            
            ventana_stats = tk.Toplevel(self.root)
            ventana_stats.title("Características Estadísticas")
            ventana_stats.geometry("650x450")
            
            main_frame = ttk.Frame(ventana_stats)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            ttk.Label(main_frame, text="Características Estadísticas", 
                      font=('Arial', 14, 'bold')).pack(pady=(0, 5))
            ttk.Label(main_frame, text=f"Tipo: {tipo_imagen}", 
                      font=('Arial', 11, 'italic')).pack(pady=(0, 10))
            
            text_frame = ttk.Frame(main_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            texto_scroll = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=texto_scroll.yview)
            texto_scroll.configure(yscrollcommand=scrollbar.set)
            
            resultado_texto = ""
            for canal, stats in caracteristicas.items():
                resultado_texto += f"\n{'='*35}\n"
                resultado_texto += f"{canal.upper()}\n"
                resultado_texto += f"{'='*35}\n"
                resultado_texto += f"Energía:    {stats['Energía']:.6f}\n"
                resultado_texto += f"Entropía:   {stats['Entropía']:.6f}\n"
                resultado_texto += f"Asimetría:  {stats['Asimetría']:.6f}\n"
                resultado_texto += f"Media:      {stats['Media']:.2f}\n"
                resultado_texto += f"Varianza:   {stats['Varianza']:.2f}\n"
            
            texto_scroll.insert(tk.END, resultado_texto)
            texto_scroll.config(state=tk.DISABLED)
            
            texto_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=10)
            
            ttk.Button(button_frame, text="Exportar", 
                       command=lambda: self.exportar_caracteristicas(caracteristicas, tipo_imagen)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cerrar", 
                       command=ventana_stats.destroy).pack(side=tk.LEFT, padx=5)
                       
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
    
    def exportar_caracteristicas(self, caracteristicas, tipo_imagen):
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Exportar características..."
        )
        
        if ruta_archivo:
            try:
                with open(ruta_archivo, 'w', encoding='utf-8') as f:
                    f.write("CARACTERÍSTICAS ESTADÍSTICAS\n")
                    f.write("=" * 45 + "\n")
                    f.write(f"Tipo: {tipo_imagen}\n\n")
                    
                    for canal, stats in caracteristicas.items():
                        f.write(f"\n{canal.upper()}:\n")
                        f.write(f"  Energía:    {stats['Energía']:.6f}\n")
                        f.write(f"  Entropía:   {stats['Entropía']:.6f}\n")
                        f.write(f"  Asimetría:  {stats['Asimetría']:.6f}\n")
                        f.write(f"  Media:      {stats['Media']:.2f}\n")
                        f.write(f"  Varianza:   {stats['Varianza']:.2f}\n")
                
                messagebox.showinfo("Éxito", f"Exportado: {os.path.basename(ruta_archivo)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")
    
    # --- FUNCIONES DE GUARDADO ---
    
    def revertir_a_original(self):
        if self.original_image is None:
            messagebox.showerror("Error", "No hay imagen original.")
            return
        self._mostrar_imagen_original()
        messagebox.showinfo("Revertido", "Revertido a imagen original.")

    def guardar_imagen_actual(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        try:
            if self.current_state == "morphology" and self.morphology_result is not None:
                self._guardar_resultado_morfologia()
            elif self.current_state == "original":
                self._guardar_original()
            elif self.current_state == "rgb_channels":
                self._guardar_canales_individuales()
            elif self.current_state == "grayscale":
                self._guardar_escala_grises()
            elif self.current_state == "binary":
                self._guardar_imagen_binaria()
            elif self.current_state == "pseudocolor":
                self._guardar_figura_actual("Guardar vista Pseudocolor...")
                
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"Error: {e}")

    def _guardar_resultado_morfologia(self):
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar resultado morfológico..."
        )
        if ruta_archivo:
            imagen_pil = Image.fromarray(self.morphology_result)
            imagen_pil.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")

    def _guardar_figura_actual(self, title="Guardar vista actual..."):
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title=title
        )
        if ruta_archivo:
            try:
                self.fig.savefig(ruta_archivo, dpi=150)
                messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {e}")

    def _guardar_original(self):
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar imagen original..."
        )
        if ruta_archivo:
            self.original_image.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")

    def _guardar_canales_individuales(self):
        if self.current_rgb_channels is None:
            messagebox.showerror("Error", "No hay canales RGB.")
            return
            
        ruta_base = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Guardar canales RGB..."
        )
        if not ruta_base:
            return
            
        directorio, nombre_completo = os.path.split(ruta_base)
        nombre_base, extension = os.path.splitext(nombre_completo)

        r, g, b = self.current_rgb_channels
        
        imagen_roja = Image.merge('RGB', (r, Image.new('L', r.size, 0), Image.new('L', r.size, 0)))
        imagen_verde = Image.merge('RGB', (Image.new('L', g.size, 0), g, Image.new('L', g.size, 0)))
        imagen_azul = Image.merge('RGB', (Image.new('L', b.size, 0), Image.new('L', b.size, 0), b))
        
        imagen_roja.save(os.path.join(directorio, f"{nombre_base}_R.png"))
        imagen_verde.save(os.path.join(directorio, f"{nombre_base}_G.png"))
        imagen_azul.save(os.path.join(directorio, f"{nombre_base}_B.png"))
        
        messagebox.showinfo("Éxito", f"Canales guardados en:\n{directorio}")

    def _guardar_escala_grises(self):
        # Si hay imagen de trabajo procesada, guardar esa
        img_guardar = self.imagen_trabajo_actual if self.imagen_trabajo_actual is not None else self.grayscale_image_cv
        
        if img_guardar is None:
            messagebox.showerror("Error", "No hay imagen en escala de grises.")
            return
            
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar imagen en escala de grises..."
        )
        if ruta_archivo:
            imagen_pil = Image.fromarray(img_guardar)
            imagen_pil.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")

    def _guardar_imagen_binaria(self):
        if self.binary_image_cv is None:
            messagebox.showerror("Error", "No hay imagen binaria.")
            return
            
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Guardar imagen binarizada..."
        )
        if ruta_archivo:
            imagen_pil = Image.fromarray(self.binary_image_cv)
            imagen_pil.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")


# --- 4. Bloque Principal de Ejecución ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MatplotlibApp(root)
    root.mainloop()