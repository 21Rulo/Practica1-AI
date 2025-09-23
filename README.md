# Practica1-AI

Una aplicación para el análisis y procesamiento básico de imágenes. La interfaz gráfica está construida con **Tkinter**, el procesamiento de imágenes se realiza con **OpenCV** y **Pillow**, y las visualizaciones (histogramas y canales) se generan con **Matplotlib**.

## ✨ Características Principales

Esta herramienta permite a los usuarios realizar las siguientes operaciones:

  * **Carga de Imágenes**: Soporta los formatos de imagen más comunes (`.jpg`, `.png`, `.bmp`.).
  * **Visualización de Canales RGB**: Separa una imagen a color en sus componentes Rojo, Verde y Azul y los muestra individualmente.
  * **Conversión a Escala de Grises**: Transforma la imagen original a su representación en escala de grises.
  * **Binarización de Imágenes**: Convierte una imagen en escala de grises a una imagen binaria (blanco y negro) usando un umbral fijo.
  * **Análisis de Histogramas**: Genera y visualiza histogramas de color para:
      * La imagen original (canales RGB superpuestos).
      * Cada canal RGB por separado.
      * La imagen en escala de grises.
      * La imagen binarizada.
  * **Cálculo de Características Estadísticas**: Extrae métricas clave de la imagen actual (ya sea RGB, grises o binaria):
      * **Energía**: Medida de la uniformidad de la imagen.
      * **Entropía**: Medida de la aleatoriedad o información en la imagen.
      * **Asimetría (Skewness)**: Indica si la distribución de intensidades está sesgada.
      * **Media**: El nivel de intensidad promedio.
      * **Varianza**: La dispersión de los niveles de intensidad.
  * **Guardado de Resultados**: Guarda la imagen en su estado actual (original, canales separados, grises o binarizada).
  * **Reversión**: Permite descartar todos los cambios y volver a la imagen original cargada con un solo clic.

-----

## 🛠️ Tecnologías Utilizadas

  * **Python 3.12.8**: Lenguaje de programación principal.
  * **Tkinter**: Para la creación de la interfaz gráfica de usuario (GUI).
  * **Pillow (PIL Fork)**: Para la carga y manipulación básica de imágenes.
  * **OpenCV-Python**: Para las operaciones de procesamiento de imágenes (conversión de color, binarización, cálculo de histogramas).
  * **NumPy**: Para el manejo eficiente de arreglos y matrices de imágenes.
  * **Matplotlib**: Para incrustar y renderizar los gráficos (imágenes e histogramas) dentro de la interfaz de Tkinter.
  * **SciPy**: Para cálculos estadísticos avanzados como la asimetría (`skew`).

-----

## 🚀 Instalación y Puesta en Marcha

Sigue estos pasos para ejecutar la aplicación en tu máquina local.

1.  **Clona el repositorio** (o simplemente descarga el archivo `.py`):

    ```bash
    git clone https://github.com/21Rulo/tu-repositorio.git
    cd tu-repositorio
    ```

2.  **Crea un entorno virtual** (recomendado para mantener las dependencias aisladas):

    ```bash
    # En Windows
    python -m venv venv
    .\venv\Scripts\activate

    # En macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instala las dependencias necesarias**:
    Puedes crear un archivo `requirements.txt` con el siguiente contenido:

    ```txt
    numpy
    opencv-python
    Pillow
    matplotlib
    scipy
    ```

    Y luego instalarlo con pip:

    ```bash
    pip install -r requirements.txt
    ```

-----

## 📖 ¿Cómo se usa?

1.  **Ejecuta el script** desde tu terminal:

    ```bash
    python s1.py
    ```

2.  **Cargar Imagen**: Haz clic en el botón **"Cargar Imagen"** para seleccionar un archivo de imagen de tu sistema. La imagen se mostrará en el panel principal.

3.  **Aplicar Transformaciones**:

      * **"Separar Canales RGB"**: Muestra los tres canales de color.
      * **"Escala de Grises"**: Convierte la imagen original a grises.
      * **"Binarizar Imagen"**: Convierte la imagen en escala de grises a blanco y negro. Debes pasar a escala de grises primero.

4.  **Analizar la Imagen**:

      * **"Histograma"**: Muestra el histograma correspondiente a la vista actual (RGB, grises, etc.).
      * **"Características"**: Abre una nueva ventana con las métricas estadísticas de la imagen actual y ofrece una breve interpretación.

5.  **Guardar y Exportar**:

      * **"Guardar Actual"**: Guarda la imagen que se está mostrando. Si estás viendo los canales RGB, guardará los 3 canales como archivos separados (`_R.png`, `_G.png`, `_B.png`).

6.  **Revertir**: Si en cualquier momento quieres volver a la imagen original, haz clic en **"Revertir"**.

-----

## 📁 Estructura del Código

El código está organizado en tres secciones lógicas para mayor claridad:

1.  **Funciones de Procesamiento**: Contiene funciones puras que realizan una única tarea de procesamiento de imágenes (ej. `separar_canales`, `calcular_histograma_rgb`). Estas funciones no dependen de la interfaz gráfica, lo que las hace reutilizables.

2.  **Clase de la Aplicación GUI (`MatplotlibApp`)**: Encapsula toda la lógica de la interfaz gráfica. Gestiona los widgets de Tkinter, el lienzo de Matplotlib, el estado de la aplicación (qué imagen se está mostrando) y maneja los eventos de los botones, llamando a las funciones de procesamiento cuando es necesario.

3.  **Bloque Principal de Ejecución**: Es el punto de entrada del programa (`if __name__ == "__main__":`). Inicializa la ventana principal de Tkinter y crea una instancia de la clase `MatplotlibApp` para lanzar la aplicación.
