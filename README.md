## CarDetection
# Proyecto de Detección de Objetos y Estimación de Distancia en Video

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/) 
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![PyTorch](https://img.shields.io/badge/pytorch-%20-violet.svg)](https://pytorch.org/) 
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-4a0072?logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-blue?logo=opencv)](https://opencv.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) 

Este proyecto utiliza el modelo YOLOv8 de Ultralytics para la detección en tiempo real (o en video procesado) de vehículos y otros objetos comunes. Adicionalmente, implementa una estimación de distancia monocular basada en el tamaño conocido de los objetos y los parámetros de la cámara.

El desarrollo incluye tanto un script para ejecución local como un notebook de Google Colab para entrenamiento y experimentación en la nube.

## Características Principales

*   **Detección de Múltiples Clases:**
    *   Utiliza un modelo YOLOv8 fine-tuneado para la detección especializada de vehículos (ej. carros, mototaxis, buses, camiones, motocicletas, vans).
    *   Complementa con un modelo YOLOv8 pre-entrenado en COCO para detectar objetos comunes como personas, bicicletas y perros.
*   **Estimación de Distancia:** Calcula una distancia aproximada a los objetos detectados utilizando su tamaño real conocido y la distancia focal de la cámara.
*   **Procesamiento de Video:** Capaz de procesar archivos de video, aplicando las detecciones y estimaciones de distancia frame a frame, y generando un video de salida con las anotaciones.
*   **Entrenamiento / Fine-tuning:** Incluye scripts (principalmente en el notebook de Colab) para el fine-tuning de modelos YOLOv8 con datasets personalizados.
*   **Flexibilidad:** Puede ser ejecutado tanto en un entorno local (con GPU si está disponible) como en Google Colab.
*   

## Configuración del Entorno Local

### Prerrequisitos

*   [Python](https://www.python.org/downloads/) (versión 3.8 o superior recomendada)
*   [pip](https://pip.pypa.io/en/stable/installation/) (generalmente viene con Python)
*   [FFmpeg](https://ffmpeg.org/download.html) (esencial para el procesamiento de video con OpenCV y YOLO). Asegúrate de que esté instalado y añadido a tu variable de entorno PATH.
 
### Pasos de Instalación

1.  **Clona el repositorio (si está en GitHub):**
    ```bash
    git clone https://github.com/[ocjorge]/[CarDetection].git
    cd [NOMBRE_DEL_REPO]
    ```

2.  **Crea un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Activa el entorno virtual
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instala las Dependencias:**
    Asegúrate de tener un archivo `requirements.txt` con las librerías necesarias. Como mínimo:
    ```
    ultralytics>=8.0.0
    opencv-python
    numpy
    matplotlib
    # torch torchvision torchaudio (Ultralytics suele instalarlos si son necesarios para GPU)
    ```
    Luego ejecuta:
    ```bash
    pip install -r requirements.txt
    ```
    Si no tienes `requirements.txt`, puedes instalar las principales directamente:
    ```bash
    pip install ultralytics opencv-python numpy matplotlib
    ```

### Configuración del Script Local (`src/process_video_local.py`)

Antes de ejecutar el script local, asegúrate de configurar las siguientes rutas y parámetros dentro del archivo:

*   `MODEL_PATH_VEHICLES`: Ruta a tu archivo `.pt` del modelo fine-tuneado para vehículos.
*   `VIDEO_INPUT_PATH`: Ruta al video que deseas procesar.
    *   **Nota Importante sobre Videos:** Se ha observado que algunos archivos MP4 con ciertas codificaciones de audio pueden causar problemas de procesamiento. Se recomienda usar videos sin audio o re-codificarlos a un formato MP4 limpio (ej. usando FFmpeg: `ffmpeg -i entrada.mp4 -c:v copy -an salida_no_audio.mp4`) para una mayor estabilidad.
*   `FOCAL_LENGTH_PX`: Distancia focal de la cámara en píxeles (¡CRUCIAL para la estimación de distancia!).
*   `REAL_OBJECT_SIZES_M`: Diccionario con los tamaños reales (ancho o alto en metros) de los objetos a detectar.
*   `CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD_NMS`: Umbrales para la detección.

### Ejecución Local

Navega al directorio `src` (o donde esté tu script principal) y ejecuta:
```bash
python process_video_local.py
**Cómo añadir los Shields (Insignias):**

