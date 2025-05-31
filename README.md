## CarDetection
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

1.  **Elige tus Shields:** Ve a [shields.io](https://shields.io/). Puedes buscar shields para muchas cosas (versión de Python, licencia, estado de build, etc.).
2.  **Genera el Markdown:** El sitio te ayudará a generar el código Markdown para cada shield.
3.  **Copia y Pega:** Copia el Markdown generado y pégalo al principio de tu `README.md`.

**Ejemplos de Markdown para los Shields Incluidos:**

*   **Python Version:**
    ```markdown
    [![Python Version][python-shield]][python-url]

    [python-shield]: https://img.shields.io/badge/python-3.8+-blue.svg
    [python-url]: https://www.python.org/downloads/
    ```
*   **License:** (Asegúrate de tener un archivo LICENSE, por ejemplo, con el texto de la licencia MIT)
    ```markdown
    [![License: MIT][license-shield]][license-url]

    [license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
    [license-url]: https://opensource.org/licenses/MIT
    ```
*   **PyTorch:**
    ```markdown
    [![Pytorch][pytorch-shield]][pytorch-url]

    [pytorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white
    [pytorch-url]: https://pytorch.org/
    ```
*   **Ultralytics YOLOv8:**
    ```markdown
    [![Ultralytics YOLOv8][yolov8-shield]][yolov8-url]

    [yolov8-shield]: https://img.shields.io/badge/YOLOv8-Ultralytics- সেটাও.svg
    [yolov8-url]: https://github.com/ultralytics/ultralytics
    ```
*   **OpenCV:**
    ```markdown
    [![OpenCV][opencv-shield]][opencv-url]

    [opencv-shield]: https://img.shields.io/badge/OpenCV-blue?logo=opencv&logoColor=white
    [opencv-url]: https://opencv.org/
    ```
*   **Colab Notebook (Opcional):** Si haces público tu notebook:
    ```markdown
    [![Colab Notebook][colab-shield]](TU_ENLACE_PUBLICO_AL_NOTEBOOK_DE_COLAB)

    [colab-shield]: https://colab.research.google.com/assets/colab-badge.svg
    ```
