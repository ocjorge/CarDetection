import os
import cv2
from ultralytics import YOLO
import shutil  # Opcional, si decides limpiar carpetas
import glob  # Para buscar archivos

print("Paso 0: Librerías importadas correctamente.")

# ==============================================================================
# PASO 1: CONFIGURACIÓN DE RUTAS
# ==============================================================================
print("\nPaso 1: Configurando rutas...")

# Rutas para tu entorno local (ajusta si es necesario)
MODEL_PATH = 'F:/Documents/PycharmProjects/Deteccion/best.pt'  # <<<< VERIFICA ESTA RUTA
VIDEO_INPUT_PATH = 'F:\Documents\PycharmProjects\Deteccion\GH012372_no_audio.mp4'  # <<<< VERIFICA ESTA RUTA

# Carpeta local donde se guardarán los resultados (similar al script original)
OUTPUT_PROJECT_DIR = "./runs_original_working_structure/detect_video"
VIDEO_BASENAME_NO_EXT_ORIG = os.path.splitext(os.path.basename(VIDEO_INPUT_PATH))[0]
OUTPUT_RUN_NAME = f"{VIDEO_BASENAME_NO_EXT_ORIG}_processed_original"  # Mantener estructura de nombre

# Crear directorio base si no existe
os.makedirs(OUTPUT_PROJECT_DIR, exist_ok=True)

# Opcional: Limpiar la carpeta del run específico si quieres un inicio limpio cada vez.
# Esto es útil si exist_ok=True y no quieres que YOLO simplemente sobrescriba.
# output_run_folder_to_clean_orig = os.path.join(OUTPUT_PROJECT_DIR, OUTPUT_RUN_NAME)
# if os.path.exists(output_run_folder_to_clean_orig):
#     print(f"Limpiando directorio de run previo: {output_run_folder_to_clean_orig}")
#     shutil.rmtree(output_run_folder_to_clean_orig)


print(f"Ruta del modelo: {os.path.abspath(MODEL_PATH)}")
print(f"Ruta del video de entrada: {os.path.abspath(VIDEO_INPUT_PATH)}")
print(
    f"Resultados se guardarán en la carpeta de run: {os.path.abspath(os.path.join(OUTPUT_PROJECT_DIR, OUTPUT_RUN_NAME))}")

# ==============================================================================
# PASO 2: CARGAR EL MODELO
# ==============================================================================
print("\nPaso 2: Cargando el modelo...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print(f"Modelo '{MODEL_PATH}' cargado correctamente.")
print(f"Clases conocidas por el modelo: {model.names}")

# ==============================================================================
# PASO 3: VERIFICAR VIDEO DE ENTRADA
# ==============================================================================
print("\nPaso 3: Verificando video de entrada...")
if not os.path.exists(VIDEO_INPUT_PATH):
    raise FileNotFoundError(f"Video de entrada no encontrado en: {VIDEO_INPUT_PATH}")

try:
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {VIDEO_INPUT_PATH}")

    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds_orig = frame_count_orig / fps_orig if fps_orig > 0 else 0
    print(
        f"Video verificado: {width_orig}x{height_orig} @ {fps_orig:.2f} FPS, Frames: {frame_count_orig}, Duración: ~{duration_seconds_orig:.2f}s")
    cap.release()
except Exception as e_verif_orig:
    print(f"Error al verificar el video: {e_verif_orig}")
    raise

# ==============================================================================
# PASO 4: PROCESAR VIDEO CON YOLO.PREDICT
# (Usando los mismos parámetros que el script que dijiste que funcionó)
# ==============================================================================
print("\nPaso 4: Procesando video con YOLO.predict (save=True)...")

CONF_ORIG = 0.3  # Parámetro del script que funcionó
IOU_ORIG = 0.5  # Parámetro del script que funcionó
# El script original no especificaba imgsz, así que YOLO usaría su default o el tamaño del modelo.
# Para replicar, no lo especificaremos aquí tampoco inicialmente.

processed_frames_orig = 0
actual_run_save_dir_orig = None  # Para almacenar la ruta del directorio donde YOLO guarda

try:
    results_generator_orig = model.predict(
        source=VIDEO_INPUT_PATH,
        stream=True,
        save=True,
        project=OUTPUT_PROJECT_DIR,
        name=OUTPUT_RUN_NAME,
        exist_ok=True,  # Importante para poder re-ejecutar sin que cree carpetas ...2, ...3
        conf=CONF_ORIG,
        iou=IOU_ORIG
        # No se especifica imgsz para replicar el script original que funcionó.
        # Si ese script tenía un imgsz implícito o si tu modelo tiene un tamaño fijo, YOLO lo usará.
        # Puedes añadir imgsz=640 si crees que es necesario o estaba implícito.
    )

    print(f"Iterando sobre los resultados de la predicción del video '{os.path.basename(VIDEO_INPUT_PATH)}'...")
    for i, result_orig in enumerate(results_generator_orig):
        processed_frames_orig += 1
        if i == 0:
            if hasattr(result_orig, 'save_dir') and result_orig.save_dir:
                actual_run_save_dir_orig = result_orig.save_dir
                print(f"  Directorio de guardado para este run: {actual_run_save_dir_orig}")

        if (processed_frames_orig) % 100 == 0:
            print(f"  Procesado frame {processed_frames_orig}...")

    print(f"Procesamiento de video con YOLO.predict completado. Total de frames procesados: {processed_frames_orig}")

except Exception as e_proc_orig:
    print(f"Ocurrió un error durante el procesamiento del video con YOLO.predict: {e_proc_orig}")
    import traceback

    traceback.print_exc()

# ==============================================================================
# PASO 5: ENCONTRAR Y MOSTRAR RUTA DEL VIDEO DE SALIDA (LÓGICA MEJORADA)
# ==============================================================================
print("\nPaso 5: Buscando el video de salida procesado...")

video_output_path_found_orig = None

if actual_run_save_dir_orig and os.path.isdir(actual_run_save_dir_orig):
    print(f"  Buscando video en el directorio del run: {actual_run_save_dir_orig}")
    video_extensions_orig = ('.mp4', '.avi', '.mov', '.mkv')
    found_videos_in_run_orig = []
    for f_name_orig in os.listdir(actual_run_save_dir_orig):
        if f_name_orig.lower().endswith(video_extensions_orig):
            found_videos_in_run_orig.append(f_name_orig)

    if found_videos_in_run_orig:
        input_video_basename_no_ext_orig_match = os.path.splitext(os.path.basename(VIDEO_INPUT_PATH))[0]
        for video_file_orig in found_videos_in_run_orig:
            if os.path.splitext(video_file_orig)[0] == input_video_basename_no_ext_orig_match:
                video_output_path_found_orig = os.path.join(actual_run_save_dir_orig, video_file_orig)
                break
        if not video_output_path_found_orig and found_videos_in_run_orig:
            video_output_path_found_orig = os.path.join(actual_run_save_dir_orig, found_videos_in_run_orig[0])

        if video_output_path_found_orig:
            print(f"  Video de salida detectado automáticamente: {video_output_path_found_orig}")
        else:
            print(
                f"  ⚠️ No se pudo determinar el nombre específico del video de salida en '{actual_run_save_dir_orig}', aunque se encontraron archivos de video: {found_videos_in_run_orig}")
            if found_videos_in_run_orig:  # Si no se pudo matchear por nombre pero hay videos, toma el primero.
                video_output_path_found_orig = os.path.join(actual_run_save_dir_orig, found_videos_in_run_orig[0])
                print(f"  Usando el primer video encontrado: {video_output_path_found_orig}")

    else:
        print(
            f"  ⚠️ No se encontraron archivos de video ({', '.join(video_extensions_orig)}) en: {actual_run_save_dir_orig}")
else:
    # Fallback si actual_run_save_dir_orig no se pudo determinar
    expected_run_dir_orig = os.path.join(OUTPUT_PROJECT_DIR, OUTPUT_RUN_NAME)
    print(f"  Directorio de guardado del run ('actual_run_save_dir_orig') no se determinó o no existe.")
    print(f"  Intentando verificar la ruta esperada: {expected_run_dir_orig}")
    if os.path.isdir(expected_run_dir_orig):
        print(f"  Contenido de la carpeta esperada '{expected_run_dir_orig}': {os.listdir(expected_run_dir_orig)}")

if video_output_path_found_orig and os.path.exists(video_output_path_found_orig):
    print(f"\n✅ Video procesado guardado exitosamente en:")
    print(f"   {os.path.abspath(video_output_path_found_orig)}")
    print("\n   Puedes abrir este archivo con tu reproductor de video.")

    try:
        cap_out_final = cv2.VideoCapture(video_output_path_found_orig)
        if cap_out_final.isOpened():
            fps_out_final = cap_out_final.get(cv2.CAP_PROP_FPS)
            frame_count_out_final = int(cap_out_final.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds_out_final = frame_count_out_final / fps_out_final if fps_out_final > 0 else 0
            print(f"\n   Verificación del video de salida:")
            print(
                f"   Frames: {frame_count_out_final}, Duración: ~{duration_seconds_out_final:.2f}s @ {fps_out_final:.2f} FPS")
            cap_out_final.release()
        else:
            print("\n   No se pudo abrir el video de salida para verificar su duración.")
    except Exception as e_verif_final:
        print(f"\n   Error verificando el video de salida: {e_verif_final}")
else:
    print("\n⚠️ No se pudo encontrar o confirmar el video de salida procesado.")
    print("   Verifica la consola por errores durante el procesamiento y el contenido de las carpetas de salida.")

print("\n--- Proceso de inferencia de video en local (Estructura Original) Finalizado ---")
