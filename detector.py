import os
import cv2
from ultralytics import YOLO
import shutil
import glob
import numpy as np  # Necesario para el manejo de arrays de YOLO y colores
import matplotlib.pyplot as plt  # Necesario para el colormap

print("Paso 0: Librerías importadas correctamente.")

# ==============================================================================
# PASO 1: CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
print("\nPaso 1: Configurando rutas y parámetros...")

MODEL_PATH_VEHICLES = 'F:/Documents/PycharmProjects/Deteccion/best.pt'
VIDEO_INPUT_PATH = 'F:/Documents/PycharmProjects/Deteccion/GH012372_no_audio.mp4'  # Usa el video sin audio o re-codificado

# Carpeta para el video de salida procesado manualmente
OUTPUT_DIR_MANUAL = "./runs_local/manual_video_processing"
os.makedirs(OUTPUT_DIR_MANUAL, exist_ok=True)
VIDEO_BASENAME = os.path.splitext(os.path.basename(VIDEO_INPUT_PATH))[0]
VIDEO_OUTPUT_PATH_MANUAL = os.path.join(OUTPUT_DIR_MANUAL, f"{VIDEO_BASENAME}_annotated_distance.mp4")

print(f"Modelo de vehículos: {os.path.abspath(MODEL_PATH_VEHICLES)}")
print(f"Video de entrada: {os.path.abspath(VIDEO_INPUT_PATH)}")
print(f"Video de salida se guardará en: {os.path.abspath(VIDEO_OUTPUT_PATH_MANUAL)}")

# Parámetros de inferencia y visualización
CONFIDENCE_THRESHOLD = 0.35
FOCAL_LENGTH_PX = 700  # <<<< AJUSTA ESTO PARA TU CÁMARA!
REAL_OBJECT_SIZES_M = {
    # De tu modelo de vehículos
    'car': 1.8, 'threewheel': 1.2, 'bus': 2.5, 'truck': 2.6, 'motorbike': 0.8, 'van': 2.0,
    # De COCO (asegúrate que los nombres coincidan con model_coco.names)
    'person': 0.5, 'bicycle': 0.4, 'dog': 0.3
}  # <<<< AJUSTA ESTOS VALORES!

COCO_CLASSES_TO_SEEK = ['person', 'bicycle', 'dog']  # Clases de COCO que nos interesan

# ==============================================================================
# PASO 2: CARGAR LOS MODELOS
# ==============================================================================
print("\nPaso 2: Cargando modelos...")

# Cargar tu modelo fine-tuned para vehículos
if not os.path.exists(MODEL_PATH_VEHICLES):
    raise FileNotFoundError(f"Modelo de vehículos no encontrado en: {MODEL_PATH_VEHICLES}")
model_vehicles = YOLO(MODEL_PATH_VEHICLES)
print(f"Modelo de vehículos '{MODEL_PATH_VEHICLES}' cargado. Clases: {model_vehicles.names}")

# Cargar modelo COCO pre-entrenado
model_coco = YOLO('yolov8n.pt')
print(f"Modelo COCO 'yolov8n.pt' cargado.")
# print(f"Clases COCO: {model_coco.names}") # Descomenta para ver todas

# Obtener los IDs de las clases de COCO que nos interesan
coco_target_ids = []
if isinstance(model_coco.names, dict):
    for name_to_seek in COCO_CLASSES_TO_SEEK:
        found_id = None
        for class_id_coco, class_name_coco in model_coco.names.items():
            if class_name_coco == name_to_seek:
                found_id = class_id_coco
                break
        if found_id is not None:
            coco_target_ids.append(found_id)
        else:
            print(f"ADVERTENCIA: Clase COCO '{name_to_seek}' no encontrada en model_coco.names.")
else:  # Fallback por si model.names no es dict (raro)
    for name_to_seek in COCO_CLASSES_TO_SEEK:
        try:
            coco_target_ids.append(model_coco.names.index(name_to_seek))
        except ValueError:
            print(f"ADVERTENCIA: Clase COCO '{name_to_seek}' no encontrada (names como lista).")
print(f"IDs de clases COCO a detectar: {coco_target_ids}")

# ==============================================================================
# PASO 3: INICIAR PROCESAMIENTO DE VIDEO MANUAL
# ==============================================================================
print("\nPaso 3: Iniciando procesamiento manual del video...")

if not os.path.exists(VIDEO_INPUT_PATH):
    raise FileNotFoundError(f"Video de entrada no encontrado en: {VIDEO_INPUT_PATH}")

cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
if not cap.isOpened():
    raise IOError(f"No se pudo abrir el video de entrada: {VIDEO_INPUT_PATH}")

# Obtener propiedades del video para el VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = cap.get(cv2.CAP_PROP_FPS)
total_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_output = fps_input if fps_input > 0 else 30.0

print(f"Video de entrada: {frame_width}x{frame_height} @ {fps_input:.2f} FPS, Total Frames: {total_frames_input}")

# Definir el codec y crear el objeto VideoWriter
# 'mp4v' es un buen codec para .mp4. Si falla, prueba 'XVID' (y cambia extensión a .avi)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(VIDEO_OUTPUT_PATH_MANUAL, fourcc, fps_output, (frame_width, frame_height))

if not out_video.isOpened():
    cap.release()
    raise IOError(f"No se pudo abrir el VideoWriter para el archivo de salida: {VIDEO_OUTPUT_PATH_MANUAL}")

print(f"Procesando video y guardando en: {VIDEO_OUTPUT_PATH_MANUAL}")
frames_processed_count = 0
frames_written_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Fin del video o error de lectura en el frame {frames_processed_count + 1}.")
            break

        frames_processed_count += 1
        annotated_frame = frame.copy()
        all_detections_current_frame = []  # Para este frame

        # 1. Inferencia con modelo de vehículos
        results_v = model_vehicles.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        if results_v:
            res_v_data = results_v[0]  # El resultado para el frame
            for box_v_data in res_v_data.boxes:
                all_detections_current_frame.append({
                    "xywh": box_v_data.xywh.cpu().numpy()[0],
                    "cls_id": int(box_v_data.cls.cpu().item()),
                    "conf": float(box_v_data.conf.cpu().item()),
                    "model_names_map": model_vehicles.names,
                    "is_vehicle": True  # Flag para diferenciar
                })

        # 2. Inferencia con modelo COCO (si hay clases objetivo)
        if coco_target_ids:
            results_c = model_coco.predict(source=frame, conf=CONFIDENCE_THRESHOLD, classes=coco_target_ids,
                                           verbose=False)
            if results_c:
                res_c_data = results_c[0]
                for box_c_data in res_c_data.boxes:
                    all_detections_current_frame.append({
                        "xywh": box_c_data.xywh.cpu().numpy()[0],
                        "cls_id": int(box_c_data.cls.cpu().item()),  # ID relativo a model_coco.names
                        "conf": float(box_c_data.conf.cpu().item()),
                        "model_names_map": model_coco.names,
                        "is_vehicle": False  # Flag para diferenciar
                    })

        # 3. Dibujar todas las detecciones del frame actual
        for det in all_detections_current_frame:
            x_c, y_c, w, h = det["xywh"]
            cls_id = det["cls_id"]
            conf_val = det["conf"]
            current_model_names = det["model_names_map"]

            label_name = current_model_names[cls_id]

            x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
            x2, y2 = int(x_c + w / 2), int(y_c + h / 2)

            # Color (simple: verde para vehículos, azul para COCO)
            box_clr = (0, 255, 0) if det["is_vehicle"] else (255, 0, 0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_clr, 2)

            dist_m = -1
            if label_name in REAL_OBJECT_SIZES_M and REAL_OBJECT_SIZES_M[label_name] > 0:
                w_real_m = REAL_OBJECT_SIZES_M[label_name]
                if w > 0: dist_m = (w_real_m * FOCAL_LENGTH_PX) / w

            txt_lbl = f"{label_name} {conf_val:.2f}"
            if dist_m > 0: txt_lbl += f" {dist_m:.1f}m"

            (txt_w, txt_h), _ = cv2.getTextSize(txt_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - txt_h - 4), (x1 + txt_w, y1 - 2), box_clr, -1)
            cv2.putText(annotated_frame, txt_lbl, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)

        # 4. Escribir el frame anotado
        out_video.write(annotated_frame)
        frames_written_count += 1

        if frames_processed_count % (int(fps_output) * 5) == 0:  # Progreso cada ~5 segundos
            print(f"  Procesado y escrito frame {frames_processed_count}/{total_frames_input}...")

except Exception as e_proc_manual:
    print(f"Error durante el procesamiento manual del video: {e_proc_manual}")
    import traceback

    traceback.print_exc()
finally:
    print(f"\nCerrando archivos de video...")
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()  # Aunque no se muestren ventanas, es buena práctica

print(f"\nProcesamiento manual de video completado.")
print(f"Total frames leídos del video de entrada: {frames_processed_count}")
print(f"Total frames escritos en el video de salida: {frames_written_count}")

if os.path.exists(VIDEO_OUTPUT_PATH_MANUAL) and frames_written_count > 0:
    print(f"\n✅ Video procesado manualmente guardado en:")
    print(f"   {os.path.abspath(VIDEO_OUTPUT_PATH_MANUAL)}")

    # Verificar duración del video de salida
    try:
        cap_out_check = cv2.VideoCapture(VIDEO_OUTPUT_PATH_MANUAL)
        if cap_out_check.isOpened():
            fps_out_chk = cap_out_check.get(cv2.CAP_PROP_FPS)
            fc_out_chk = int(cap_out_check.get(cv2.CAP_PROP_FRAME_COUNT))
            dur_s_out_chk = fc_out_chk / fps_out_chk if fps_out_chk > 0 else 0
            print(f"\n   Verificación del video de salida:")
            print(f"   Frames: {fc_out_chk}, Duración: ~{dur_s_out_chk:.2f}s @ {fps_out_chk:.2f} FPS")
            cap_out_check.release()
    except Exception as e_verif_final_manual:
        print(f"\n   Error verificando el video de salida: {e_verif_final_manual}")
else:
    print("\n⚠️ No se guardó el video de salida o no se escribieron frames.")

print("\n--- Proceso de Inferencia de Video Manual Local Finalizado ---")
