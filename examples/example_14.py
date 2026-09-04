"""
example_13.py

Ejemplo de integración con YOLO11, filtrado por clases.
"""

#____________________________________________________________________________________
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
#____________________________________________________________________________________

import cv2
import threading
import time
import torch
from ultralytics import YOLO

#------------------------------------------------------------------------------------
# BLOQUE 1: Captura Asíncrona de Video (Manejo de Latencia RTSP)
#------------------------------------------------------------------------------------

class VideoStreamIP:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(
            src,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ],
        )
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la camara RTSP.")

        self.lock = threading.Lock()
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret, self.frame = ret, frame
                else:
                    time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

#------------------------------------------------------------------------------------
# BLOQUE 2: Configuración del Modelo y Filtro de Clases
#------------------------------------------------------------------------------------

def main():
    # 1. Carga de la arquitectura YOLO11 preentrenada en el dataset COCO (80 clases)
    model = YOLO("yolo11n.pt")
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # 2. Lista de IDs de las clases que nos interesa filtrar (Dataset COCO)
    car_id = list(model.names.values()).index("car")
    moto_id = list(model.names.values()).index("motorcycle")
    bus_id = list(model.names.values()).index("bus")
    truck_id = list(model.names.values()).index("truck")
    person_id = list(model.names.values()).index("person")
       
    #TARGET_CLASSES = [car_id, moto_id, bus_id, truck_id]
    TARGET_CLASSES = [person_id]
    
    # 2. URL RTSP de tu cámara IP
    rtsp_url = "rtsp://admin:LKKHTL@192.168.0.2:554/streaming/channels/102"

    print("Conectando a la cámara IP...")
    cam = VideoStreamIP(rtsp_url)

    fps_samples = []
    previous_processed_time = None
    speed_totals = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    speed_sample_count = 0
    speed_window_start = time.perf_counter()
    average_speeds = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    
    # dimensiona tamaño de la pantalla de visualización
    cv2.namedWindow("Reconocimiento YOLO11", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reconocimiento YOLO11", 960, 540)

    # ------------------------------------------------------------------------------
    # BLOQUE 3: Bucle Principal e Inferencia Filtrada
    # ------------------------------------------------------------------------------

    try:
        while True:
            ret, frame = cam.read()

            # Verificar si se obtuvo un fotograma válido
            if not ret or frame is None:
                continue

            frame_start = time.perf_counter()
            
            if previous_processed_time is not None:
                elapsed = frame_start - previous_processed_time
                fps_samples.append(1.0 / elapsed)
                fps_samples = fps_samples[-30:]
            
            previous_processed_time = frame_start
            processed_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0

            # Inferencia filtrada:
            # - classes: limita las detecciones solo a los IDs indicados
            # - conf: umbral mínimo de confianza (ej. 0.5 = 50%)
            results = model(
                frame, 
                classes=TARGET_CLASSES, 
                conf=0.40, 
                device=device, 
                stream=True,
                imgsz=960
            )

            # 4. Dibujar los resultados en el cuadro
            for r in results:
                annotated_frame = frame.copy()
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.int().cpu().tolist()

                for stage in speed_totals:
                    speed_totals[stage] += r.speed[stage]
                speed_sample_count += 1

                for box, conf, class_id in zip(boxes, confs, class_ids):
                    x1, y1, x2, y2 = box
                    label = f'{model.names[class_id]} {conf:.2f}'
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2,
                    )

            current_time = time.perf_counter()
            if current_time - speed_window_start >= 1.0 and speed_sample_count:
                average_speeds = {
                    stage: total / speed_sample_count
                    for stage, total in speed_totals.items()
                }
                speed_totals = {stage: 0.0 for stage in speed_totals}
                speed_sample_count = 0
                speed_window_start = current_time

            performance_texts = [
                (f'Preprocess: {average_speeds["preprocess"]:.1f} ms', 0.5),
                (f'Inference: {average_speeds["inference"]:.1f} ms', 0.5),
                (f'Postprocess: {average_speeds["postprocess"]:.1f} ms', 0.5),
                (f'Procesados: {processed_fps:.1f} FPS', 0.6),
            ]
            text_margin = 20
            line_spacing = 25
            for index, (text, font_scale) in enumerate(performance_texts):
                text_width, _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
                )[0]
                text_x = annotated_frame.shape[1] - text_width - text_margin
                text_y = (
                    annotated_frame.shape[0]
                    - text_margin
                    - line_spacing * (len(performance_texts) - 1 - index)
                )
                cv2.putText(
                    annotated_frame,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 0, 0),
                    2,
                )

            # 5. Mostrar la imagen con las detecciones
            cv2.imshow("Reconocimiento YOLO11", annotated_frame)

            # Presiona 'q' para salir del bucle
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Liberar recursos y cerrar ventanas
        print("Cerrando la transmisión...")
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#  2ms preprocess
# 20ms inference
#  3ms postprocess
# ------------------
# 25ms total
#
# 33ms per frame