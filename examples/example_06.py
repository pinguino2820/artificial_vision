"""
example_05.py

Ejemplo de integración con YOLO11, filtrado por clases.
"""

#____________________________________________________________________________________
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
#____________________________________________________________________________________

import cv2
import threading
import time
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
    
    # 2. Lista de IDs de las clases que nos interesa filtrar (Dataset COCO)
    car_id = list(model.names.values()).index("car")
    moto_id = list(model.names.values()).index("motorcycle")
    bus_id = list(model.names.values()).index("bus")
    truck_id = list(model.names.values()).index("truck")   
    TARGET_CLASSES = [car_id, moto_id, bus_id, truck_id]
    
    # 2. URL RTSP de tu cámara IP
    rtsp_url = "rtsp://admin:LKKHTL@192.168.0.2:554/streaming/channels/102"

    print("Conectando a la cámara IP...")
    cam = VideoStreamIP(rtsp_url)
    
    # dimensiona tamaño de la pantalla de visualización
    cv2.namedWindow("Reconocimiento YOLO11 en vivo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reconocimiento YOLO11 en vivo", 768, 432)

    # ------------------------------------------------------------------------------
    # BLOQUE 3: Bucle Principal e Inferencia Filtrada
    # ------------------------------------------------------------------------------

    try:
        while True:
            ret, frame = cam.read()

            # Verificar si se obtuvo un fotograma válido
            if not ret or frame is None:
                continue

            # Inferencia filtrada:
            # - classes: limita las detecciones solo a los IDs indicados
            # - conf: umbral mínimo de confianza (ej. 0.5 = 50%)
            results = model(frame, classes=TARGET_CLASSES, conf=0.5, stream=True)

            # 4. Dibujar los resultados en el cuadro
            for r in results:
                annotated_frame = r.plot()

            # 5. Mostrar la imagen con las detecciones
            cv2.imshow("Reconocimiento YOLO11 en vivo", annotated_frame)

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

# FUNCIONA CORRECTAMENTE

# Página de ultralytics en español
# https://docs.ultralytics.com/es