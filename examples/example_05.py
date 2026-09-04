"""
example_05.py

Ejemplo de integración con YOLO11
"""

#____________________________________________________________________________________
import os

"""
Esta línea configura una variable de entorno antes de abrir la cámara con OpenCV:
`OPENCV_FFMPEG_CAPTURE_OPTIONS` pasa opciones adicionales al backend **FFmpeg** que 
OpenCV utiliza para procesar flujos RTSP.

El valor `"rtsp_transport;tcp"` indica que el stream RTSP debe transportarse por 
**TCP**, en vez de UDP. TCP suele ser más fiable en redes locales o cámaras IP 
con pérdida de paquetes, firewalls o configuraciones que no manejan bien UDP.

Debe ejecutarse **antes** de crear `cv2.VideoCapture(...)`. De lo contrario, 
FFmpeg puede abrir la cámara sin aplicar esta opción.

Como contrapartida, TCP puede introducir algo más de latencia que UDP, ya que 
prioriza recibir los datos correctamente antes que descartar paquetes perdidos.
"""
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
#____________________________________________________________________________________

import cv2
import threading
import time
from ultralytics import YOLO

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


def main():
    # 1. Cargar el modelo YOLO11 (yolo11n.pt es la versión Nano, ideal para tiempo real)
    model = YOLO("yolo11n.pt")

    # 2. URL RTSP de tu cámara IP
    rtsp_url = "rtsp://admin:AAAAAA@192.168.0.100:554/streaming/channels/102"

    print("Conectando a la cámara IP...")
    cam = VideoStreamIP(rtsp_url)
    
    # dimensiona tamaño de la pantalla de visualización
    cv2.namedWindow("Reconocimiento YOLO11 en vivo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reconocimiento YOLO11 en vivo", 768, 432)

    try:
        while True:
            ret, frame = cam.read()

            # Verificar si se obtuvo un fotograma válido
            if not ret or frame is None:
                continue

            # 3. Inferencia con YOLO11
            # stream = True ayuda con el manejo eficiente de memoria en streams continuos
            results = model(frame, stream=True)

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


