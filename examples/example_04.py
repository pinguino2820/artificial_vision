"""
example_04.py

Muestra el video generado en tiempo real por una cámara WiFi
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

# Uso con YOLO11
stream = VideoStreamIP("rtsp://admin:LKKHTL@192.168.0.2:554/streaming/channels/102")

# Permite redimensionar la ventana y establece ancho x alto.
cv2.namedWindow("Camara IP", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camara IP", 768, 432)

"""
No es obligatorio que los valores sean proporcionales: cuando usas cv2.resizeWindow(), 
OpenCV ajusta la visualización de la imagen dentro de la ventana y, normalmente, 
conserva su proporción (16:9 si el video de la cámara origen fuese 768:432). 
"""

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    # Aquí ejecutas tu modelo YOLO11
    # results = model(frame)

    cv2.imshow("Camara IP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # tecla 'q' cierra la ventana
        break

stream.stop()
cv2.destroyAllWindows()

# FUNCIONA CORRECTAMENTE
