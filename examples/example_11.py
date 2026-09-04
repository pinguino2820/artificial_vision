# ---------------------------------------------------------------------------
# example_11.py
# Contador de vehículos usando una solo línea de referencia
# avance: 22:00
# ---------------------------------------------------------------------------

# imports required libraries
import cv2
import cvzone
import torch
from ultralytics import YOLO
import time

# load YOLO model
model = YOLO('yolov8s.pt')
names = model.names
device = 0 if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# define vertical line's X position
line_y = 400

# track previous center positions
track_hist = {}

# IN/OUT counters
car_in = 0
car_out = 0
bus_in = 0
bus_out = 0
truck_in = 0
truck_out = 0

# open video file or webcam
cap = cv2.VideoCapture('videos/highway_02.mp4')   # use 0 for webcam

frame_count = 0
fps_samples = []
previous_processed_time = None
source_fps = cap.get(cv2.CAP_PROP_FPS)
skip_factor = 3
display_duration = skip_factor / source_fps

# define the mouse callback function
def RGB(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            print(f'Mouse move to: [{x}, {y}]')
            
# create a named OpenCV window and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

while True:
    # read video frame
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 3 != 0:
        continue    # cada 3 frames descarta 2
    
    frame_start = time.perf_counter()
    if previous_processed_time is not None:
        elapsed = frame_start - previous_processed_time
        fps_samples.append(1.0 / elapsed)
        fps_samples = fps_samples[-30:]
    
    previous_processed_time = frame_start
    processed_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
    playback_speed = (processed_fps * skip_factor) / source_fps

    frame = cv2.resize(frame, (1020, 600))
    
    # detect and track cars, buses and trucks
    results = model.track(
        frame, 
        #persist=True, 
        classes=[2, 5, 7], 
        device=device, 
        conf=0.5, 
        iou=0.5, 
        agnostic_nms=True,
        #imgsz=640
    )
    
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            # center box coordinates            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            name = names[class_id]
                                     
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 1)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cvzone.putTextRect(frame, f'{name}', (x1, y1), scale=1, thickness=1, colorT=(255, 255, 255), colorR=(0, 128, 0))
            cvzone.putTextRect(frame, f'{track_id}', (x2, y2), scale=1, thickness=1, colorT=(255, 255, 255), colorR=(0, 128, 120))
            
            # stores the center (cx, cy) in the track_id position of track_hist
            track_hist[track_id] = (cx, cy)
        
    #cv2.putText(frame, f'car_in: {car_in}', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    #cv2.putText(frame, f'car_out: {car_out}', (640, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    #cv2.putText(frame, f'bus_in: {bus_in}', (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    #cv2.putText(frame, f'bus_out: {bus_out}', (640, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    #cv2.putText(frame, f'truck_in: {truck_in}', (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    #cv2.putText(frame, f'truck_out: {truck_out}', (640, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 255, 255), 2)
        
    cv2.putText(
    frame,
    f'Procesados: {processed_fps:.1f} FPS',
    (420, 540),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2,
    )

    cv2.putText(
        frame,
        f'Velocidad: {playback_speed:.2f}x',
        (420, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    
    print(track_hist)
    
    processing_time = time.perf_counter() - frame_start
    delay = max(1, int((display_duration - processing_time) * 1000))

    cv2.imshow('RGB', frame)
    
    if cv2.waitKey(delay) & 0xFF == 27:
        break
# end of while

cap.release()
cv2.destroyAllWindows()

    
"""
Estamos viendo ahora una velocidad de 0.91x, bastante cercana a la original 
y la tasa de frames procesadas es 7,6

Es un resultado consistente: velocidad = (7.6 x 3) / 25 = 0.912x

Es decir, procesas y muestras unos 7.6 FPS, pero cada frame mostrado representa
tres frames de la fuente. Por eso el vídeo avanza a unos 22.8 FPS efectivos 
frente a los 25 FPS originales: aproximadamente 0.91x.

La diferencia restante se debe a que el coste de model.track(...), el dibujo y 
la interfaz supera un poco el presupuesto de 120 ms por frame mostrado: 3 / 12 = 0.12s

Con 7.6 FPS, cada frame procesado tarda unos: 1 / 7.6 ≈ 0.132 s

Como el procesamiento ya tarda más que 120 ms, el cálculo deja delay en 1 ms; 
no puede recuperar esos alrededor de 12 ms extra.
"""

"""
Para acercarte a 1.0x, tienes tres vías:

* Reducir imgsz de 640 a 576 o 512. Es la primera prueba razonable; verifica que los
vehículos lejanos sigan detectándose.

* Usar GPU CUDA si actualmente el mensaje dice Using device: cpu.

* Cambiar a un modelo más ligero, por ejemplo yolo11n.pt o yolo26n.pt, comparando antes 

la precisión.
No intentaría reducir el salto a 1 de cada 2: exigiría procesar 12.5 FPS y el vídeo 
volvería a ralentizarse. Con la configuración actual, 0.91x es una reproducción 
bastante próxima a la velocidad original y mantiene un intervalo de tracking 
manejable para vehículos.
"""

"""
La duda razonable acá es qué pasaría si estuviéramos procesando un video en tiempo real, 
por ejemplo el proveniente de una cámara IP colocada en algún lugar de una autopista.

En una cámara IP en tiempo real no puedes “recuperar” los frames que no procesaste: 
mientras YOLO analiza un frame, la cámara continúa produciendo imágenes. 
Si procesas 7.6 FPS y la cámara entrega 25 FPS, se acumularían:
    25 - 7.6 = 17.4 frames por segundo

Si OpenCV mantiene una cola interna, esa acumulación genera latencia: 
el vídeo mostrado quedaría cada vez más atrasado respecto a lo que ocurre en la autopista.

Para analítica en vivo, normalmente no se intenta conservar todos los frames. 
La regla es: procesar siempre el frame más reciente y descartar los atrasados. 
Así sacrificas continuidad visual, pero mantienes una latencia baja.

La arquitectura recomendada es:

Cámara IP RTSP ->
    Hilo de captura -> 
        Buffer de un frame -> 
            YOLO y tracker ->
                Vista y conteo

El hilo de captura lee la cámara continuamente. En vez de una cola ilimitada, 
deja solo el último frame disponible: el hilo de inferencia toma el que exista y, 
si llegaron otros mientras YOLO trabaja, los anteriores se sustituyen.

Un patrón básico sería:

    import threading
    import cv2

    latest_frame = None
    frame_lock = threading.Lock()
    running = True

    def capture_frames():
        global latest_frame

        cap = cv2.VideoCapture(
            'rtsp://usuario:contrasena@IP_DE_LA_CAMARA:554/stream'
        )
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while running:
            ok, frame = cap.read()
            if not ok:
                continue

            with frame_lock:
                latest_frame = frame

    threading.Thread(target=capture_frames, daemon=True).start()
    
Y el bucle de YOLO toma una copia del último frame:

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            continue

        results = model.track(
            frame,
            classes=[2, 5, 7],
            device=device,
            conf=0.5,
            iou=0.5,
            agnostic_nms=True,
            persist=True,
            imgsz=640,
        )

        # Dibujo, conteo y cv2.imshow(...)

Dos observaciones importantes para tu caso:

* En una autopista, 7 a 10 FPS suele bastar para conteo si los vehículos no 
atraviesan demasiados píxeles entre inferencias y el tracker conserva los IDs. 
Ajusta la distancia entre líneas o la tolerancia de cruce a la velocidad y ángulo 
de la cámara.

* Al pasar de archivo a cámara, activa persist=True; mantiene el estado de seguimiento 
entre llamadas. 

También debes diseñar reconexión RTSP, validar cap.isOpened(), y evitar poner 
credenciales de cámara en el código: usa variables de entorno o un archivo de 
configuración ignorado por Git. Para uso real, la prioridad no es reproducir exactamente 
a 25 FPS, sino detectar el estado reciente con latencia baja y sin duplicar conteos.    
"""