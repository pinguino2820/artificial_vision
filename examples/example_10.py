#---------------------------------------------------------------------------
# example_10.py 
# contando vehículos usando 2 líneas horizontales
#
# Cuidado: tracker_02.py es distinto de tracker.py
#---------------------------------------------------------------------------

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker_02 import Tracker

# model = YOLO('yolov8s.pt')
model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('videos/highway_02.mp4')
if not cap.isOpened():
    raise RuntimeError('No se pudo abrir el vídeo: videos/highway_02.mp4')

my_file = open('coco.txt', 'r')
data = my_file.read()
class_list = data.split('\n')

count = 0
tracker = Tracker()
vehicle_classes = {'car', 'truck', 'bus'}

cy1 = 350
cy2 = 400
offset = 20

vh_down = {}
vh_up = {}
counterDown = set()
counterUp = set()

image = cv2.imread('assets/image_02.png', cv2.IMREAD_UNCHANGED)
if image is None:
    raise RuntimeError('No se pudo cargar la imagen: assets/image_01.png')
image = cv2.resize(image, (80, 80))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (1020, 600))
    results = model.predict(frame, conf=0.5, iou=0.5, agnostic_nms=True)    
    
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype('float')
    
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2]) 
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        if c in vehicle_classes:
           list.append([x1, y1, x2, y2])
    
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        
        if cy1 < (cy + offset) and cy1 > (cy - offset): 
            vh_down[id] = cy
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):                       
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                counterDown.add(id)
                
        if cy2 < (cy + offset) and cy2 > (cy - offset): 
            vh_up[id] = cy
        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):                       
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                counterUp.add(id)

    cv2.line(frame, (40, cy1), (980, cy1), (255, 255, 255), 1)
    cv2.putText(frame, ('Line 1'), (950, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    cv2.line(frame, (10, cy2), (1010, cy2), (255, 255, 255), 1)
    cv2.putText(frame, ('Line 2'), (950, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)    

    # pongo color al fondo y al texto de los contadores    
    down_text = 'going down: ' + str(len(counterDown))
    up_text = 'going up:    ' + str(len(counterUp))
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    thickness = 2

    for text, position, color in (
        (down_text, (690, 80), (0, 0, 255)),
        (up_text, (690, 40), (0, 255, 0)),
    ):
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        cv2.rectangle(
            frame,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + baseline + 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
    
    # coloco una imagen solo para mostrar cómo es posible hacerlo
    image_height, image_width = image.shape[:2]
    y1 = (frame.shape[0] - image_height) // 2
    x1 = 10
    y2 = y1 + image_height
    x2 = x1 + image_width
    alpha = image[:, :, 3:] / 255.0
    frame[y1:y2, x1:x2] = (
        alpha * image[:, :, :3] + (1.0 - alpha) * frame[y1:y2, x1:x2]
    ).astype(np.uint8)
    
    cv2.imshow('RGB', frame)
    
    if cv2.waitKey(1)&0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
