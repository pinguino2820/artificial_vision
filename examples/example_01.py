import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('videos/cctv.mp4')


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

tracker = Tracker()
area_1 = [(377, 315), (429, 373), (535, 339), (500, 296)] 
area1 = set()

while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1020,500))
    cv2.polylines(frame, [np.array(area_1, np.int32)], True, (0,255,0), 3)
    
    results = model(frame)
    list = []
    
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        b = str(row['name'])
        
        if 'person' in b:        
            list.append([x1, y1, x2, y2])
                
    boxes_ids = tracker.update(list)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
        cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
        
        result = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(w), int(h)), False)
        if result > 0:
            area1.add(id)
    
    p = (len(area1))
    text_size, _ = cv2.getTextSize(str(p), cv2.FONT_HERSHEY_PLAIN, 3, 3)
    text_x = frame.shape[1] - text_size[0] - 200
    text_y = 50 + text_size[1]
    cv2.putText(frame, 'Count: ' + str(p), (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    
    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

    
