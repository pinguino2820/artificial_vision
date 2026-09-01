# import required libraries
import cv2
from ultralytics import YOLO
import cvzone

# load YOLO11 model
model = YOLO("yolo11n.pt")
names = model.names

# define vertical line's X position
line_x = 500

# track previos center positions
track_history = {}

# IN/OUT counters
in_count = 0
out_count = 0

# open video file or webcam
cap = cv2.VideoCapture('videos/video_03.mp4')     # use 0 for webcam

# define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f'Mouse moved to: [{x}, {y}]') 

# show circle, rectangle and track_id        
def showInfo():
    cv2.circle(
        frame, 
        (cx, cy), 
        4, 
        (255, 0, 0), 
        -1
    )
    cv2.rectangle(
        frame, 
        (x1, y1), 
        (x2, y2), 
        (0, 255, 0), 
        2
    )
    cvzone.putTextRect(
        frame, 
        f'{track_id}', 
        (x1, y1), 
        1, 
        1
    )
            
# create a named OpenCV window and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
frame_count = 0
while True:
    # read video frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 600))
    
    # detect and track persons class(0)
    results = model.track(frame, persist=True, classes=[0])
    
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            # center rectangle coordinates
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            if track_id in track_history:
                prev_cx, prev_cy = track_history[track_id]
                # left to right
                if (prev_cx < line_x <= cx):
                    in_count += 1                    
                    showInfo()
                # right to left
                if (prev_cx > line_x >= cx):
                    out_count += 1                    
                    showInfo()
    
            track_history[track_id] = (cx, cy) 
            
    # display counts using cvzone's putTextRect
    cvzone.putTextRect(
        frame, 
        f'To right: {in_count}', 
        (750, 60), 
        scale=2, 
        thickness=1, 
        colorT=(255, 255, 255), 
        colorR=(0, 128, 0)
    )
    cvzone.putTextRect(
        frame, 
        f'To left  : {out_count}', 
        (750, 120), 
        scale=2, 
        thickness=1, 
        colorT=(255, 255, 255), 
        colorR=(0, 0, 255)
    )
    cv2.line(
        frame, 
        (line_x, 0), 
        (line_x, frame.shape[0]), # frame.shape[0] is the height, and [1] is the width of video screen
        (255, 255, 255),
        2
    )
    
    # show the frame
    cv2.imshow('RGB', frame)
    
    # press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

    

