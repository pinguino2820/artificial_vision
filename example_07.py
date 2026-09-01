"""
example_07.py - face recognition

https://www.youtube.com/watch?v=o3EXBpMGdxM&list=PLh6FA5h81jplWXGXOlhAb4POMQOvVcQlT&index=2
"""

import cv2
import numpy as np
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent / "models" / "age_gender"
face_net = cv2.dnn.readNet(
    str(MODEL_DIR / "opencv_face_detector_uint8.pb"),
    str(MODEL_DIR / "opencv_face_detector.pbtxt"),
)
gender_net = cv2.dnn.readNet(
    str(MODEL_DIR / "gender_net.caffemodel"),
    str(MODEL_DIR / "gender_deploy.prototxt"),
)
GENDER_LABELS = ("male", "female")

cap = cv2.VideoCapture('videos/people2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))
    height, width = frame.shape[:2]
    face_blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104, 117, 123), swapRB=True,
    )
    face_net.setInput(face_blob)
    detections = face_net.forward()

    for index in range(detections.shape[2]):
        confidence = detections[0, 0, index, 2]
        if confidence < 0.7:
            continue

        x1 = max(0, int(detections[0, 0, index, 3] * width))
        y1 = max(0, int(detections[0, 0, index, 4] * height))
        x2 = min(width, int(detections[0, 0, index, 5] * width))
        y2 = min(height, int(detections[0, 0, index, 6] * height))
        crop = frame[y1:y2, x1:x2]
        if not crop.size:
            continue

        gender_blob = cv2.dnn.blobFromImage(
            crop,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
        )
        gender_net.setInput(gender_blob)
        label = GENDER_LABELS[np.argmax(gender_net.forward()[0])]
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        label_color = (255, 0, 255)
        padding = 5
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            1,
        )
        label_width = text_width + (padding * 2)
        label_height = text_height + baseline + (padding * 2)
        label_x1 = min(x1, width - label_width)
        label_y1 = max(0, y1 - label_height)
        label_x2 = label_x1 + label_width
        label_y2 = label_y1 + label_height
        cv2.rectangle(
            frame,
            (label_x1, label_y1),
            (label_x2, label_y2),
            label_color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x1 + padding, label_y2 - padding - baseline),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
        )
         
    cv2.imshow('FRAME', frame)
    
    if cv2.waitKey(1)&0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()