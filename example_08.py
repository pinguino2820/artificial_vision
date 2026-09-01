# ======================================
# Ultralytics YOLO + Tracking + Forecast
#
# Computer Vision With Ultralytics::
# https://www.youtube.com/playlist?list=PL1FZnkj4ad1OEHaqPKFKJFVDfQ2gSJoc4
# ======================================

import cv2
import torch
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# ------
# Config
# ------
CONF = 0.4
TRACKER = "bytetrack.yaml"       # botsort.yaml or bytetrack.yaml
CLASSES = [0, 2, 3, 5, 7]        # Detect specific classes, i.e person, car etc.
HISTORY = 60                     # store last N centers per track
MIN_POINTS = 6                   # need at least this many points to forecast
FORECAST_STEPS = 35              # how many future points
VEL_WINDOW = 8                   # how many recent points to estimate velocity
EMA_ALPHA = 0.6                  # smoothing for centers (0..1). higher = follow new more   
FORECAST_COLOR = (108, 27, 255)
FONT_SCALE = 1.2
FONT_THICKNESS = 4
PADDING = 8


# ---------------
# Drawing helpers
# ---------------
def draw_polyline(frame, pts, color, thickness=2):
    if len(pts) < 2:
        return
    p = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [p], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

def draw_forecast(frame, pts, color):
    if len(pts) < 2:
        return
    draw_polyline(frame, pts, color, thickness=1)
    for (x, y) in pts[::max(1, len(pts)//5)]:
        cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)

def clamp_points(pts, w, h):
    out = []
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            out.append((int(x), int(y)))
    return out
    

# -----------
# Forecasting
# -----------
def estimate_velocity(pts, fps, window=8):
    n = len(pts)
    if n < 2:
        return 0.0, 0.0
    k = min(window, n - 1)
    vxs, vys, dt = [], [], 1.0 / fps
    for i in range(n - k, n):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        vxs.append((x2 - x1) / dt)
        vys.append((y2 - y1) / dt)
    return float(np.median(vxs)), float(np.median(vys))

def forecast_points(last_xy, vx, vy, fps, steps):
    dt = 1.0 / fps
    lx, ly = float(last_xy[0]), float(last_xy[1])
    return [(lx + vx * dt * s, ly + vy * dt * s) for s in range(1, steps + 1)]

# ---------
# Inference
# ---------
def inference(model_path, source, output_path, display_track_id_only=True):
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")


    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 1e-6 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    history = defaultdict(lambda: deque(maxlen=HISTORY))  # track_id -> deque[(x,y)]
    last_smooth = {}                                     # track_id -> (x,y)

    cv2.namedWindow("Tracking + Forecast | q to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking + Forecast | q to quit", 1060, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=CONF, classes=CLASSES, tracker=TRACKER)

        active_ids = set()
        r0 = results[0]

        if r0.boxes is not None and r0.boxes.id is not None:
            boxes = r0.boxes.xyxy.cpu().numpy()
            ids = r0.boxes.id.cpu().numpy().astype(int)
            clss = r0.boxes.cls.cpu().numpy().astype(int)

            for bbox, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                active_ids.add(tid)

                # --------------------
                # EMA smoothing center
                # --------------------
                if tid in last_smooth:
                    lx, ly = last_smooth[tid]
                    sx, sy = EMA_ALPHA * cx + (1.0 - EMA_ALPHA) * lx, EMA_ALPHA * cy + (1.0 - EMA_ALPHA) * ly
                else:
                    sx, sy = cx, cy
                last_smooth[tid] = (sx, sy)
                history[tid].append((sx, sy))
                # --------------------

                # ------------
                # Bounding box
                # ------------
                bbox_color = colors(0 if cls==2 else cls, True)
                label = f"#{tid} {model.names[cls]}" if not display_track_id_only else f"{tid}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2, cv2.LINE_AA)
                x, y = int(x1), int(y1)
                (tw, th) = cv2.getTextSize(label, 0, FONT_SCALE, FONT_THICKNESS)[0]
                rect_w, rect_h = tw + 2 * PADDING, th + 2 * PADDING
                x2, y2 = x + rect_w, y + rect_h
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, -1)
                text_x, text_y = x1 + (rect_w - tw) // 2, y1 + (rect_h + th) // 2 
                cv2.putText(frame, label, (text_x, text_y), 0, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
                # ------------

                # -------------                
                # Past polyline
                # -------------
                past_pts = clamp_points(list(history[tid]), width, height)
                draw_polyline(frame, past_pts, bbox_color, thickness=2)
                # -------------

                # -----------------               
                # Forecast polyline
                # -----------------
                if len(history[tid]) >= MIN_POINTS:
                    vx, vy = estimate_velocity(list(history[tid]), fps, window=VEL_WINDOW)
                    if np.hypot(vx, vy) > 1.0:  # stationary gate
                        fpts = forecast_points(history[tid][-1], vx, vy, fps, FORECAST_STEPS)
                        fpts = clamp_points(fpts, width, height)
                        draw_forecast(frame, fpts, FORECAST_COLOR)
                # -----------------

        
        # --------------------------
        # Cleanup disappeared tracks
        # --------------------------
        for tid in list(history.keys()):
            if tid not in active_ids:
                history.pop(tid, None)
                last_smooth.pop(tid, None)
        # --------------------------

        cv2.imshow("Tracking + Forecast | q to quit", frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference(
        model_path="yolo26s.pt",
        source="videos/cars-on-highway.mp4",
        output_path="results-track-forecast.mp4",
        display_track_id_only=True,
    )