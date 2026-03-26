from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/yolo26n/full_train/final/weights/best.pt"

VIDEO_PATH = "D:/Videos/20240808_exp1_goprodata_full.mp4"
BOTSORT_FILE = f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/bostort_files/test.yaml"

CONF_THRESHOLD = 0.05       # threshold for displaying all detections (untracked)
MAX_FRAMES_PROCESSED = 24*60*2
TRACK_HISTORY_LENGTH = 48  # how long to draw past track points

start_frame = 24*60*3.1

# -----------------------------
# Class Colors (BGR)
# -----------------------------
CLASS_COLOR_MAP = {
    0: (180, 0, 180),   # purple
    1: (0, 140, 255),   # red-orange
    2: (0, 200, 0)      # green
}

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Video Setup
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("tracked_output_classcolor.mp4", fourcc, fps, (width, height))

# -----------------------------
# Track Memory
# -----------------------------
track_history = defaultdict(list)
track_colors = {}

def get_track_color(track_id):
    if track_id not in track_colors:
        track_colors[track_id] = tuple(np.random.randint(80, 255, 3).tolist())
    return track_colors[track_id]

def get_class_color(cls_id):
    return CLASS_COLOR_MAP.get(cls_id, (255, 255, 255))  # white if unknown

# -----------------------------
# Main Loop
# -----------------------------
frame_id = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success or frame_id > MAX_FRAMES_PROCESSED:
        break

    # Run YOLO + BoT-SORT
    results = model.track(
        frame,
        tracker=BOTSORT_FILE,
        persist=True,
        verbose=False,
    )[0]

    # --- Draw all detections above CONF_THRESHOLD (class-colored) ---
    if results.boxes:
        for box, conf, cls in zip(
            results.boxes.xyxy.cpu(),
            results.boxes.conf.cpu(),
            results.boxes.cls.cpu()
        ):
            if conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                color = get_class_color(int(cls))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw confidence as text (rounded to 2 decimals)
                conf_text = f"{conf:.2f}"
                font_scale = 3
                thickness = 3
                text_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_w, text_h = text_size

                # Background rectangle for text for readability
                cv2.rectangle(frame, (x1, y1 - text_h - 2), (x1 + text_w, y1), color, -1)
                cv2.putText(frame, conf_text, (x1, y1 - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # --- Draw tracks with colored lines ---
    if results.boxes and results.boxes.is_track:
        track_ids = results.boxes.id.int().cpu().tolist()
        boxes_xywh = results.boxes.xywh.cpu()
        for track_id, box in zip(track_ids, boxes_xywh):
            x, y, w, h = box
            cx, cy = int(x), int(y)
            track_history[track_id].append((cx, cy))

            # Keep only recent points
            if len(track_history[track_id]) > TRACK_HISTORY_LENGTH:
                track_history[track_id].pop(0)

            # Draw track line
            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            color = get_track_color(track_id)
            if len(points) > 1:
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=3)

    # Display / Save
    cv2.imshow("Tracking", cv2.resize(frame, (int(width*0.5), int(height*0.5))))
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()