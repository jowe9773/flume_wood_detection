from collections import defaultdict
import cv2
import numpy as np
import csv
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
#define models used
MODEL_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/models/yolov26n_long_train_uc_only/weights/best.pt"
BOTSORT_FILE = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/models/bytetrack_tuned.yaml"

#input data
VIDEO_PATH = "D:/Videos/20240606_exp1_goprodata_full.mp4"

#output data
OUTPUT_VIDEO = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/bytetrack/raw_tracking_outputs/0_25/first5.mp4"
CSV_OUTPUT = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/bytetrack/raw_tracking_outputs/0_25/first5_tracking_data.csv"

# -----------------------------
# TOGGLES
# -----------------------------
DISPLAY_VIDEO = True
SAVE_VIDEO = True
SAVE_CSV = True

# -----------------------------
# Performance Controls
# -----------------------------
TRACE_LENGTH = 300
REMOVE_STALE_TRACKS = False
DRAW_TRAILS = True

# -----------------------------
# Frame Controls
# -----------------------------

minutes = 5
seconds = 55

start_frame = (24 * 60 * minutes) + (24 * seconds)
max_frames_processed = 24 * 60 * 5


# -----------------------------
# Classes (IMPORTANT)
# -----------------------------
CLASS_IDS = [0, 1, 2]  # short, int, long

# -----------------------------
# Load Models (one per class)
# -----------------------------
trackers = {
    cls_id: YOLO(MODEL_PATH)
    for cls_id in CLASS_IDS
}

# -----------------------------
# Video Setup
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
end_frame = min(start_frame + max_frames_processed, total_frames)

pbar = tqdm(total=(end_frame - start_frame), desc="Processing Video")

# -----------------------------
# Video Writer
# -----------------------------
out = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# -----------------------------
# CSV Writer
# -----------------------------
csv_file = None
writer = None
if SAVE_CSV:
    csv_file = open(CSV_OUTPUT, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "track_id", "frame", "center_x", "center_y",
        "width", "height", "confidence",
        "class_id", "class_name", "orientation"
    ])

# -----------------------------
# Track Memory
# -----------------------------
track_history = defaultdict(list)
track_colors = {}

# -----------------------------
# Colors
# -----------------------------
CLASS_COLOR_MAP = {
    0: (180, 0, 180),
    1: (0, 140, 255),
    2: (0, 200, 0)
}

def get_track_color(track_id):
    if track_id not in track_colors:
        track_colors[track_id] = (
            int(np.random.randint(80, 255)),
            int(np.random.randint(80, 255)),
            int(np.random.randint(80, 255)),
        )
    return track_colors[track_id]

def get_class_color(cls_id):
    return CLASS_COLOR_MAP.get(cls_id, (255, 255, 255))

# -----------------------------
# Drawing
# -----------------------------
def draw_bounding_box(frame, x1, y1, x2, y2, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def draw_track_id(frame, track_id, x1, y1, color):
    label = f"ID {track_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 4),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_traces(frame):
    for tid, points in track_history.items():
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame,
                [pts],
                isClosed=False,
                color=get_track_color(tid),
                thickness=3
            )

# -----------------------------
# Main Loop
# -----------------------------
frame_id = start_frame

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    all_detections = []
    current_ids = []

    # -----------------------------
    # Run one tracker per class
    # -----------------------------
    for cls_id, tracker_model in trackers.items():

        results = tracker_model.track(
            frame,
            tracker=BOTSORT_FILE,
            persist=True,
            verbose=False,
            imgsz=3008,
            classes=[cls_id]
        )[0]

        if results.boxes and results.boxes.is_track:

            boxes_xywh = results.boxes.xywh.cpu()
            boxes_xyxy = results.boxes.xyxy.cpu()
            track_ids = results.boxes.id.int().cpu().tolist()
            confs = results.boxes.conf.cpu().tolist()
            classes = results.boxes.cls.int().cpu().tolist()

            # Offset IDs to prevent collisions
            track_ids = [tid + cls_id * 100000 for tid in track_ids]

            for box, box_xyxy, track_id, conf, cls in zip(
                boxes_xywh, boxes_xyxy, track_ids, confs, classes
            ):
                all_detections.append((box, box_xyxy, track_id, conf, cls))

    # -----------------------------
    # Process detections
    # -----------------------------
    for box, box_xyxy, track_id, conf, cls in all_detections:

        x, y, w, h = box
        x1, y1, x2, y2 = map(int, box_xyxy)

        current_ids.append(track_id)

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Track history
        track_history[track_id].append((int(x), int(y)))
        if len(track_history[track_id]) > TRACE_LENGTH:
            track_history[track_id].pop(0)

        scaling_factor = 2.5
        tp_x = x * scaling_factor
        tp_y = (y * scaling_factor * -1) + 2000

        # CSV
        if SAVE_CSV:
            writer.writerow([
                track_id,
                frame_id,
                float(tp_x),
                float(tp_y),
                float(w * scaling_factor),
                float(h * scaling_factor),
                float(conf),
                int(cls),
                tracker_model.names[int(cls)],
                None
            ])

        # Draw
        color = get_class_color(int(cls))
        draw_bounding_box(frame, x1, y1, x2, y2, color)
        draw_track_id(frame, track_id, x1, y1, color)

    # -----------------------------
    # Remove stale tracks
    # -----------------------------
    if REMOVE_STALE_TRACKS:
        active = set(current_ids)
        for tid in list(track_history.keys()):
            if tid not in active:
                del track_history[tid]

    # -----------------------------
    # Draw trails
    # -----------------------------
    if DRAW_TRAILS:
        draw_traces(frame)

    # -----------------------------
    # Display
    # -----------------------------
    if DISPLAY_VIDEO:
        scale = 0.5
        display_frame = cv2.resize(frame, (
            int(width * scale), int(height * scale)
        ))
        cv2.imshow("Tracking", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -----------------------------
    # Save Video
    # -----------------------------
    if SAVE_VIDEO and out is not None:
        out.write(frame)

    frame_id += 1
    pbar.update(1)

    if frame_id > end_frame:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
pbar.close()

if SAVE_VIDEO and out is not None:
    out.release()

if SAVE_CSV and csv_file is not None:
    csv_file.close()

cv2.destroyAllWindows()