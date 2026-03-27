from collections import defaultdict
import cv2
import numpy as np
import csv
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/yolo26n/full_train/final/weights/best.pt"
VIDEO_PATH = "D:/Videos/20240808_exp1_goprodata_full.mp4"
BOTSORT_FILE = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/bostort_files/test.yaml"

OUTPUT_VIDEO = f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/20240808_exp1_23-25/tracks_and_bb.mp4"
CSV_OUTPUT = f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/20240808_exp1_23-25/uc_tracking_data.csv"

# -----------------------------
# TOGGLES
# -----------------------------
DISPLAY_VIDEO = True
SAVE_VIDEO = True
SAVE_CSV = True

# -----------------------------
# Performance Controls
# -----------------------------
TRACE_LENGTH = 300          # max points per track
REMOVE_STALE_TRACKS = False  # remove tracks not in current frame
DRAW_TRAILS = True

# -----------------------------
# Frame Controls
# -----------------------------
start_frame = 24*60*23
max_frames_processed = 24*60*2

# -----------------------------
# Load Model
# -----------------------------
model = YOLO(MODEL_PATH)

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
# Video Writer (optional)
# -----------------------------
out = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# -----------------------------
# CSV Writer (optional)
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
# Drawing Functions
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

    results = model.track(
        frame,
        tracker=BOTSORT_FILE,
        persist=True,
        verbose=False,
        imgsz=3008   # reduced for performance
    )[0]

    current_ids = []

    if results.boxes and results.boxes.is_track:

        boxes_xywh = results.boxes.xywh.cpu()
        boxes_xyxy = results.boxes.xyxy.cpu()
        track_ids = results.boxes.id.int().cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()
        classes = results.boxes.cls.int().cpu().tolist()

        current_ids = track_ids

        for box, box_xyxy, track_id, conf, cls in zip(
            boxes_xywh, boxes_xyxy, track_ids, confs, classes
        ):
            x, y, w, h = box
            x1, y1, x2, y2 = map(int, box_xyxy)

            # Clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Update track history
            track_history[track_id].append((int(x), int(y)))
            if len(track_history[track_id]) > TRACE_LENGTH:
                track_history[track_id].pop(0)

            scaling_factor = 2.5
            tp_x = x * scaling_factor
            tp_y = (y * scaling_factor * -1) + 2000

            # Save CSV
            if SAVE_CSV:
                writer.writerow([
                    track_id,
                    frame_id,
                    float(tp_x),
                    float(tp_y),
                    float(w*scaling_factor),
                    float(h*scaling_factor),
                    float(conf),
                    int(cls),
                    model.names[int(cls)],
                    None
                ])

            # Draw overlays
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

    if frame_id > start_frame + max_frames_processed:
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