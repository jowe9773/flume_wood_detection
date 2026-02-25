from collections import defaultdict
import cv2
import numpy as np
import csv
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "C:/Users/josie/local_data/YOLO/models/yolo26n/fold_1/weights/best.pt"
VIDEO_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/20240529_exp2_goprodata_short.mp4"

save = True
OUTPUT_VIDEO = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/20240529_exp2_goprodata_short_with_tracks_DIAGONAL.mp4"
tracking_data_output_fn = "tracking_data.csv"

start_frame = 0

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

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# -----------------------------
# Track Memory
# -----------------------------
track_history = defaultdict(list)
track_colors = {}
track_orientation_state = {}

# -----------------------------
# Class Colors (BGR)
# -----------------------------
CLASS_COLOR_MAP = {
    0: (180, 0, 180),   # purple
    1: (0, 140, 255),   # red-orange
    2: (0, 200, 0)      # green
}

# ------------------------------
# Track Colors
# ------------------------------
def get_track_color(track_id):
    if track_id not in track_colors:
        # Generate bright distinct color
        track_colors[track_id] = (
            int(np.random.randint(80, 255)),  # B
            int(np.random.randint(80, 255)),  # G
            int(np.random.randint(80, 255)),  # R
        )
    return track_colors[track_id]

def get_class_color(cls_id):
    return CLASS_COLOR_MAP.get(cls_id, (255, 255, 255))

# -----------------------------
# Orientation Function
# -----------------------------
def compute_orientation_exact_diagonal(crop, w, h, track_id):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    ys, xs = np.where(edges > 0)
    if len(xs) < 20:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)

    vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    line_vec = np.array([vx, vy]).flatten()
    line_vec /= np.linalg.norm(line_vec)

    # Exact bounding-box diagonals
    d1 = np.array([w, h], dtype=np.float32)
    d2 = np.array([w, -h], dtype=np.float32)

    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)

    score1 = abs(np.dot(line_vec, d1))
    score2 = abs(np.dot(line_vec, d2))

    winner = 0 if score1 > score2 else 1

    # Initialize track state
    if track_id not in track_orientation_state:
        track_orientation_state[track_id] = {
            "current_diag": winner,
            "switch_counter": 0
        }

    state = track_orientation_state[track_id]
    current = state["current_diag"]

    # Exact diagonal angles
    angle_d1 = np.degrees(np.arctan2(h, w)) % 180
    angle_d2 = np.degrees(np.arctan2(-h, w)) % 180
    exact_angles = [angle_d1, angle_d2]

    current_angle = exact_angles[current]

    # Unstable zone (near 0, 90, 180)
    unstable = (
        abs(current_angle - 0) <2.5 or
        abs(current_angle - 90) < 2.5 or
        abs(current_angle - 180) < 2.5
    )

    if winner != current:
        if unstable:
            state["current_diag"] = winner
            state["switch_counter"] = 0
        else:
            state["switch_counter"] += 1
            if state["switch_counter"] >= 3:
                state["current_diag"] = winner
                state["switch_counter"] = 0
    else:
        state["switch_counter"] = 0

    return float(exact_angles[state["current_diag"]])

# -----------------------------
# CSV Output
# -----------------------------
csv_file = open(tracking_data_output_fn, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["track_id", "frame", "center_x", "center_y",
                 "width", "height", "confidence",
                 "class_id", "class_name", "orientation"])

frame_id = 0

# -----------------------------
# Main Loop
# -----------------------------
while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    results = model.track(
        frame,
        tracker="test_botsort.yaml",
        persist=True,
        verbose=False
    )[0]

    if results.boxes and results.boxes.is_track:

        boxes_xywh = results.boxes.xywh.cpu()
        boxes_xyxy = results.boxes.xyxy.cpu()
        track_ids = results.boxes.id.int().cpu().tolist()
        confs = results.boxes.conf.cpu().tolist()
        classes = results.boxes.cls.int().cpu().tolist()

        for box, box_xyxy, track_id, conf, cls in zip(
            boxes_xywh, boxes_xyxy, track_ids, confs, classes
        ):

            x, y, w, h = box
            x1, y1, x2, y2 = map(int, box_xyxy)

            # Clamp to frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            crop = frame[y1:y2, x1:x2]

            orientation = None
            if crop.size > 0:
                orientation = compute_orientation_exact_diagonal(
                    crop, float(w), float(h), track_id
                )

            # Save CSV
            writer.writerow([
                track_id,
                frame_id,
                float(x),
                float(y),
                float(w),
                float(h),
                float(conf),
                int(cls),
                model.names[int(cls)],
                orientation
            ])

            #append location of bb center to the track history
            track_history[track_id].append((int(x), int(y)))

            # shorten track history
            if len(track_history[track_id]) > 50:
                track_history[track_id].pop(0)

            # Draw bounding box
            #cv2.rectangle(frame, (x1, y1), (x2, y2), track_colors[track_id], 3)
            
            color = get_class_color(int(cls))
            # Draw orientation line
            if orientation is not None:
                cx, cy = int(x), int(y)
                line_length = int(max(w, h))
                theta = np.deg2rad(orientation)

                dx = np.cos(theta) * line_length / 2
                dy = np.sin(theta) * line_length / 2

                pt1 = (int(cx - dx), int(cy - dy))
                pt2 = (int(cx + dx), int(cy + dy))

                cv2.line(frame, pt1, pt2,
                         color, 4)
            
            # ---- DRAW TRACK POLYLINES ----
            for tid, points in track_history.items():
                if len(points) > 1:
                    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                    color = get_track_color(tid)

                    cv2.polylines(
                        frame,
                        [pts],
                        isClosed=False,
                        color=color,
                        thickness=4
                    )

    # Resize for display
    scale = 0.4
    display_frame = cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("YOLO Diagonal Tracking", display_frame)

    if save:
        out.write(frame)

    frame_id += 1

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
out.release()
cap.release()
csv_file.close()
cv2.destroyAllWindows()