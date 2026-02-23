from collections import defaultdict

import cv2
import numpy as np
import csv

from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------

MODEL_PATH = "C:/Users/josie/local_data/YOLO/models/first_test_new_data_format/first_test_new_data_format/weights/best.pt"
VIDEO_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/20240529_exp2_goprodata_short.mp4"

save = True
OUTPUT_VIDEO = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/20240529_exp2_goprodata_short_with_tracks4.mp4"


start_frame = 0
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Assign a fixed color per class
# BGR format!
CLASS_COLOR_MAP = {
    0: (180, 0, 180),   #int # purple 
    1: (0, 140, 255),   #long # red-orange
    2: (0, 200, 0)      #short # green 
}

def get_class_color(cls_id):
    return CLASS_COLOR_MAP.get(cls_id, (255, 255, 255))  # fallback white

# Store the track history
track_history = defaultdict(lambda: [])

# Store all tracks that have ever existed (even if killed)
all_tracks = defaultdict(lambda: [])

# Assign a persistent random color to each track ID
track_colors = {}
track_orientations = {}

def dim_color(color, factor=0.9):
    """Darken a BGR color by a factor (0-1)."""
    return tuple(int(c * factor) for c in color)

def compute_orientation(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Threshold (adjust if needed)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Get coordinates of white pixels
    ys, xs = np.where(thresh > 0)

    if len(xs) < 10:
        return None  # Not enough pixels

    coords = np.column_stack((xs, ys)).astype(np.float32)

    # PCA
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)

    # First eigenvector = major axis direction
    vx, vy = eigenvectors[0]

    angle = np.arctan2(vy, vx)
    angle_deg = np.degrees(angle)

    return angle_deg

def compute_orientation_fitline(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Blur slightly to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edges instead of raw threshold
    edges = cv2.Canny(gray, 50, 150)

    # Get edge pixel coordinates
    ys, xs = np.where(edges > 0)

    if len(xs) < 20:
        return None  # Not enough edge points

    points = np.column_stack((xs, ys)).astype(np.float32)

    # Fit a line through the points
    # Returns: vx, vy (direction vector), x0, y0 (a point on the line)
    [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    # Convert direction vector to angle
    angle = np.arctan2(vy, vx)
    angle_deg = np.degrees(angle)

    # Normalize to 0–180°
    if angle_deg < 0:
        angle_deg += 180
    angle_deg = angle_deg % 180

    return float(angle_deg)

def compute_orientation_constrained(crop, w, h, prev_angle=None, smoothing=0.8):
    """
    Computes orientation constrained to the two bounding-box diagonals.

    Args:
        crop: image crop (BGR)
        w, h: bounding box width and height
        prev_angle: previous frame angle for this track (optional)
        smoothing: 0-1 smoothing factor (higher = smoother)

    Returns:
        angle in degrees (0–180) or None
    """

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    ys, xs = np.where(edges > 0)
    if len(xs) < 20:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)

    # Fit line to edge pixels
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    line_vec = np.array([vx, vy]).flatten()
    line_vec /= np.linalg.norm(line_vec)

    # ---- Compute the two valid diagonals ----
    d1 = np.array([w, h], dtype=np.float32)
    d2 = np.array([w, -h], dtype=np.float32)

    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)

    # ---- Compare alignment using dot product ----
    score1 = abs(np.dot(line_vec, d1))
    score2 = abs(np.dot(line_vec, d2))

    if score1 > score2:
        chosen = d1
    else:
        chosen = d2

    angle = np.degrees(np.arctan2(chosen[1], chosen[0]))

    if angle < 0:
        angle += 180
    angle = angle % 180

    # ---- Optional temporal smoothing ----
    if prev_angle is not None:
        angle = smoothing * prev_angle + (1 - smoothing) * angle

    return float(angle)

#save data of interest in each frame
csv_file = open("tracking_data.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["track_id", "frame", "center_x", "center_y", "width", "height", "confidence", "class_id", "class_name", "orientation"])

frame_id = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
     
    if success:
        # Run YOLO26 tracking on the frame, persisting tracks between frames
        result = model.track(frame, 
                             tracker= "test_botsort.yaml", persist=True, verbose = False)[0]

        if result.boxes and result.boxes.is_track:
            boxes_xywh = result.boxes.xywh.cpu()
            boxes_xyxy = result.boxes.xyxy.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()

            active_ids = set(track_ids)

            for box, box_xyxy, track_id, conf, cls in zip(boxes_xywh, boxes_xyxy, track_ids, confs, classes):
                x, y, w, h = box
                x1, y1, x2, y2 = box_xyxy


                # Convert to ints for cropping
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Clamp to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]

                orientation = None
                prev_angle = track_orientations.get(track_id, None)

                orientation = compute_orientation_constrained(crop, w, h, prev_angle)

                if orientation is not None:
                    track_orientations[track_id] = orientation

                x_out = float(x * 2.5)
                y_out = -1*float(y * 2.5 - 2000)
                w_out = float(w * 2.5)
                h_out = float(h * 2.5)

                class_name = model.names[cls]



                writer.writerow([
                    track_id,
                    frame_id,
                    x_out,
                    y_out,
                    w_out,
                    h_out,
                    float(conf),
                    int(cls),
                    class_name,
                    orientation
                ])

                print(track_id, frame_id, x_out, y_out, w_out, h_out, float(conf), int(cls), class_name, orientation)


                # Assign a random color if we haven't seen this ID before
                if track_id not in track_colors:
                    track_colors[track_id] = (
                        int(np.random.randint(50, 255)),  # B
                        int(np.random.randint(50, 255)),  # G
                        int(np.random.randint(50, 255)),  # R
                    )

                # Update histories
                track_history[track_id].append((float(x), float(y)))
                all_tracks[track_id].append((float(x), float(y)))

            #if you want to plot the bounding box
            frame = result.plot(labels=False, conf=False, line_width=2)

            # ---------------- DRAW ORIENTATION LINE ----------------
            if orientation is not None:
                cx, cy = int(x), int(y)

                # choose line length (good starting rule)
                line_length = int(max(w, h))

                theta = np.deg2rad(orientation)

                dx = np.cos(theta) * line_length / 2
                dy = np.sin(theta) * line_length / 2

                pt1 = (int(cx - dx), int(cy - dy))
                pt2 = (int(cx + dx), int(cy + dy))

                color = get_class_color(int(cls))
                cv2.line(frame, pt1, pt2, color, thickness=4)
                    

            # ---- DRAW ALL TRACKS (ACTIVE = BRIGHT, DEAD = DIM) ----
            for track_id, track in all_tracks.items():
                if len(track) > 1:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    color = track_colors[track_id]

                    # If this track is NOT active in this frame → make it dimmer
                    if track_id not in active_ids:
                        draw_color = dim_color(color, factor=0.5)  # <<< tweak this
                        thickness = 4
                    else:
                        draw_color = color
                        thickness = 8

                    """cv2.polylines(
                        frame,
                        [points],
                        isClosed=False,
                        color=draw_color,
                        thickness=thickness,
                    )"""


        scale = 0.4
        h,w = frame.shape[:2]
        new_w = int(w*scale)
        new_h = int(h*scale)

        display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("YOLO ByteTrack", display_frame)

        frame_id += 1

        if save == True:
            out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
out.release()
cap.release()
cv2.destroyAllWindows()