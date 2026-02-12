from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------

MODEL_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/0-24_annotations_three_classes_vconcat/yolo26n2/weights/best.pt"
VIDEO_PATH = f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/20240529_exp2_goprodata_short.mp4"

save = True
OUTPUT_VIDEO = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/20240529_exp2_goprodata_short_with_tracks2.mp4"


start_frame = 0
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


# Store the track history
track_history = defaultdict(lambda: [])

# Store all tracks that have ever existed (even if killed)
all_tracks = defaultdict(lambda: [])

# Assign a persistent random color to each track ID
track_colors = {}

def dim_color(color, factor=0.9):
    """Darken a BGR color by a factor (0-1)."""
    return tuple(int(c * factor) for c in color)

import numpy as np
import cv2

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


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO26 tracking on the frame, persisting tracks between frames
        result = model.track(frame, 
                             tracker= "test_botsort.yaml", persist=True, verbose = False)[0]

        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            frame = result.plot(labels=False, conf=False)

            active_ids = set(track_ids)

            # ---- UPDATE TRACKS ----
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box

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

                    cv2.polylines(
                        frame,
                        [points],
                        isClosed=False,
                        color=draw_color,
                        thickness=thickness,
                    )


        scale = 0.5
        h,w = frame.shape[:2]
        new_w = int(w*scale)
        new_h = int(h*scale)

        display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("YOLO ByteTrack", display_frame)

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