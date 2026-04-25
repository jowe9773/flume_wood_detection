import cv2
import os
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/models/yolov26n_long_train_uc_only/weights/best.pt"
VIDEO_PATH = "D:/Videos/20240617_exp1_goprodata_full.mp4"

minutes = 4
seconds = 21
FPS_ASSUMED = 24

CONF_THRESH = 0.1
START_FRAME = int(FPS_ASSUMED * 60 * minutes + FPS_ASSUMED * seconds)

MAX_FRAMES = START_FRAME + FPS_ASSUMED * 60 * 5

# Resize display (set max width so it fits on screen)
DISPLAY_WIDTH = 1800

SAVE_VIDEO = True
OUTPUT_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/detections/2_0/output.mp4"

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)
# model.to("cuda")  # uncomment if using GPU


# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or FPS_ASSUMED

# Jump once
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
cur_frame = START_FRAME

# -----------------------------
# Video writer (init later)
# -----------------------------
out = None

if SAVE_VIDEO:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)


# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret or cur_frame >= MAX_FRAMES:
        break

    # YOLO inference
    results = model.predict(
        frame,
        conf=CONF_THRESH,
        verbose=False
    )

    annotated = results[0].plot()

    # Resize for display
    h, w = annotated.shape[:2]
    scale = DISPLAY_WIDTH / w
    display = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    cv2.imshow("YOLO", display)

    # -----------------------------
    # Initialize writer ONCE (after we know frame size)
    # -----------------------------
    if SAVE_VIDEO and out is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            OUTPUT_PATH,
            fourcc,
            fps,
            (w, h)  # original resolution (not resized)
        )

    # Write full-resolution annotated frame
    if SAVE_VIDEO:
        out.write(annotated)

    # Minimal wait (fast playback)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    cur_frame += 1


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()