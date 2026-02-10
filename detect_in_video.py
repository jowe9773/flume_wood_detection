import cv2
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------

MODEL_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/0-24_annotations_three_classes/test/weights/best.pt"
VIDEO_PATH = "D:/Videos/20240708_exp1_goprodata_full.mp4"
CONF_THRESH = 0.25

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Could not open video: {VIDEO_PATH}")

cap.set(cv2.CAP_PROP_POS_FRAMES, 6000) 

# Optional: get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


print(f"Video: {width}x{height} @ {fps:.1f} FPS")

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (width, height))
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)


    # YOLO inference (single frame)
    results = model(
        frame,
        conf=CONF_THRESH,
        verbose=False
    )

    # results[0] corresponds to this frame
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLO Detection", annotated_frame)

    # Quit on 'q' or ESC
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
