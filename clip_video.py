import cv2

# -----------------------------
# Config (EDIT THESE)
# -----------------------------
date = 20240529
INPUT_VIDEO = f"D:/Videos/{date}_exp2_goprodata_full.mp4"
OUTPUT_VIDEO = f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/{date}_exp2_goprodata_short.mp4"

START_SEC = 45.0   # start time in seconds
END_SEC = 60 + START_SEC     # end time in seconds
# -----------------------------

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Convert seconds to frame numbers
start_frame = int(START_SEC * fps)
end_frame = int(END_SEC * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_idx = start_frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx > end_frame:
        break

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Saved clip from {START_SEC}s to {END_SEC}s → {OUTPUT_VIDEO}")
