from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------

MODEL_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/0-24_annotations_one_class/yolo11m/weights/best.pt"
VIDEO_PATH = "D:/Videos/20240529_exp2_goprodata_full.mp4"
start_frame = 1000
model = YOLO(MODEL_PATH)

# Load the YOLO26 model
model = YOLO(MODEL_PATH)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO26 tracking on the frame, persisting tracks between frames
        result = model.track(frame, 
                             tracker= "test_botsort.yaml", persist=True)[0]

        # Get the boxes and track IDs
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            # Visualize the result on the frame
            frame = result.plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                """if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)"""

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        scale = 0.5
        h,w = frame.shape[:2]
        new_w = int(w*scale)
        new_h = int(h*scale)

        display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("YOLO ByteTrack", display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()