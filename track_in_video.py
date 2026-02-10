import cv2
import random
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------

MODEL_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/0-24_annotations_one_class/yolo11m/weights/best.pt"
VIDEO_PATH = "D:/Videos/20240808_exp1_goprodata_full.mp4"

model = YOLO(MODEL_PATH)


def bytetrack(path, target_classes=None, start_frame=0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Jump to desired starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    track_id_colors = {}

    frame_id = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Run tracking on this single frame
        results = model.track(
            frame,
            tracker="test_botsort.yaml",
            persist=True,
            classes=target_classes
        )

        result = results[0]  # Single-frame result

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            bboxes = result.boxes.xyxy.cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()

            for track_id, bbox, cls_id in zip(track_ids, bboxes, class_ids):
                if cls_id not in target_classes:
                    continue

                if track_id not in track_id_colors:
                    track_id_colors[track_id] = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )
                color = track_id_colors[track_id]

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                text = f"ID: {track_id}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                )

                bg_x1 = x1
                bg_y1 = max(0, y1 - 15 - text_height)
                bg_x2 = x1 + text_width
                bg_y2 = y1 - 15 + baseline

                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                cv2.putText(frame, text, 
                            (x1, y1 - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, color, 2)

        scale = 0.5
        h,w = frame.shape[:2]
        new_w = int(w*scale)
        new_h = int(h*scale)

        display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("YOLO ByteTrack", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"Processed frame {frame_id}", end='\r')
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

bytetrack(VIDEO_PATH, target_classes=[0], start_frame = 6000)