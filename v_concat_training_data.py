import os
import cv2
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIG — EDIT THESE
# -----------------------------

INPUT_DATASET = "C:/Users/josie/local_data/YOLO/training_data/800_single_img"
OUTPUT_DATASET = "C:/Users/josie/local_data/YOLO/training_data/400_merged_vert"

RESIZE_WIDTH = None
STITCH_MODE = "vertical"

VISUALIZE_CHECK = True
DISPLAY_TIME_MS = 1000  # 1 second

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def load_yolo_annotations(txt_path):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, w, h = map(float, parts)
                boxes.append((int(cls), xc, yc, w, h))
    return boxes


def save_yolo_annotations(boxes, txt_path):
    with open(txt_path, "w") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def resize_keep_aspect(img, target_width):
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (target_width, int(h * scale)))

def draw_yolo_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    h, w = image.shape[:2]
    img = image.copy()

    for cls, xc, yc, bw, bh in boxes:
        x_center = xc * w
        y_center = yc * h
        box_w = bw * w
        box_h = bh * h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img,
            str(cls),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return img


# -----------------------------
# MAIN LOOP — iterate experiments
# -----------------------------

in_img_root = Path(INPUT_DATASET) / "images"
in_lbl_root = Path(INPUT_DATASET) / "labels"

out_img_root = Path(OUTPUT_DATASET) / "images"
out_lbl_root = Path(OUTPUT_DATASET) / "labels"

experiment_dirs = sorted([d for d in in_img_root.iterdir() if d.is_dir()])

print(f"Found {len(experiment_dirs)} experiments")

for exp_dir in experiment_dirs:

    exp_name = exp_dir.name
    print(f"/n=== Processing experiment: {exp_name} ===")

    in_img_dir = in_img_root / exp_name
    in_lbl_dir = in_lbl_root / exp_name

    out_img_dir = out_img_root / exp_name
    out_lbl_dir = out_lbl_root / exp_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(in_img_dir.glob("*.png"))

    print(f"Images found: {len(image_paths)}")

    # Pair images: (0,1), (2,3), ...
    pair_index = 0

    for i in range(0, len(image_paths) - 1, 2):

        img1_path = image_paths[i]
        img2_path = image_paths[i + 1]

        lbl1_path = in_lbl_dir / (img1_path.stem + ".txt")
        lbl2_path = in_lbl_dir / (img2_path.stem + ".txt")

        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Optional resize
        if RESIZE_WIDTH is not None:
            img1 = resize_keep_aspect(img1, RESIZE_WIDTH)
            img2 = resize_keep_aspect(img2, RESIZE_WIDTH)
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

        # -----------------------------
        # STITCH
        # -----------------------------
        if STITCH_MODE == "vertical":
            if w1 != w2:
                raise ValueError(f"Width mismatch in {exp_name}")
            stitched = np.vstack([img1, img2])
            new_h = h1 + h2
            new_w = w1

        elif STITCH_MODE == "horizontal":
            if h1 != h2:
                raise ValueError(f"Height mismatch in {exp_name}")
            stitched = np.hstack([img1, img2])
            new_h = h1
            new_w = w1 + w2

        else:
            raise ValueError("STITCH_MODE must be 'vertical' or 'horizontal'")

        # -----------------------------
        # LOAD LABELS
        # -----------------------------
        boxes1 = load_yolo_annotations(lbl1_path)
        boxes2 = load_yolo_annotations(lbl2_path)

        new_boxes = []

        # --- image 1 (top) ---
        for cls, xc, yc, w, h in boxes1:
            x_center_px = xc * w1
            y_center_px = yc * h1

            new_boxes.append((
                cls,
                x_center_px / new_w,
                y_center_px / new_h,
                w * (w1 / new_w),
                h * (h1 / new_h),
            ))

        # --- image 2 (bottom) ---
        for cls, xc, yc, w, h in boxes2:
            x_center_px = xc * w2
            y_center_px = yc * h2

            if STITCH_MODE == "vertical":
                y_center_px += h1
            else:
                x_center_px += w1

            new_boxes.append((
                cls,
                x_center_px / new_w,
                y_center_px / new_h,
                w * (w2 / new_w),
                h * (h2 / new_h),
            ))

        # -----------------------------
        # SAVE
        # -----------------------------
        out_img_name = f"{exp_name}_pair{pair_index:02d}.png"
        out_lbl_name = f"{exp_name}_pair{pair_index:02d}.txt"
        pair_index += 1

        cv2.imwrite(str(out_img_dir / out_img_name), stitched)
        save_yolo_annotations(new_boxes, out_lbl_dir / out_lbl_name)

        # -----------------------------
        # QUICK DISPLAY CHECK
        # -----------------------------
        if VISUALIZE_CHECK:
            viz_img = draw_yolo_boxes(stitched, new_boxes)

            scale = 0.25
            h,w = viz_img.shape[:2]
            new_w = int(w*scale)
            new_h = int(h*scale)

            display_frame = cv2.resize(viz_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.imshow("Stitch Check", display_frame)
            key = cv2.waitKey(DISPLAY_TIME_MS)

            # Optional: press 'q' to stop early
            if key & 0xFF == ord("q"):
                VISUALIZE_CHECK = False
                cv2.destroyAllWindows()

        print(f"Saved pair {pair_index:02d}")

cv2.destroyAllWindows()
print("/n✅ Done stitching dataset!")