import os
import cv2
import numpy as np
from pathlib import Path

# -----------------------------
# CONFIG — EDIT THESE
# -----------------------------

INPUT_DATASET = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/0-24_annotations_three_classes"

OUTPUT_DATASET = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/0-24_annotations_three_classes_vconcat"


RESIZE_WIDTH = None   # set to None to keep original size
STITCH_MODE = "vertical"  # "vertical" recommended for your 3900x1600 case

# which splits to process
SPLITS = ["train", "val"]  # add "test" if needed

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

# -----------------------------
# PROCESS EACH SPLIT
# -----------------------------

for split in SPLITS:

    in_img_dir = Path(INPUT_DATASET) / "images" / split
    in_lbl_dir = Path(INPUT_DATASET) / "labels" / split

    out_img_dir = Path(OUTPUT_DATASET) / "images" / split
    out_lbl_dir = Path(OUTPUT_DATASET) / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(in_img_dir.glob("*.png"))

    print(f"\nProcessing split: {split} — {len(image_paths)} images")

    # Stitch in pairs: (0,1), (2,3), (4,5)...
    for i in range(0, len(image_paths) - 1, 2):

        img1_path = image_paths[i]
        img2_path = image_paths[i + 1]

        lbl1_path = in_lbl_dir / (img1_path.stem + ".txt")
        lbl2_path = in_lbl_dir / (img2_path.stem + ".txt")

        # Load images
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

        # Stitch
        if STITCH_MODE == "vertical":
            if w1 != w2:
                raise ValueError("Widths must match for vertical stitching.")
            stitched = np.vstack([img1, img2])
            new_h = h1 + h2
            new_w = w1

        elif STITCH_MODE == "horizontal":
            if h1 != h2:
                raise ValueError("Heights must match for horizontal stitching.")
            stitched = np.hstack([img1, img2])
            new_h = h1
            new_w = w1 + w2

        else:
            raise ValueError("STITCH_MODE must be 'vertical' or 'horizontal'")

        # Load annotations
        boxes1 = load_yolo_annotations(lbl1_path)
        boxes2 = load_yolo_annotations(lbl2_path)

        new_boxes = []

        # -----------------------------
        # FIXED: Adjust boxes from IMAGE 1 (top image)
        # -----------------------------
        for cls, xc, yc, w, h in boxes1:

            # Convert normalized → pixel coords in image 1
            x_center_px = xc * w1
            y_center_px = yc * h1

            if STITCH_MODE == "vertical":
                # For top image: x stays the same, y stays the same
                new_xc = x_center_px / new_w
                new_yc = y_center_px / new_h
                new_w_norm = w * (w1 / new_w)
                new_h_norm = h * (h1 / new_h)

            else:  # horizontal stitch
                new_xc = x_center_px / new_w
                new_yc = y_center_px / new_h
                new_w_norm = w * (w1 / new_w)
                new_h_norm = h * (h1 / new_h)

            new_boxes.append((cls, new_xc, new_yc, new_w_norm, new_h_norm))

        # -----------------------------
        # Adjust boxes from IMAGE 2 (bottom image if vertical)
        # -----------------------------
        for cls, xc, yc, w, h in boxes2:

            # Convert normalized → pixel coords in image 2
            x_center_px = xc * w2
            y_center_px = yc * h2

            if STITCH_MODE == "vertical":
                # Shift DOWN by height of image 1
                y_center_px += h1

                new_xc = x_center_px / new_w
                new_yc = y_center_px / new_h
                new_w_norm = w * (w2 / new_w)
                new_h_norm = h * (h2 / new_h)

            else:  # horizontal
                # Shift RIGHT by width of image 1
                x_center_px += w1

                new_xc = x_center_px / new_w
                new_yc = y_center_px / new_h
                new_w_norm = w * (w2 / new_w)
                new_h_norm = h * (h2 / new_h)

            new_boxes.append((cls, new_xc, new_yc, new_w_norm, new_h_norm))

        # Save outputs
        out_img_name = f"stitched_{img1_path.stem}_{img2_path.stem}.jpg"
        out_lbl_name = f"stitched_{img1_path.stem}_{img2_path.stem}.txt"

        cv2.imwrite(str(out_img_dir / out_img_name), stitched)
        save_yolo_annotations(new_boxes, out_lbl_dir / out_lbl_name)

        print(f"Stitched: {img1_path.name} + {img2_path.name} -> {out_img_name}")

print("\n✅ Done stitching dataset!")