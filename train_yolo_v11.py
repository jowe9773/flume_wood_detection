from ultralytics import YOLO
from pathlib import Path


def train_yolov11(
    data_yaml: str,
    model_size: str = "yolov11s.pt",
    project_dir: str = "runs",
    exp_name: str = "yolov11_from_scratch_finetune",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: int = 0,
):
    """
    Fine-tune a standard YOLOv11 pretrained model
    using YOLO's internal augmentation.
    """

    model = YOLO(model_size)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name=exp_name,

        # -------------------
        # YOLO INTERNAL AUGMENTATION
        # -------------------
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,

        # -------------------
        # TRAINING BEHAVIOR
        # -------------------
        close_mosaic=10,
        patience=20,
        workers=8,
        cache=False,
    )

    return Path(project_dir) / exp_name / "weights" / "best.pt"


if __name__ == "__main__":
    DATA_YAML = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/trying stuff out/testing_125_frames/project-11-at-2026-02-03-10-33-a34089e0/dataset.yaml"

    best_model = train_yolov11(
        data_yaml=DATA_YAML,
        model_size="yolov8s.pt",
        epochs=80,
        batch=12,
        device= "cpu",
        project_dir= "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/trying stuff out/testing_125_frames/"
    )

    print("Training finished.")
    print("Best model:", best_model)