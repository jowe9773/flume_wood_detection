from ultralytics import YOLO
from pathlib import Path


def train_yolov11(
    data_yaml: str,
    model_size: str = "yolo11s.pt",
    project_dir: str = "runs",
    exp_name: str = "test",
    epochs: int = 100,
    imgsz: int = 800,
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
    DATA_YAML = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/0-24_annotations_three_classes/dataset.yaml"

    best_model = train_yolov11(
        data_yaml=DATA_YAML,
        model_size= "yolo11m.pt", #"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/trying stuff out/using_pretrained_model/best.pt",
        epochs=500,
        batch=16,
        imgsz=800,
        device= "0",
        exp_name= "yolo11m",
        project_dir= "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/0-24_annotations_three_classes"
    )

    print("Training finished.")
    print("Best model:", best_model)