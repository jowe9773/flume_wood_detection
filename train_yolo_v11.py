from ultralytics import YOLO
from pathlib import Path


def train_yolov11(
    data_yaml: str,
    model_size: str = "yolo26n.pt",
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
        mixup=0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.2,
        scale=0,
        fliplr=0.5,
        flipud=0.5,
        degrees=0.2,

        # -------------------
        # TRAINING BEHAVIOR
        # -------------------
        close_mosaic=10,
        patience=0,
        workers=8,
        cache=False,
    )

    return Path(project_dir) / exp_name / "weights" / "best.pt"


if __name__ == "__main__":

    import torch
    print("cuda available", torch.cuda.is_available())
    print("cuda version", torch.version.cuda)
    print("GPU count", torch.cuda.device_count())
    print("device name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)



    DATA_YAML = "C:/Users/jwelsh/YOLO/training data/by_exp_0-49/dataset.yaml"

    best_model = train_yolov11(
        data_yaml=DATA_YAML,
        model_size= "yolo26n.pt",
        epochs=500,
        batch=1,
        imgsz=3000,
        device= "0",
        exp_name= "first_test_new_data_format",
        project_dir= "C:/Users/jwelsh/YOLO/results"
    )

    print("Training finished.")
    print("Best model:", best_model)