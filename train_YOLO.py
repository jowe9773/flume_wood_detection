from ultralytics import YOLO
from pathlib import Path

# create a function that will train yolo

def train_yolo(
    data_yaml: str,
    model_size: str = "yolo26n.pt",
    project_dir: str = "runs",
    exp_name: str = "test",
    epochs: int = 500,
    imgsz: int = 800,
    batch: int = 16,
    device: int = 0,
    PATIENCE: int = 40
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
        mixup=0.0,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.0,
        fliplr=0.5,
        flipud=0.5,
        degrees=0.1,

        # -------------------
        # TRAINING BEHAVIOR
        # -------------------
        close_mosaic=10,
        patience=PATIENCE,
        workers=8,
        cache=False,
    )

    return Path(project_dir) / exp_name / "weights" / "best.pt"


if __name__ == "__main__":
    #if this script is run, then it will do a single training session
    DATA_YAML = "C:/Users/josie/local_data/YOLO/training_data/400_merged_vert/cross_val_yamls/fold_8.yaml"

    best_model = train_yolo(
        data_yaml=DATA_YAML,
        model_size= "yolo26n.pt",
        epochs=500,
        batch=1,
        imgsz=3000, 
        device= "0",
        exp_name= "Fold_8",
        project_dir= "C:/Users/josie/local_data/YOLO/models/yolo26n",
        PATIENCE=50
    )

    print("Training finished.")
    print("Best model:", best_model)