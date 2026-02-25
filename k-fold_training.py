from ultralytics import YOLO
from pathlib import Path
from save_fold_metrics import save_fold_metrics

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
        mosaic=0.25,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,

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
    #specify location of YAMLs for the folds
    yaml_loc = Path("C:/Users/josie/local_data/YOLO/training_data/400_merged_vert/cross_val_yamls")

    #get yaml paths into a list
    yaml_files = list(yaml_loc.glob("*.yaml"))

    #for each yaml file, train the experiment

    for i, yaml in enumerate(yaml_files):
        fold = i+1

        train_yolo(data_yaml= yaml,
                model_size= "yolo26n.pt",
                project_dir= "C:/Users/josie/local_data/YOLO/models/yolo26n/mosaic_0.25",
                exp_name= f"Fold_{fold}",
                epochs= 50,
                PATIENCE= 0,
                imgsz= 3008,
                batch= 1,
                device= 0)
    
    save_fold_metrics(base_dir= "C:/Users/josie/local_data/YOLO/models/yolo26n/mosaic_0.25", output_fn= "C:/Users/josie/local_data/YOLO/models/yolo26n/mosaic_025", map_col= "metrics/mAP50(B)")

