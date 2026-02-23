from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("C:/Users/josie/local_data/YOLO/models/yolo26n/Fold_7/weights/last.pt")
    model.train(resume=True)