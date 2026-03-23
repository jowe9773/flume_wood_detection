from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("C:/Users/josie/local_data/YOLO/models/mosaic_1_cp_0_hsv_low_translate_02/Fold_6/weights/last.pt")
    model.train(resume=True)