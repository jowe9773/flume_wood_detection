import cv2
import pandas as pd
import random
import plotly.express as px
from trace_viewer import view_traces

# -----------------------------
# Load your tracking data
# -----------------------------
# Example DataFrame columns: frame, track_id, center_x, center_y
df = pd.read_csv(f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/20240808_exp1_43-45/uc_tracking_data_merged.csv")

# -----------------------------
# Video paths
# -----------------------------
video_path = "D:/Videos/20240808_exp1_goprodata_full.mp4"
output_path = f"C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/20240808_exp1_43-45/uc_tracking_data_merged.mp4"


def convert_to_image_coords(df, image_width, image_height):
    """
    Convert real-world coordinates to image pixel coordinates (in-place).

    Assumes:
        x: [0, 9760]
        y: [-2000, 2000]
    """

    # Scale center coordinates
    df['center_x'] = (df['center_x'] / 9760.0) * image_width
    df['center_y'] = ((df['center_y'] + 2000.0) / 4000.0) * image_height
    df['center_y'] = image_height - df['center_y']

    # Scale bounding box size
    df['width']  = (df['width']  / 9760.0) * image_width
    df['height'] = (df['height'] / 4000.0) * image_height

    return df
   
cap = cv2.VideoCapture(video_path)
img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

df = convert_to_image_coords(df, img_w, img_h)


view_traces(df, video_path, output_path, start_frame= 24*60*43, duration = 24*60*2, trail_len=100, scale = 0.4) 


