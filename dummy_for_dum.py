import plotly.express as px
import pandas as pd

def build_track_info(df: pd.DataFrame) -> pd.DataFrame:

    tracks = {tid: group.sort_values("frame").reset_index(drop=True)
              for tid, group in df.groupby("track_id")}

    rows = []
    
    for tid, tdf in tracks.items():
        start = tdf.iloc[0]
        end = tdf.iloc[-1]
        
        rows.append({
            "trace_id": tid,
            "start_frame": start.frame,
            "end_frame": end.frame,
            "start_x": start.center_x,
            "start_y": start.center_y,
            "end_x": end.center_x,
            "end_y": end.center_y,
            "class_id": end.class_name,
            "duration": end.frame - start.frame,
        })
    
    df = pd.DataFrame(rows)
    return df


df = pd.read_csv("C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/20240808_exp1_43-45/uc_tracking_data.csv")
df = df.sort_values("frame").reset_index(drop=True)

# -----------------------------
# get metadata for each trace
# -----------------------------
trace_metadata = build_track_info(df)

# print details

print(f"There are {len(df)} detections in this tracking dataset with {len(trace_metadata)} traces.")

