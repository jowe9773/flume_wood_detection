import pandas as pd
import plotly.express as px
import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment

# -----------------------------
# FUNCTION DEFINITIONS
# -----------------------------

def load_tracks(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("frame").reset_index(drop=True)
    return df

def build_track_info(df: pd.DataFrame) -> pd.DataFrame:

    tracks = {
        tid: group.sort_values("frame").reset_index(drop=True)
        for tid, group in df.groupby("track_id")
    }

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

    return pd.DataFrame(rows)

def display_traces_interactive(all_points_df):
    fig = px.line(
        data_frame=all_points_df,
        x="center_x",
        y="center_y",
        color="track_id",
        line_group="track_id",
        hover_data=["track_id", "frame"],
        markers=True,
    )

    fig.update_xaxes(range=[0, 9760])
    fig.update_yaxes(range=[-2000, 2000])

    fig.update_layout(title="Interactive Trace Lines")
    return fig

# -----------------------------
# HUNGARIAN MATCHING FUNCTION
# -----------------------------
def match_traces_hungarian_simple(
    trace_metadata,
    max_frame_gap=48,
    time_scale=20,
    x_scale = 1,
    y_scale = 1,            
    max_distance=800,           # cutoff for valid matches
    downstream_tolerance=50     # how much upstream allowed
):

    traces = trace_metadata.reset_index(drop=True)
    n = len(traces)

    cost_matrix = np.full((n, n), fill_value=1e6)

    for i, us in traces.iterrows():
        for j, ds in traces.iterrows():

            # -------------------------
            # BASIC FILTERS
            # -------------------------

            # must be future
            if ds["start_frame"] <= us["end_frame"]:
                continue

            dt = ds["start_frame"] - us["end_frame"]

            if dt > max_frame_gap:
                continue

            # same class
            if ds["class_id"] != us["class_id"]:
                continue

            # roughly downstream (allow small upstream)
            dx = ds["start_x"] - us["end_x"]
            if dx < -downstream_tolerance:
                continue

            # -------------------------
            # 3D DISTANCE (x, y, time)
            # -------------------------
            dx = x_scale*(ds["start_x"] - us["end_x"])
            dy = y_scale*(ds["start_y"] - us["end_y"])

            dt_scaled = dt * time_scale

            distance = np.sqrt(dx**2 + dy**2 + dt_scaled**2)

            # apply cutoff
            if distance > max_distance:
                continue

            cost_matrix[i, j] = distance

    # -------------------------
    # Hungarian assignment
    # -------------------------
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]

        if cost >= 1e5:
            continue

        matches.append((
            traces.loc[r, "trace_id"],
            traces.loc[c, "trace_id"],
            cost
        ))

    return pd.DataFrame(matches, columns=[
        "us_trace_id", "ds_trace_id", "distance"
    ])

# -----------------------------
# MESSY TRACK MATCHING FUNCTION
# -----------------------------
def merge_overlapping_tracks(df: pd.DataFrame, max_spatial_distance: float = 15.0) -> pd.DataFrame:
    """
    Finds pairs of tracks that overlap in time and are spatially close during
    the overlap window. Merges them into one track, keeping the upstream
    (smaller start_x) track's detections during the overlap, and appending
    the downstream track's non-overlapping tail.
    
    Run this after build_complete_traces and before filter_short_traces.
    """
    df = df.copy()
    
    # We'll iteratively merge until no more merges are found
    # (handles chains of overlapping tracks)
    merged_any = True
    
    while merged_any:
        merged_any = False
        
        track_ids = df["track_id"].unique()
        
        # Build a quick lookup: track_id -> (start_frame, end_frame, start_x)
        track_info = {}
        for tid in track_ids:
            t = df[df["track_id"] == tid]
            track_info[tid] = {
                "start_frame": t["frame"].min(),
                "end_frame":   t["frame"].max(),
                "start_x":     t.loc[t["frame"].idxmin(), "center_x"]
            }
        
        # Find overlapping pairs that are spatially close
        merged_in_pass = set()
        
        for i, tid_a in enumerate(track_ids):
            if tid_a in merged_in_pass:
                continue
                
            for tid_b in track_ids[i+1:]:
                if tid_b in merged_in_pass:
                    continue
                
                info_a = track_info[tid_a]
                info_b = track_info[tid_b]
                
                # Find overlapping frame range
                overlap_start = max(info_a["start_frame"], info_b["start_frame"])
                overlap_end   = min(info_a["end_frame"],   info_b["end_frame"])
                
                if overlap_start > overlap_end:
                    # No temporal overlap
                    continue
                
                # Get detections from both tracks during overlap window
                frames_a = df[
                    (df["track_id"] == tid_a) &
                    (df["frame"] >= overlap_start) &
                    (df["frame"] <= overlap_end)
                ][["frame", "center_x", "center_y"]].set_index("frame")
                
                frames_b = df[
                    (df["track_id"] == tid_b) &
                    (df["frame"] >= overlap_start) &
                    (df["frame"] <= overlap_end)
                ][["frame", "center_x", "center_y"]].set_index("frame")
                
                # Only compare frames where BOTH tracks have a detection
                common_frames = frames_a.index.intersection(frames_b.index)
                
                if len(common_frames) == 0:
                    continue
                
                # Compute mean spatial distance across shared frames
                dx = frames_a.loc[common_frames, "center_x"] - frames_b.loc[common_frames, "center_x"]
                dy = frames_a.loc[common_frames, "center_y"] - frames_b.loc[common_frames, "center_y"]
                mean_dist = np.sqrt(dx**2 + dy**2).mean()
                
                if mean_dist > max_spatial_distance:
                    continue
                
                # ---------------------------------
                # These tracks should be merged
                # Upstream = smaller start_x
                # ---------------------------------
                if info_a["start_x"] <= info_b["start_x"]:
                    us_id, ds_id = tid_a, tid_b
                else:
                    us_id, ds_id = tid_b, tid_a
                
                us_end_frame = track_info[us_id]["end_frame"]
                
                # Keep all upstream detections
                us_rows = df[df["track_id"] == us_id].copy()
                
                # Keep only the non-overlapping tail of the downstream track
                ds_tail = df[
                    (df["track_id"] == ds_id) &
                    (df["frame"] > us_end_frame)
                ].copy()
                ds_tail["track_id"] = us_id
                
                # Drop both tracks from df, re-add merged version
                df = df[~df["track_id"].isin([us_id, ds_id])]
                df = pd.concat([df, us_rows, ds_tail], ignore_index=True)
                
                merged_in_pass.add(ds_id)
                merged_any = True
                break  # restart inner loop for tid_a since track_ids changed
            
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)

# -----------------------------
# TRACE CHAINING
# -----------------------------
def build_complete_traces(matches_df, all_points_df):

    # -------------------------
    # Build mapping
    # -------------------------
    mapping = dict(
        matches_df.set_index("us_trace_id")["ds_trace_id"]
    )

    # reverse mapping (optional but helpful)
    reverse_mapping = dict(
        matches_df.set_index("ds_trace_id")["us_trace_id"]
    )

    all_us = set(matches_df["us_trace_id"])
    all_ds = set(matches_df["ds_trace_id"])

    # start nodes = never downstream of anything
    start_nodes = all_us - all_ds

    #create a characteristic that shows as true for an actual detection
    all_points_df = all_points_df.copy()
    all_points_df["is_real"] = True

    # -------------------------
    # Build full chain
    # -------------------------
    def build_chain(start):
        chain = [start]
        visited = set(chain)

        while chain[-1] in mapping:
            nxt = mapping[chain[-1]]
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)

        return chain

    chains = [build_chain(s) for s in start_nodes]

    # -------------------------
    # Assign upstream-most ID
    # -------------------------
    merged_dfs = []

    for chain in chains:

        if len(chain) == 0:
            continue

        upstream_id = chain[0]   # 👈 THIS IS THE KEY CHANGE

        subset = all_points_df[
            all_points_df["track_id"].isin(chain)
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values("frame")

        # overwrite ALL IDs in chain with upstream ID
        subset["track_id"] = upstream_id

        merged_dfs.append(subset)

    # -------------------------
    # Handle unused traces (no matches)
    # -------------------------
    used_ids = set(t for chain in chains for t in chain)
    all_ids = set(all_points_df["track_id"])
    unused_ids = all_ids - used_ids

    for tid in unused_ids:
        subset = all_points_df[
            all_points_df["track_id"] == tid
        ].copy()

        subset = subset.sort_values("frame")

        # keep original ID (no change)
        subset["track_id"] = tid

        merged_dfs.append(subset)

    # -------------------------
    # Final output
    # -------------------------
    if merged_dfs:
        return pd.concat(merged_dfs, ignore_index=True), chains
    else:
        return pd.DataFrame(columns=all_points_df.columns), chains

# -----------------------------
# INTERPOLATION
# -----------------------------
def interpolate_traces(df):

    df = df.copy()
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)

    interpolated_list = []

    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame")

        start = int(group["frame"].min())
        end = int(group["frame"].max())

        track_frames = pd.DataFrame({
            "frame": range(start, end + 1)
        })
        track_frames["track_id"] = tid

        track_group = pd.merge(
            track_frames,
            group,
            on=["track_id", "frame"],
            how="left"
        )

        # mark real points BEFORE interpolation
        track_group["is_real"] = track_group["center_x"].notna()

        track_group["center_x"] = track_group["center_x"].interpolate()
        track_group["center_y"] = track_group["center_y"].interpolate()

        interpolated_list.append(track_group)

    return pd.concat(interpolated_list, ignore_index=True)

# -----------------------------
# REMOVAL OF SMALL TRACES
# -----------------------------
def filter_short_traces(df: pd.DataFrame, min_points: int = 5) -> pd.DataFrame:
    """
    Remove track_ids that have fewer than `min_points` rows (detections).
    """

    counts = df.groupby("track_id").size()

    valid_ids = counts[counts >= min_points].index

    filtered_df = df[df["track_id"].isin(valid_ids)].copy()

    return filtered_df

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)

    CSV_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/bytetrack/raw_tracking_outputs/2_0/first5_tracking_data.csv"
    OUTPUT_CSV = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/EGU_analyses/bytetrack/postprocessed_tracking_outputs/2_0/first5_pp_tracking_data.csv"

    # Load
    all_df = load_tracks(CSV_PATH)

    trace_metadata = build_track_info(all_df)

    # Filter
    trace_metadata = trace_metadata[trace_metadata["end_x"] > 500]
    valid_ids = set(trace_metadata["trace_id"])
    all_df = all_df[all_df["track_id"].isin(valid_ids)]

    # -----------------------------
    # MATCHING (GLOBAL)
    # -----------------------------
    matches_df = match_traces_hungarian_simple(
        trace_metadata,
        max_frame_gap=48,
        time_scale=10,
        x_scale = 1,
        y_scale = 10,
        max_distance=1000,
        downstream_tolerance=50
    )
    print(matches_df)

    # -----------------------------
    # BUILD TRACKS
    # -----------------------------
    complete_traces, chains = build_complete_traces(matches_df, all_df)

    complete_traces = merge_overlapping_tracks(complete_traces, max_spatial_distance=15.0)

    complete_traces_no_short = filter_short_traces(complete_traces, 12)

    # -----------------------------
    # INTERPOLATE
    # -----------------------------
    interpolated_traces = interpolate_traces(complete_traces_no_short)

    # -----------------------------
    # Save!
    # -----------------------------
    interpolated_traces.to_csv(OUTPUT_CSV)

    # -----------------------------
    # PLOT
    # -----------------------------
    print(len(np.unique(interpolated_traces["track_id"])))
    fig = display_traces_interactive(interpolated_traces)
    fig.show()