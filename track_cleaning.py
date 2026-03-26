import pandas as pd
import plotly.express as px
import numpy as np
import warnings

# -----------------------------
# FUNCTION DEFINITIONS
# -----------------------------

def load_tracks(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df = df.sort_values("frame").reset_index(drop=True)
    
    return df

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

def display_traces_interactive(all_points_df, x='center_x', y='center_y', color='track_id', line_group='track_id', hover_data=['track_id', "frame"], markers=True):
    # Plot lines for each track_id
    fig = px.line(
        data_frame = all_points_df,
        x=x,
        y=y,
        color=color,       # Each track gets a distinct color
        line_group=line_group,  # Ensures points for the same track are connected
        hover_data=hover_data,
        markers=markers,           # Show markers at each point (optional)
    )

    # Fix axis ranges
    fig.update_xaxes(range=[0, 9760])
    fig.update_yaxes(range=[-2000, 2000])

    fig.update_layout(title='Interactive Trace Lines')
    return fig

def find_possible_matches_for_trace(row, trace_metadata, max_frames=12, max_dist=400):

    print(f"Trace {row.trace_id}")   

def build_complete_traces(matches_df, all_points_df):
    """
    Combine fragmented traces into continuous tracks using upstream/downstream matches.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Must contain columns: ['us_trace_id', 'ds_trace_id']
    all_points_df : pd.DataFrame
        Must contain 'track_id' column + detection data

    Returns
    -------
    complete_traces : pd.DataFrame
        Same format as all_points_df, but with merged track_ids
    chains : list of lists
        The trace chains that were constructed
    """

    # -------------------------
    # 1. Build mapping (us -> ds)
    # -------------------------
    mapping = dict(
        matches_df.dropna(subset=["ds_trace_id"])
                .set_index("us_trace_id")["ds_trace_id"]
    )

    # -------------------------
    # 2. Find start nodes
    # -------------------------
    all_us = set(matches_df["us_trace_id"])
    all_ds = set(matches_df["ds_trace_id"].dropna())

    start_nodes = all_us - all_ds

    # -------------------------
    # 3. Chain builder
    # -------------------------
    def build_chain(start):
        chain = [start]
        visited = set(chain)

        while chain[-1] in mapping:
            nxt = mapping[chain[-1]]

            # prevent infinite loops (just in case)
            if nxt in visited:
                break

            chain.append(nxt)
            visited.add(nxt)

        return chain

    # -------------------------
    # 4. Build all chains
    # -------------------------
    chains = [build_chain(start) for start in start_nodes]

    # -------------------------
    # 5. Merge traces
    # -------------------------
    merged_dfs = []
    new_track_id = 0

    for chain in chains:
        subset = all_points_df[
            all_points_df["track_id"].isin(chain)
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values("frame")
        subset["track_id"] = chain[0]

        merged_dfs.append(subset)
        new_track_id += 1

    # -------------------------
    # 6. Handle unused traces
    # -------------------------
    used_ids = set([tid for chain in chains for tid in chain])
    all_ids = set(all_points_df["track_id"])

    unused_ids = all_ids - used_ids

    for tid in unused_ids:
        subset = all_points_df[
            all_points_df["track_id"] == tid
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values("frame")
        subset["track_id"] = new_track_id

        merged_dfs.append(subset)
        new_track_id += 1

    # -------------------------
    # 7. Combine everything
    # -------------------------
    if merged_dfs:
        complete_traces = pd.concat(merged_dfs, ignore_index=True)
    else:
        complete_traces = pd.DataFrame(columns=all_points_df.columns)

    return complete_traces, chains
    
def interpolate_traces(df):
    # Ensure frame is integer
    df['frame'] = df['frame'].astype(int)

    # Sort for proper interpolation order
    df = df.sort_values(['track_id', 'frame']).reset_index(drop=True)

    interpolated_list = []

    for tid, group in df.groupby('track_id'):
        group = group.sort_values('frame')

        # Convert frame bounds to int explicitly
        start = int(group['frame'].min())
        end = int(group['frame'].max())

        # Create frame range ONLY for this track
        track_frames = pd.DataFrame({
            'frame': range(start, end + 1)
        })
        track_frames['track_id'] = tid

        # Merge to introduce missing frames
        track_group = pd.merge(track_frames, group, on=['track_id', 'frame'], how='left')

        # Interpolate all gaps
        track_group['center_x'] = track_group['center_x'].interpolate(method='linear')
        track_group['center_y'] = track_group['center_y'].interpolate(method='linear')

        interpolated_list.append(track_group)

    return pd.concat(interpolated_list, ignore_index=True)

if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)
    # -----------------------------
    # PARAMETERS
    # -----------------------------

    CSV_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/test_20240530_exp1_uc_tracking_data.csv"
    OUTPUT_CSV = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/test_20240530_exp1_uc_merged_tracks.csv"

    # -----------------------------
    # Load all trace info
    # -----------------------------
    all_df = load_tracks(CSV_PATH)

    # -----------------------------
    # get metadata for each trace
    # -----------------------------
    trace_metadata = build_track_info(all_df)

    #delete traces that dont get past x = 500
    trace_metadata = trace_metadata[trace_metadata["end_x"] > 500]

    # keep only valid track_ids
    valid_ids = set(trace_metadata["trace_id"])

    # filter the full dataframe
    all_df = all_df[all_df["track_id"].isin(valid_ids)]



    # -----------------------------
    # Make a matches_df to store matching info
    # -----------------------------

    matches_df = pd.DataFrame(trace_metadata["trace_id"]).rename(columns={"trace_id": "us_trace_id"}).reset_index()
    matches_df["ds_trace_id"] = pd.NA

    # -----------------------------
    # Next, iterate through traces starting with the most upstream start position, then iterating down from there, and try to find a match
    # -----------------------------
    max_frames = 100
    start_dist = 100
    inc_rate = 50

    for i in range(max_frames): #iterate through the number of frames you want to check after the end of each trace
        for trace in trace_metadata.itertuples(index=False): #iterate through the traces (metadata df)

            search_frame = trace.end_frame + 1 + i #calc the frame number you are searching for new traces


            search_frame = trace_metadata["start_frame"] == search_frame #traces that start in the search frame in the search frame
            possible_matches = trace_metadata[search_frame].copy()

            if len(possible_matches) > 0:
                #print(f"All new traces that started {i+1} frames after trace {trace.trace_id} ended")
                #print(possible_matches)

                downstream = possible_matches["start_x"].between(trace.end_x - 15, trace.end_x + start_dist + i*inc_rate) #starts downstream of where the last trace ends, expand searcha area as more time passes
                possible_matches = possible_matches[downstream]
                #print(f"Traces that are also in the search window")
                #print(possible_matches)

                if len(possible_matches) > 0:
                    same_size = possible_matches["class_id"] == trace.class_id #has the same size class id
                    possible_matches = possible_matches[same_size]
                    #print(f"Traces that are also the same size class")
                    #print(possible_matches)

                if len(possible_matches) > 0:
                    non_matched = ~possible_matches["trace_id"].isin(matches_df["ds_trace_id"])
                    possible_matches = possible_matches[non_matched]

                    #print(f"Searching trace {trace.trace_id} for possible matches in the frame {i+1} after the end of the trace")
                    #print(f"Traces that are also not already matched")
                
                if len(possible_matches) > 0:
                    #calcualte a distance metric including spatial and temporal distance
                    possible_matches["distance"] = np.sqrt((trace.end_x - possible_matches["start_x"])**2 + (trace.end_y - possible_matches["start_y"])**2)

                    possible_matches = possible_matches.sort_values("distance", ignore_index=True)
                    match = possible_matches["trace_id"][0]

                    print(f"Best match for trace {trace.trace_id} = trace {match} which started {i+1} frames after end")

                    matches_df.loc[matches_df["us_trace_id"] == trace.trace_id, "ds_trace_id"] = match

                    print(" ")

    pd.set_option("display.max_rows", None)
    print(matches_df)


    complete_traces, chains = build_complete_traces(matches_df, all_df)

    interpolated_traces = interpolate_traces(complete_traces)
    interpolated_traces.to_csv(OUTPUT_CSV)


    plot = display_traces_interactive(interpolated_traces)
    plot.show()