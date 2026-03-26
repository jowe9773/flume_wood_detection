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

def display_traces_interactive(all_points_df, x='center_x', y='center_y', color='track_id', line_group='track_id', hover_data=['track_id'], markers=True):
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
    
if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)
    # -----------------------------
    # PARAMETERS
    # -----------------------------

    CSV_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/baseline/test_uc_tracking_data.csv"
    OUTPUT_CSV = "merged_tracks.csv"
    MAX_FRAMES = 24 # maximum number of frames after a trace ends to look for a trace to match
    MAX_DIST = 100 # maximum distance between predicted location and start location of another trace that will be considered a possible match

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

    for trace in trace_metadata.itertuples(index=False): #iterate through the traces (metadata df) and detect possible matches

        possible_matches = pd.DataFrame(columns= trace_metadata.columns) #make a dataframe for possible matches for the trace

        #first, we are going to start at the frame after the trace ended, and look for new traces that started downstream, iterating through a selected number of frames
        max_frames = 48 #look for a trace to connect up to this number of frames after the trace dissapeared

        for i in range(max_frames): # iterate through the number of frames to look at
            search_frame = trace.end_frame + 1 + i #calculate the number of the frame to look at, starting with the one after the trace ended
            
            #parameters for finding possible traces
            search_frame = trace_metadata["start_frame"] == search_frame #starts in the search frame
            downstream = trace_metadata["start_x"] > trace.end_x - 15 #starts downstream of where the last trace ends
            same_size = trace_metadata["class_id"] == trace.class_id #has the same size class id
            
            #apply the parameters to find possible matches for a given frame
            possible_matches_frame = trace_metadata[search_frame & downstream & same_size].copy()

            #calcualte a distance metric including spatial and temporal distance
            possible_matches_frame["distance"] = np.sqrt((trace.end_x - possible_matches_frame["start_x"])**2 + 
                                                         (trace.end_y - possible_matches_frame["start_y"])**2 + 
                                                         (i+1)**2)

            possible_matches = pd.concat([possible_matches, possible_matches_frame], ignore_index=True)

        possible_matches = possible_matches.sort_values("distance", ignore_index=True)

        #now, choose the best possible match
        if len(possible_matches) > 0:
            match = possible_matches["trace_id"][0]
            print(f"Best match for trace {trace.trace_id} = trace {match}")

            matches_df.loc[matches_df["us_trace_id"] == trace.trace_id, "ds_trace_id"] = match

        else: #if no good match, then skip
            print(f"No good match for trace {trace.trace_id}")

        

    complete_traces, chains = build_complete_traces(matches_df, all_df)

    plot = display_traces_interactive(complete_traces)
    plot.show()
    