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
        if len(tdf) < 2: 
            #print(f"Trace {tid} only has {len(tdf)} points, removing from data")
            continue

        start = tdf.iloc[0]
        one_from_end = tdf.iloc[-2]
        end = tdf.iloc[-1]

        #find velocity between last two points in track
        point_1 = np.array((one_from_end.center_x, one_from_end.center_y))
        point_2 = np.array((end.center_x, end.center_y))

        velocity = (point_2-point_1)/ (end.frame - one_from_end.frame)
        
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
            "end_velocity": velocity
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
    
    #start by making a possible matches dataframe
    possible_matches = pd.DataFrame(columns=trace_metadata.columns) #make an empty dataframe that will contain possible match metadata
    possible_matches["est_x"] = pd.NA                               #add column for the estimated x position (will fill in based on velocity and time)
    possible_matches["est_y"] = pd.NA                               #add column for the estimated y position (will fill in based on velocity and time)

    for i in range(max_frames): #iterate through a chosen number of frames after the end of the trace, looking for traces that could be matches

        #find traces whose start frame is the frame that we are looking at
        frame_to_check = row.end_frame + i + 1
        possible_matches_from_frame = trace_metadata[trace_metadata["start_frame"] == frame_to_check].copy() 

        #add information about about where to expect the piece
        possible_matches_from_frame["est_x"] = est_x = row.end_x + row.end_velocity[0]*(i+1)
        possible_matches_from_frame["est_y"] = est_y = row.end_y + row.end_velocity[1]*(i+1)

        possible_matches_from_frame["distance"] = np.sqrt((possible_matches_from_frame["est_x"] - possible_matches_from_frame["start_x"])**2 + (possible_matches_from_frame["est_y"] - possible_matches_from_frame["start_y"])**2)

        
        #append possible matches from the frame to a list of possible matches from the MAX_FRAMES
        possible_matches = pd.concat([possible_matches, possible_matches_from_frame], ignore_index=True)

    #now lets check out these time aligned matches and remove impossible ones based on size class
    possible_matches = possible_matches[possible_matches["class_id"] == row.class_id]


    #now lets check matches for being close to where we would expect to find the trace
    possible_matches = possible_matches[possible_matches["distance"] < max_dist]

    #finally, lets check possible matches for already having been matched to another trace and remove them from the possible matches
    #possible_matches = possible_matches[~possible_matches["trace_id"].isin(matches_df["ds_trace_id"])]
    
    print(row.trace_id)
    print(possible_matches)
    print(" ")

    return possible_matches

def find_possible_matches_longer_time(row, trace_metadata, start_dist, increase_rate):
    print(" ")
    print(f"Trace {row.trace_id}")
    print(f"start_frame: {row.start_frame}, end_frame: {row.end_frame}, class_id: {row.class_id}")

    #find the lower and uper bounds of x values to consider
    lower_x = row.end_x - 10

    #find all matches within a certain x range around the last known position
    possible_matches = trace_metadata[trace_metadata['start_x'].between(lower_x, 9800)].copy()
    
    #remove matches with the same trace id as the row
    possible_matches = possible_matches[possible_matches["trace_id"] != row.trace_id]


    #only those that start after the row trace ends
    possible_matches = possible_matches[possible_matches["start_frame"] > row.end_frame]

    #now lets check out these time aligned matches and remove impossible ones based on size class
    possible_matches = possible_matches[possible_matches["class_id"] == row.class_id]
    print("Only traces that start downstream and after row trace, AND are of the same size class")
    print(possible_matches)

    #finally, lets check possible matches for already having been matched to another trace and remove them from the possible matches
    #possible_matches = possible_matches[~possible_matches["trace_id"].isin(matches_df["ds_trace_id"])]

    return possible_matches

    
if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)
    # -----------------------------
    # PARAMETERS
    # -----------------------------

    CSV_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/botsort_by_claude_uc_tracking_data.csv"
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

    # -----------------------------
    # Make a matches_df to store matching info
    # -----------------------------

    matches_df = pd.DataFrame(trace_metadata["trace_id"]).rename(columns={"trace_id": "us_trace_id"}).reset_index()
    matches_df["ds_trace_id"] = pd.NA

    # -----------------------------
    # Next, iterate through traces starting with the most upstream start position, then iterating down from there, and try to find a match
    # -----------------------------

    for row in trace_metadata.itertuples(index=False): #sort the trace metadata from upstream most to downstream most, then iterate through them

        possible_matches = find_possible_matches_for_trace(row, trace_metadata, max_frames=MAX_FRAMES, max_dist=MAX_DIST) #find all possible downstream matches for the trace being looked at (row)

        #possible_matches = find_possible_matches_longer_time(row, trace_metadata, start_dist=100, increase_rate=10)

        #handle the cases where there is 0 possible matches (do nothing), 1 possible matches (grab that match!), and more than 1 possible match (choose the closest one!)
        if len(possible_matches) < 1:
            #print(f"No downstream matches for trace {row.trace_id}")
            match = pd.NA

        if len(possible_matches) == 1:
            match = possible_matches.iloc[0]["trace_id"]
            #print(f"There is 1 possible downstream match for trace {row.trace_id} and it is trace {match}")

        if len(possible_matches) > 1: 
            match = possible_matches.sort_values("trace_id").iloc[0]["trace_id"]
            #print(f"There is more than 1 possible downstream match for trace {row.trace_id}")

        matches_df.loc[matches_df["us_trace_id"] == row.trace_id, "ds_trace_id"] = match

    pd.set_option("display.max_rows", None)
    print(matches_df.sort_values("us_trace_id").reset_index(drop=True))


    # Build a mapping from upstream to downstream
    match_dict = dict(zip(matches_df["us_trace_id"], matches_df["ds_trace_id"]))

    # Track which original track_id should be the merged id
    track_to_merged = {}

    for us in matches_df["us_trace_id"]:
        merged_id = us
        current = us
        
        # Follow the chain downstream until there is no downstream match
        while pd.notna(match_dict.get(current)):
            current = match_dict[current]
            track_to_merged[current] = merged_id  # map downstream trace to upstream merged id

    # Create new dataframe with merged track_ids
    complete_traces = all_df.copy()
    complete_traces["track_id"] = complete_traces["track_id"].map(lambda x: track_to_merged.get(x, x))

    # Optional: sort by track_id and frame
    complete_traces = complete_traces.sort_values(["track_id", "frame"]).reset_index(drop=True)

    #plot = display_traces_interactive(complete_traces)
    #plot.show()
