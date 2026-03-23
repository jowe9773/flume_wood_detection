import pandas as pd
import numpy as np

# -----------------------------
# PARAMETERS
# -----------------------------

CSV_PATH = "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/training_model/BOTsort/hyperparameter_tuning/uncongested/baseline_uc_tracking_data.csv"
OUTPUT_CSV = "merged_tracks.csv"

MAX_FRAME_GAP = 15     # max frames allowed between tracks
MAX_DIST = 500         # max distance between end of A and start of B
VERBOSE = True        # turn on debug printing

# -----------------------------
# FUNCTION DEFINITIONS
# -----------------------------

def load_tracks(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df = df.sort_values("frame").reset_index(drop=True)
    tracks = {tid: group.sort_values("frame").reset_index(drop=True)
              for tid, group in df.groupby("track_id")}
    return tracks

def build_track_info(tracks: dict) -> dict:
    info = {}
    for tid, tdf in tracks.items():
        start = tdf.iloc[0]
        end = tdf.iloc[-1]
        info[tid] = {
            "start_frame": start.frame,
            "end_frame": end.frame,
            "start_x": start.center_x,
            "start_y": start.center_y,
            "end_x": end.center_x,
            "end_y": end.center_y,
        }
    return info

def should_merge(trackA: dict, trackB: dict, max_frame_gap: int, max_dist: float, verbose=False) -> bool:
    gap = trackB["start_frame"] - trackA["end_frame"]
    if gap <= 0 or gap > max_frame_gap:
        if verbose:
            print(f"Reject merge: frame gap too large or negative ({gap})")
        return False
    if trackB["start_x"] <= trackA["end_x"]:
        if verbose:
            print(f"Reject merge: not downstream (trackA end_x={trackA['end_x']}, trackB start_x={trackB['start_x']})")
        return False
    end_pos = np.array([trackA["end_x"], trackA["end_y"]])
    start_pos = np.array([trackB["start_x"], trackB["start_y"]])
    dist = np.linalg.norm(start_pos - end_pos)
    if dist >= max_dist:
        if verbose:
            print(f"Reject merge: distance too large ({dist:.1f} > {max_dist})")
        return False
    if verbose:
        print(f"Accept merge: frame gap={gap}, distance={dist:.1f}")
    return True

def find_merge_pairs(tracks: dict, track_info: dict, max_frame_gap: int, max_dist: float, verbose=False) -> list:
    sorted_ids = sorted(track_info, key=lambda t: track_info[t]["start_x"])
    merge_pairs = []
    used_children = set()

    for i, idA in enumerate(sorted_ids):
        if idA not in track_info:
            continue

        best_candidate = None
        best_dist = np.inf

        for idB in sorted_ids[i+1:]:
            if idB in used_children or idB not in track_info:
                continue
            if verbose:
                print(f"Considering merge: parent={idA}, child={idB}")
            if should_merge(track_info[idA], track_info[idB], max_frame_gap, max_dist, verbose=verbose):
                end_pos = np.array([track_info[idA]["end_x"], track_info[idA]["end_y"]])
                start_pos = np.array([track_info[idB]["start_x"], track_info[idB]["start_y"]])
                dist = np.linalg.norm(start_pos - end_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_candidate = idB

        if best_candidate is not None:
            if verbose:
                print(f"Selected merge pair: parent={idA}, child={best_candidate}, distance={best_dist:.1f}")
            merge_pairs.append((idA, best_candidate))
            used_children.add(best_candidate)

    if verbose:
        print(f"Total merge candidates found this pass: {len(merge_pairs)}")
    return merge_pairs

def merge_tracks(tracks: dict, merge_pairs: list, verbose=False) -> dict:
    for parent, child in merge_pairs:
        if parent not in tracks or child not in tracks:
            continue
        if verbose:
            print(f"Merging tracks: {parent} + {child}")
        merged = pd.concat([tracks[parent], tracks[child]]).sort_values("frame").reset_index(drop=True)
        merged["track_id"] = parent
        tracks[parent] = merged
        del tracks[child]
    return tracks

def iterative_merge(tracks: dict, max_frame_gap: int, max_dist: float, verbose=False) -> dict:
    merges_performed = True
    iteration = 0
    while merges_performed:
        iteration += 1
        if verbose:
            print(f"\n=== Merge iteration {iteration} ===")
        merges_performed = False
        track_info = build_track_info(tracks)
        merge_pairs = find_merge_pairs(tracks, track_info, max_frame_gap, max_dist, verbose=verbose)
        if merge_pairs:
            tracks = merge_tracks(tracks, merge_pairs, verbose=verbose)
            merges_performed = True
    return tracks

def combine_tracks_to_dataframe(tracks: dict) -> pd.DataFrame:
    return pd.concat(tracks.values()).sort_values("frame").reset_index(drop=True)

# -----------------------------
# MAIN SCRIPT
# -----------------------------

if __name__ == "__main__":

    tracks = load_tracks(CSV_PATH)
    print(f"Original tracks: {len(tracks)}")

    tracks = iterative_merge(tracks, MAX_FRAME_GAP, MAX_DIST, verbose=VERBOSE)
    print(f"Merged tracks: {len(tracks)}")

    merged_df = combine_tracks_to_dataframe(tracks)
    merged_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Merged track data saved to {OUTPUT_CSV}")