import pandas as pd
from pathlib import Path


def save_fold_metrics(base_dir, output_fn, map_col):
    # -----------------------------
    # Collect data
    # -----------------------------
    best_rows = []

    with pd.ExcelWriter(output_fn, engine="openpyxl") as writer:
        for fold_dir in sorted(
            base_dir.glob("Fold_*"),
            key=lambda p: int(p.name.split("_")[1])
        ):
            if not fold_dir.is_dir():
                continue

            csv_path = fold_dir / "results.csv"
            if not csv_path.exists():
                print(f"Missing: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # --- write full sheet (same as before)
            df.to_excel(writer, sheet_name=fold_dir.name, index=False)

            # --- find best epoch row
            if map_col not in df.columns:
                raise ValueError(f"{map_col} not found in {csv_path}")

            best_idx = df[map_col].idxmax()
            best_row = df.loc[best_idx].copy()
            best_row["fold"] = fold_dir.name
            best_rows.append(best_row)

            print(f"Processed {fold_dir.name}")

        # -----------------------------
        # Create all_folds summary
        # -----------------------------
        if best_rows:
            best_df = pd.DataFrame(best_rows)

            # Move fold column to front
            cols = ["fold"] + [c for c in best_df.columns if c != "fold"]
            best_df = best_df[cols]

            # numeric stats only
            numeric_cols = best_df.select_dtypes(include="number").columns

            mean_row = best_df[numeric_cols].mean()
            std_row = best_df[numeric_cols].std()

            mean_row["fold"] = "MEAN"
            std_row["fold"] = "STD"

            summary_df = pd.concat(
                [best_df, pd.DataFrame([mean_row, std_row])],
                ignore_index=True
            )

            summary_df.to_excel(writer, sheet_name="all_folds", index=False)

    print(f"\nSaved workbook to: {output_fn}")

if __name__ == "__main__":
    
    # -----------------------------
    # CONFIG — edit this
    # -----------------------------
    BASE_DIR = Path(r"C:/Users/josie/local_data/YOLO/models/yolo26n/baseline")  # folder containing Fold_* folders
    OUTPUT_XLSX = BASE_DIR / "metrics.xlsx"
    MAP_COL = "metrics/mAP50(B)"

    save_fold_metrics(BASE_DIR, OUTPUT_XLSX, MAP_COL)