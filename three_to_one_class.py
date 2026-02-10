import os
import shutil

# Original dataset root (with 'train' and 'validate')
root_dir = "c:/Users/josie/OneDrive - UCB-O365/Wood Tracking/trying stuff out/testing_125_frames/project-11-at-2026-02-03-10-33-a34089e0/labels"

# New dataset root to save merged labels
output_root_dir = "c:/Users/josie/OneDrive - UCB-O365/Wood Tracking/trying stuff out/testing_125_frames/project-11-at-2026-02-03-10-33-a34089e0/labels_single"
os.makedirs(output_root_dir, exist_ok=True)

# Subfolders to process
subfolders = ["train", "val"]

for subfolder in subfolders:
    input_dir = os.path.join(root_dir, subfolder)
    output_dir = os.path.join(output_root_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if filename.endswith(".txt"):
            with open(input_path, "r") as f:
                lines = f.readlines()

            # Use a list to store processed lines
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip malformed lines
                parts[0] = "0"  # force class ID to 0
                new_lines.append(" ".join(parts))

            # Write with explicit newline
            with open(output_path, "w", newline='\n') as f:
                f.write("\n".join(new_lines))
                f.write("\n")  # make sure file ends with newline

        else:
            # Copy images (or other files) as-is
            shutil.copy2(input_path, output_path)

print(f"Done! New dataset saved at {output_root_dir} with a single class.")