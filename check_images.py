import os

root_dir = "/ssd_data/tsy_study_venv/OptiGenesis_Lancet/dataset"
target_count = 12

print(f"Checking {root_dir}...")

total_folders = 0
for root, dirs, files in os.walk(root_dir):
    for d in dirs:
        if d.startswith("M"):
            total_folders += 1
            folder_path = os.path.join(root, d)
            # Count tiff files
            tiffs = [f for f in os.listdir(folder_path) if f.endswith('.tiff') or f.endswith('.TIFF')]
            count = len(tiffs)
            
            if count != target_count:
                print(f"⚠️  Anomaly found: {folder_path} (Images: {count})")

print(f"Total folders checked: {total_folders}")
