import os

root_dir = "/ssd_data/tsy_study_venv/OptiGenesis_Lancet/dataset"
target_count = 12

print(f"正在检查目录: {root_dir}...")

total_folders = 0
for root, dirs, files in os.walk(root_dir):
    for d in dirs:
        if d.startswith("M"):
            total_folders += 1
            folder_path = os.path.join(root, d)
            # 统计 tiff 张数
            tiffs = [f for f in os.listdir(folder_path) if f.endswith('.tiff') or f.endswith('.TIFF')]
            count = len(tiffs)
            
            if count != target_count:
                print(f"⚠️ 异常: {folder_path}（图像张数: {count}）")

print(f"共检查病例文件夹数: {total_folders}")
