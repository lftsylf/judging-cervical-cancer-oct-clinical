"""
将 LOHO fold 数据转换为训练脚本可直接读取的 development_*.csv / external_*.csv。

输入目录示例:
dataset/loho_5centers/fold_01_external_LiaoNing/
  - development_dataset.csv
  - external_dataset.csv
  - development_octData/
  - external_octData/
"""
import os
import pandas as pd


FOLD_TO_NAME = {
    "fold_01_external_LiaoNing": "liaoning",
    "fold_02_external_HuaXi": "huaxi",
    "fold_03_external_XiangYa": "xiangya",
}


def normalize_hpv(hpv_str):
    if pd.isna(hpv_str) or str(hpv_str).strip() in ["NAN", "(-)", "", "/"]:
        return 0
    hpv_str = str(hpv_str).strip().lower()
    if "+" in hpv_str or "阳性" in hpv_str or any(x in hpv_str for x in ["16", "18", "56", "17"]):
        return 1
    return 0


def normalize_tct(tct_str):
    if pd.isna(tct_str) or str(tct_str).strip() in ["NAN", "(-)", "", "/"]:
        return 0
    tct_str = str(tct_str).strip().upper()
    tct_map = {
        "NILM": 0,
        "ASCUS": 1,
        "ASC-US": 1,
        "LSIL": 2,
        "HSIL": 4,
        "挖空细胞": 2,
    }
    for key, value in tct_map.items():
        if key in tct_str:
            return value
    return 0


def build_image_folder(fold_rel_path, split_prefix, center_name, oct_id):
    return os.path.join(
        "loho_5centers",
        fold_rel_path,
        f"{split_prefix}_octData",
        center_name,
        oct_id,
    )


def convert_one_csv(dataset_root, fold_name, input_csv_name, split_prefix):
    csv_path = os.path.join(dataset_root, "loho_5centers", fold_name, input_csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        oct_id = str(row.get("Resolved_OCT_IDs", row.get("OCT_ID", ""))).strip()
        if not oct_id:
            continue

        center = str(row.get("Canonical_Center", "")).strip()
        if not center:
            continue

        image_folder = build_image_folder(fold_name, split_prefix, center, oct_id)
        abs_folder = os.path.join(dataset_root, image_folder)
        if not os.path.isdir(abs_folder):
            # 如果图像目录不存在，跳过该样本，避免训练阶段报错
            continue

        records.append(
            {
                "oct_id": oct_id,
                "image_folder": image_folder,
                "age": float(row["Age"]) if pd.notna(row.get("Age")) else 45.0,
                "hpv_status": normalize_hpv(row.get("HPV_Result")),
                "tct_result": normalize_tct(row.get("TCT_Result")),
                "pathology_class": int(row["Final_Label"]) if pd.notna(row.get("Final_Label")) else 0,
            }
        )

    return pd.DataFrame(records)


def prepare_loho_csvs(dataset_root):
    for fold_name, short_name in FOLD_TO_NAME.items():
        development_df = convert_one_csv(dataset_root, fold_name, "development_dataset.csv", "development")
        external_df = convert_one_csv(dataset_root, fold_name, "external_dataset.csv", "external")

        development_out = os.path.join(dataset_root, f"development_{short_name}.csv")
        external_out = os.path.join(dataset_root, f"external_{short_name}.csv")
        development_df.to_csv(development_out, index=False)
        external_df.to_csv(external_out, index=False)

        print(f"[{short_name}] development={len(development_df)} -> {development_out}")
        print(f"[{short_name}] external={len(external_df)} -> {external_out}")

        if len(development_df) == 0 or len(external_df) == 0:
            raise ValueError(
                f"[{short_name}] 生成结果为空，请检查该折图像路径链接。"
            )


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "dataset")
    prepare_loho_csvs(data_root)
