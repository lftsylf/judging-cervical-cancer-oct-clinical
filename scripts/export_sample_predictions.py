"""
从已有 checkpoint 导出 development / external 逐样本概率（不重训）。

用法:
    python scripts/export_sample_predictions.py --hospital xiangya
    python scripts/export_sample_predictions.py --hospital huaxi --output-root baseline_outputs
"""

import argparse
import os
import sys

import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from configs.lancet_config import Config
from data.dataset_lancet import get_dataloader
from models.optigenesis_model import OptiGenesis
from training.trainer import validate


def export_one_split(model, loader, csv_path, split_name, hospital, logs_dir, device):
    """对单个划分跑推理并写入 CSV。"""
    _, details = validate(model, loader, device, verbose=False, return_predictions=True)
    src = pd.read_csv(csv_path).reset_index(drop=True)
    n = min(len(src), len(details["targets"]))
    if n == 0:
        print(f"⚠️ 【{split_name}】无样本，跳过导出。")
        return
    out = src.iloc[:n].copy()
    out["hospital"] = hospital
    out["split"] = split_name
    out["y_true"] = details["targets"][:n].astype(int)
    out["prob_positive"] = details["probs"][:n].astype(float)
    out["uncertainty"] = details["uncertainties"][:n].astype(float)
    out["y_pred"] = details["preds"][:n].astype(int)

    save_path = os.path.join(logs_dir, f"{split_name}_sample_predictions.csv")
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 【{split_name}】逐样本结果已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="从 checkpoint 导出逐样本预测 CSV")
    parser.add_argument("--hospital", required=True, help="医院名小写，如 xiangya / huaxi / liaoning")
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="项目根下输出根目录名（默认 outputs；归档 baseline 用 baseline_outputs）",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="可选：直接指定 .pth 路径（指定后仍用 --output-root 下的 logs 目录写 CSV）",
    )
    args = parser.parse_args()

    hospital = args.hospital.lower()
    train_csv = os.path.join(Config.DATA_ROOT, f"development_{hospital}.csv")
    ext_csv = os.path.join(Config.DATA_ROOT, f"external_{hospital}.csv")
    if not os.path.exists(train_csv) or not os.path.exists(ext_csv):
        raise FileNotFoundError(f"缺少数据 CSV:\n- {train_csv}\n- {ext_csv}")

    output_root = os.path.join(PROJECT_ROOT, args.output_root)
    run_dir = os.path.join(output_root, hospital)
    logs_dir = os.path.join(run_dir, "logs")
    ckpt = args.checkpoint or os.path.join(run_dir, "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"未找到权重文件: {ckpt}")
    os.makedirs(logs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"【设备】{device}  |  【医院】{hospital}  |  【权重】{ckpt}")

    model = OptiGenesis(
        model_name=Config.BACKBONE,
        use_clinical=Config.USE_CLINICAL,
        num_classes=Config.NUM_CLASSES,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    dev_loader = get_dataloader(train_csv, mode="val")
    ext_loader = get_dataloader(ext_csv, mode="val")

    print("【开始导出】development …")
    export_one_split(model, dev_loader, train_csv, "development", hospital, logs_dir, device)
    print("【开始导出】external …")
    export_one_split(model, ext_loader, ext_csv, "external", hospital, logs_dir, device)
    print("【完成】development 与 external 均已导出。")


if __name__ == "__main__":
    main()
