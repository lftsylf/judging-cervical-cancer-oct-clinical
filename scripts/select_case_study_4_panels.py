#!/usr/bin/env python3
"""
Select 4-panel case study candidates for Grad-CAM qualitative analysis.

This version DOES NOT copy images. It only exports candidate manifests with:
- oct_id
- image_folder
- probable tiff filename (oct_id.tiff)
- probability and label info

Use these manifests to locate files on other server directories/symlink chains.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# ===== User-configurable paths =====
HUAXI_CSV = "outputs/消融实验/outputs_wma_ema_aux/huaxi/seed_42/logs/external_sample_predictions.csv"
XIANGYA_CSV = "outputs/消融实验/outputs_wma_ema_aux/xiangya/seed_42/logs/external_sample_predictions.csv"

OUT_ROOT = "case_study_4_panels"


CASE_DIRS = {
    "case1": "1_Huaxi_Easy_TP",
    "case2": "2_Xiangya_Hard_TP",
    "case3": "3_Huaxi_Easy_TN",
    "case4": "4_Xiangya_False_Positive",
}

def ensure_columns(df: pd.DataFrame) -> Tuple[str, str]:
    label_col = "true_label" if "true_label" in df.columns else ("y_true" if "y_true" in df.columns else None)
    prob_col = "prob" if "prob" in df.columns else ("prob_positive" if "prob_positive" in df.columns else None)
    if label_col is None:
        raise ValueError("Cannot find label column. Expected one of: true_label / y_true")
    if prob_col is None:
        raise ValueError("Cannot find probability column. Expected one of: prob / prob_positive")
    return label_col, prob_col


def export_manifest(out_dir: Path, title: str, center_name: str, df_case: pd.DataFrame, label_col: str, prob_col: str) -> None:
    md_path = out_dir / "candidate_manifest.md"
    txt_path = out_dir / "candidate_manifest.txt"

    lines_txt: List[str] = []
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| rank | center | oct_id | true_label | prob | image_folder | guessed_tiff_name |\n")
        f.write("|---:|---|---|---:|---:|---|---|\n")
        for i, (_, row) in enumerate(df_case.iterrows(), start=1):
            oct_id = str(row.get("oct_id", "unknown"))
            true_label = int(row[label_col])
            prob = float(row[prob_col])
            image_folder = str(row.get("image_folder", ""))
            guessed_tiff = f"{oct_id}.tiff"
            f.write(
                f"| {i} | {center_name} | {oct_id} | {true_label} | {prob:.4f} | {image_folder} | {guessed_tiff} |\n"
            )
            lines_txt.append(
                f"{i}. center={center_name} oct_id={oct_id} true_label={true_label} prob={prob:.4f} "
                f"guessed_tiff={guessed_tiff} image_folder={image_folder}"
            )

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines_txt) + "\n")


def main() -> None:
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    for d in CASE_DIRS.values():
        (out_root / d).mkdir(parents=True, exist_ok=True)

    huaxi_df = pd.read_csv(HUAXI_CSV)
    xiangya_df = pd.read_csv(XIANGYA_CSV)
    h_label, h_prob = ensure_columns(huaxi_df)
    x_label, x_prob = ensure_columns(xiangya_df)

    # Case 1: Huaxi easy TP: label==1 and prob>0.85, fallback to highest label==1 if not enough
    case1_pool = huaxi_df[(huaxi_df[h_label] == 1) & (huaxi_df[h_prob] > 0.85)].sort_values(h_prob, ascending=False)
    if len(case1_pool) < 5:
        case1_pool = huaxi_df[(huaxi_df[h_label] == 1)].sort_values(h_prob, ascending=False)
    case1 = case1_pool.head(5)

    # Case 2: Xiangya hard TP: label==1 and 0.55<prob<0.75, asc top5
    case2 = xiangya_df[(xiangya_df[x_label] == 1) & (xiangya_df[x_prob] > 0.55) & (xiangya_df[x_prob] < 0.75)].sort_values(
        x_prob, ascending=True
    ).head(5)

    # Case 3: Huaxi easy TN: label==0 and prob<0.15, fallback to lowest label==0 if not enough
    case3_pool = huaxi_df[(huaxi_df[h_label] == 0) & (huaxi_df[h_prob] < 0.15)].sort_values(h_prob, ascending=True)
    if len(case3_pool) < 5:
        case3_pool = huaxi_df[(huaxi_df[h_label] == 0)].sort_values(h_prob, ascending=True)
    case3 = case3_pool.head(5)

    # Case 4: Xiangya FP exploration: label==0 and prob>0.80 desc top5, fallback to highest probs among label==0
    case4_pool = xiangya_df[(xiangya_df[x_label] == 0) & (xiangya_df[x_prob] > 0.80)].sort_values(x_prob, ascending=False)
    if len(case4_pool) == 0:
        case4_pool = xiangya_df[(xiangya_df[x_label] == 0)].sort_values(x_prob, ascending=False)
    case4 = case4_pool.head(5)

    tasks = [
        ("Case 1 Huaxi Easy TP", "case1", "huaxi", case1, h_prob),
        ("Case 2 Xiangya Hard TP", "case2", "xiangya", case2, x_prob),
        ("Case 3 Huaxi Easy TN", "case3", "huaxi", case3, h_prob),
        ("Case 4 Xiangya False Positive", "case4", "xiangya", case4, x_prob),
    ]

    for title, case_key, center_name, df_case, prob_col in tasks:
        out_dir = out_root / CASE_DIRS[case_key]
        label_col = h_label if center_name == "huaxi" else x_label
        export_manifest(out_dir, title, center_name, df_case, label_col, prob_col)
        print(f"[OK] {title}: selected={len(df_case)} -> {out_dir}")

    print("\nDone. Please check case_study_4_panels/*/candidate_manifest.*")


if __name__ == "__main__":
    main()
