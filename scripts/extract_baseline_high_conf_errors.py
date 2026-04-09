"""
从 Baseline external 预测中提取高置信度误判样本（FP/FN），用于 Error Analysis。
"""

import argparse
import os
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDS = ["xiangya", "huaxi", "liaoning"]


def load_tstar_map(csv_path: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    df = df[df["方案"] == "t*_maxthr"].copy()
    return {str(r["折"]): float(r["threshold"]) for _, r in df.iterrows()}


def pick_top_errors(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    # 误判置信度：FP 看 p(y=1) 的绝对高值，FN 看 p(y=0)=1-p(y=1) 的绝对高值
    fp = df[(df["y_true"] == 0) & (df["y_pred_tstar"] == 1)].copy()
    fn = df[(df["y_true"] == 1) & (df["y_pred_tstar"] == 0)].copy()

    fp["error_type"] = "FP"
    fp["error_confidence"] = fp["prob_positive"]
    fn["error_type"] = "FN"
    fn["error_confidence"] = 1.0 - fn["prob_positive"]

    fp_top = fp.sort_values("error_confidence", ascending=False).head(topk)
    fn_top = fn.sort_values("error_confidence", ascending=False).head(topk)
    out = pd.concat([fp_top, fn_top], axis=0, ignore_index=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="提取 Baseline 高置信度 FP/FN 样本")
    parser.add_argument(
        "--baseline-root",
        default="all_outputs/baseline_outputs",
        help="baseline 输出根目录（相对项目根）",
    )
    parser.add_argument(
        "--tstar-csv",
        default="all_outputs/baseline_outputs/postprocess_threshold/external_metrics_at_0.5_vs_tstar_maxthr.csv",
        help="Baseline t* 表（含每折 threshold）",
    )
    parser.add_argument("--topk", type=int, default=5, help="每类错误(FP/FN)提取数量")
    parser.add_argument(
        "--output-dir",
        default="all_outputs/postprocess_threshold_all_models/error_analysis",
        help="输出目录（相对项目根）",
    )
    args = parser.parse_args()

    tstar_map = load_tstar_map(os.path.join(PROJECT_ROOT, args.tstar_csv))
    out_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[pd.DataFrame] = []
    for fold in FOLDS:
        ext_csv = os.path.join(PROJECT_ROOT, args.baseline_root, fold, "logs", "external_sample_predictions.csv")
        df = pd.read_csv(ext_csv)
        thr = tstar_map[fold]
        df["fold"] = fold
        df["tstar"] = thr
        df["y_pred_tstar"] = (df["prob_positive"] >= thr).astype(int)
        all_rows.append(df)

    full = pd.concat(all_rows, axis=0, ignore_index=True)
    selected = pick_top_errors(full, topk=args.topk)

    keep_cols = [
        "error_type",
        "error_confidence",
        "fold",
        "tstar",
        "oct_id",
        "hospital",
        "age",
        "hpv_status",
        "tct_result",
        "pathology_class",
        "y_true",
        "prob_positive",
        "y_pred_tstar",
        "uncertainty",
        "image_folder",
    ]
    selected = selected[keep_cols].sort_values(["error_type", "error_confidence"], ascending=[True, False])
    out_csv = os.path.join(out_dir, "baseline_external_high_conf_error_cases.csv")
    selected.to_csv(out_csv, index=False)

    # 同时导出按折分层版本，便于讨论章节挑案例
    out_by_fold = []
    for fold in FOLDS:
        fold_sel = pick_top_errors(full[full["fold"] == fold], topk=min(3, args.topk))
        out_by_fold.append(fold_sel[keep_cols])
    by_fold_df = pd.concat(out_by_fold, axis=0, ignore_index=True)
    out_csv_fold = os.path.join(out_dir, "baseline_external_high_conf_error_cases_by_fold.csv")
    by_fold_df.to_csv(out_csv_fold, index=False)

    print("[完成] 输出:")
    print(" -", out_csv)
    print(" -", out_csv_fold)


if __name__ == "__main__":
    main()
