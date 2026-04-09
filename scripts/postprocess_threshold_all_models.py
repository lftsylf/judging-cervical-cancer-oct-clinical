"""
批量阈值迁移后处理：
1) 在 development 集按 Sensitivity >= target_sens 选择最大阈值 t*
2) 将 t* 固定应用到 external 集
3) 导出各模型各折指标 + 三折均值 + 所有模型总表
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_CONFIGS: Dict[str, str] = {
    "Baseline": "all_outputs/baseline_outputs",
    "ViT-Small": "all_outputs/Comparisons_vit_loho_outputs/vit_small_patch16_224",
    "ViT-Base": "all_outputs/Comparisons_vit_loho_outputs/vit_base_patch16_224",
    "Ablation-EMA": "all_outputs/Ablation_Study/ema_outputs",
    "Ablation-Aux": "all_outputs/Ablation_Study/aux_50_outputs",
    "Ablation-UDA": "all_outputs/Ablation_Study/uda_50_outputs",
}

FOLDS = ["xiangya", "huaxi", "liaoning"]


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "BACC": float(balanced_accuracy_score(y_true, y_pred)),
        "Accuracy": float(acc),
        "PPV": float(ppv),
        "NPV": float(npv),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def find_tstar_max_sens(y_true: np.ndarray, y_prob: np.ndarray, target_sens: float, step: float) -> float:
    # 和 baseline 现有后处理一致：网格扫描，选满足 Sens>=target 的最大阈值
    thresholds = np.round(np.arange(0.0, 1.0 + 1e-12, step), 6)
    chosen = 0.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if sens >= target_sens:
            chosen = float(thr)
    return chosen


def read_preds(model_root: str, fold: str) -> Dict[str, pd.DataFrame]:
    logs_dir = os.path.join(PROJECT_ROOT, model_root, fold, "logs")
    dev_csv = os.path.join(logs_dir, "development_sample_predictions.csv")
    ext_csv = os.path.join(logs_dir, "external_sample_predictions.csv")
    if not os.path.exists(dev_csv) or not os.path.exists(ext_csv):
        raise FileNotFoundError(f"缺少预测文件: {dev_csv} 或 {ext_csv}")
    return {"dev": pd.read_csv(dev_csv), "ext": pd.read_csv(ext_csv)}


def main():
    parser = argparse.ArgumentParser(description="批量阈值迁移后处理")
    parser.add_argument("--target-sensitivity", type=float, default=0.85)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument(
        "--output-dir",
        default="all_outputs/postprocess_threshold_all_models",
        help="项目根目录下输出目录",
    )
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows_fold: List[Dict[str, float]] = []

    for model_name, model_root in MODEL_CONFIGS.items():
        for fold in FOLDS:
            data = read_preds(model_root, fold)
            y_dev = data["dev"]["y_true"].values.astype(int)
            p_dev = data["dev"]["prob_positive"].values.astype(float)
            y_ext = data["ext"]["y_true"].values.astype(int)
            p_ext = data["ext"]["prob_positive"].values.astype(float)

            t_star = find_tstar_max_sens(
                y_true=y_dev,
                y_prob=p_dev,
                target_sens=args.target_sensitivity,
                step=args.threshold_step,
            )
            m_ext = metrics_at_threshold(y_ext, p_ext, t_star)
            m_ext.update(
                {
                    "Model": model_name,
                    "Fold": fold,
                    "Rule": f"max threshold with Sens(dev) >= {args.target_sensitivity:.2f}",
                }
            )
            rows_fold.append(m_ext)

    df_fold = pd.DataFrame(rows_fold)
    df_fold = df_fold[
        [
            "Model",
            "Fold",
            "Rule",
            "threshold",
            "ROC-AUC",
            "PR-AUC",
            "Sensitivity",
            "Specificity",
            "MCC",
            "BACC",
            "Accuracy",
            "PPV",
            "NPV",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
    ]
    df_fold.to_csv(os.path.join(out_dir, "external_metrics_by_model_fold_tstar.csv"), index=False)

    metric_cols = ["ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "MCC", "BACC", "Accuracy", "PPV", "NPV"]
    df_mean = df_fold.groupby("Model", as_index=False)[metric_cols + ["threshold"]].mean()
    df_std = df_fold.groupby("Model", as_index=False)[metric_cols + ["threshold"]].std().rename(
        columns={c: f"{c}_std" for c in metric_cols + ["threshold"]}
    )
    df_summary = df_mean.merge(df_std, on="Model", how="left")
    df_summary.to_csv(os.path.join(out_dir, "external_metrics_summary_all_models_tstar.csv"), index=False)

    # 论文主表：每个模型在 external 上按 t* 的均值指标
    paper_cols = ["Model", "threshold", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "MCC", "BACC", "Accuracy"]
    df_paper = df_summary[paper_cols].copy()
    df_paper = df_paper.rename(columns={"threshold": "t* (mean across folds)"})
    df_paper.to_csv(os.path.join(out_dir, "paper_table_external_all_models_tstar.csv"), index=False)

    print("[完成] 输出目录:", out_dir)
    print(" - external_metrics_by_model_fold_tstar.csv")
    print(" - external_metrics_summary_all_models_tstar.csv")
    print(" - paper_table_external_all_models_tstar.csv")


if __name__ == "__main__":
    main()
