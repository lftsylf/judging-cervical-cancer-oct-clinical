"""
单医院：阈值迁移（development 选阈值 → 固定到 external）+ 可选概率校准 + 曲线图。

用法示例:
    python scripts/threshold_calibration_analysis.py --hospital xiangya
    python scripts/threshold_calibration_analysis.py --hospital xiangya --output-root baseline_outputs
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# 图表标签使用英文 + Matplotlib 自带 DejaVu Sans，避免无中文字体时出现方框
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "DejaVu Sans Mono", "Bitstream Vera Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_best_threshold(y_true, y_prob, mode="youden", target_sens=0.9):
    """在 development 上根据 ROC 选阈值。"""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    if mode == "youden":
        scores = tpr - fpr
        idx = int(np.argmax(scores))
        return float(thr[idx])
    if mode == "target_sensitivity":
        valid = np.where(tpr >= target_sens)[0]
        if len(valid) == 0:
            return 0.5
        idx = valid[0]
        return float(thr[idx])
    return 0.5


def metrics_at_threshold(y_true, y_prob, threshold):
    """给定阈值下的混淆矩阵与常用指标。"""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_threshold_table(save_path, rows):
    """CSV 表头保持英文列名，便于脚本读取；终端说明用中文。"""
    fields = [
        "set",
        "calibration",
        "threshold",
        "balanced_accuracy",
        "mcc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def plot_curves(dev_true, dev_prob, ext_true, ext_prob, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    fpr_d, tpr_d, _ = roc_curve(dev_true, dev_prob)
    fpr_e, tpr_e, _ = roc_curve(ext_true, ext_prob)
    auc_d = roc_auc_score(dev_true, dev_prob)
    auc_e = roc_auc_score(ext_true, ext_prob)
    axes[0].plot(fpr_d, tpr_d, label=f"Development AUC={auc_d:.3f}")
    axes[0].plot(fpr_e, tpr_e, label=f"External AUC={auc_e:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    prec_d, rec_d, _ = precision_recall_curve(dev_true, dev_prob)
    prec_e, rec_e, _ = precision_recall_curve(ext_true, ext_prob)
    ap_d = average_precision_score(dev_true, dev_prob)
    ap_e = average_precision_score(ext_true, ext_prob)
    axes[1].plot(rec_d, prec_d, label=f"Development AP={ap_d:.3f}")
    axes[1].plot(rec_e, prec_e, label=f"External AP={ap_e:.3f}")
    axes[1].set_title("Precision-recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_reliability(dev_true, dev_prob, ext_true, ext_prob, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, y, p, title in [
        (axes[0], dev_true, dev_prob, "Reliability (development)"),
        (axes[1], ext_true, ext_prob, "Reliability (external)"),
    ]:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "o-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="阈值迁移与校准分析")
    parser.add_argument("--hospital", required=True, help="医院目录名，如 xiangya")
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="项目根下含 <医院>/logs/ 的目录（默认 outputs；归档用 baseline_outputs）",
    )
    parser.add_argument("--threshold-mode", default="youden", choices=["youden", "target_sensitivity", "fixed_05"])
    parser.add_argument("--target-sensitivity", type=float, default=0.9, help="target_sensitivity 模式下的目标敏感度")
    parser.add_argument("--calibration", default="none", choices=["none", "platt", "isotonic"])
    args = parser.parse_args()

    output_root = os.path.join(PROJECT_ROOT, args.output_root)
    logs_dir = os.path.join(output_root, args.hospital, "logs")
    dev_csv = os.path.join(logs_dir, "development_sample_predictions.csv")
    ext_csv = os.path.join(logs_dir, "external_sample_predictions.csv")
    if not os.path.exists(dev_csv) or not os.path.exists(ext_csv):
        raise FileNotFoundError(
            f"缺少逐样本预测文件，需要同时存在:\n- {dev_csv}\n- {ext_csv}\n"
            "请先完整训练（自动导出）或运行: python scripts/export_sample_predictions.py --hospital <医院>"
        )

    save_dir = os.path.join(logs_dir, "threshold_calibration")
    os.makedirs(save_dir, exist_ok=True)

    dev = pd.read_csv(dev_csv)
    ext = pd.read_csv(ext_csv)
    y_dev = dev["y_true"].values.astype(int)
    p_dev = dev["prob_positive"].values.astype(float)
    y_ext = ext["y_true"].values.astype(int)
    p_ext = ext["prob_positive"].values.astype(float)

    if args.calibration == "platt":
        lr = LogisticRegression(max_iter=500)
        lr.fit(p_dev.reshape(-1, 1), y_dev)
        p_dev_cal = lr.predict_proba(p_dev.reshape(-1, 1))[:, 1]
        p_ext_cal = lr.predict_proba(p_ext.reshape(-1, 1))[:, 1]
    elif args.calibration == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_dev, y_dev)
        p_dev_cal = iso.predict(p_dev)
        p_ext_cal = iso.predict(p_ext)
    else:
        p_dev_cal = p_dev
        p_ext_cal = p_ext

    if args.threshold_mode == "fixed_05":
        threshold = 0.5
    else:
        threshold = get_best_threshold(
            y_dev,
            p_dev_cal,
            mode=args.threshold_mode,
            target_sens=args.target_sensitivity,
        )

    rows = []
    m_dev = metrics_at_threshold(y_dev, p_dev_cal, threshold)
    m_dev.update({"set": "development", "calibration": args.calibration})
    rows.append(m_dev)

    m_ext = metrics_at_threshold(y_ext, p_ext_cal, threshold)
    m_ext.update({"set": "external", "calibration": args.calibration})
    rows.append(m_ext)

    table_path = os.path.join(
        save_dir,
        f"threshold_report_{args.threshold_mode}_{args.calibration}.csv",
    )
    save_threshold_table(table_path, rows)

    rocpr_path = os.path.join(save_dir, f"roc_pr_{args.calibration}.png")
    plot_curves(y_dev, p_dev_cal, y_ext, p_ext_cal, rocpr_path)

    rel_path = os.path.join(save_dir, f"reliability_{args.calibration}.png")
    plot_reliability(y_dev, p_dev_cal, y_ext, p_ext_cal, rel_path)

    mode_cn = {"youden": "约登指数", "target_sensitivity": "目标敏感度", "fixed_05": "固定0.5"}.get(
        args.threshold_mode, args.threshold_mode
    )
    cal_cn = {"none": "无校准", "platt": "Platt", "isotonic": "等渗回归"}.get(args.calibration, args.calibration)

    print("【完成】已生成以下文件:")
    print(f"  - 阈值报告 CSV: {table_path}")
    print(f"  - ROC/PR 图: {rocpr_path}")
    print(f"  - 可靠性图: {rel_path}")
    print(f"【选用阈值】{threshold:.6f}  （阈值策略: {mode_cn}  |  校准: {cal_cn}）")
    print("【说明】development 行 = 在开发集上应用该阈值；external 行 = 同一阈值迁到外部集。")


if __name__ == "__main__":
    main()
