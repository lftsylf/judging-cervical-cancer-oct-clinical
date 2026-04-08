"""
LOHO 多医院结果汇总：读取 outputs/*/logs/training_history.json，生成汇总表与简易图。

用法:
    python scripts/visualize_loho_results.py
"""

import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "DejaVu Sans Mono", "Bitstream Vera Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
SAVE_DIR = os.path.join(OUTPUTS_DIR, "loho_visualization")

METRICS = [
    "auc_roc",
    "auc_pr",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "mcc",
]


def safe_get(d: Dict, key: str, default=float("nan")):
    return d.get(key, default)


def load_histories(base_dir: str) -> Dict[str, List[Dict]]:
    """扫描 base_dir 下各医院子目录中的 training_history.json。"""
    histories = {}
    for name in sorted(os.listdir(base_dir)):
        hospital_dir = os.path.join(base_dir, name)
        if not os.path.isdir(hospital_dir):
            continue
        history_path = os.path.join(hospital_dir, "logs", "training_history.json")
        if not os.path.exists(history_path):
            continue
        with open(history_path, "r", encoding="utf-8") as f:
            histories[name] = json.load(f)
    return histories


def get_best_epoch_record(history: List[Dict]) -> Tuple[int, Dict]:
    """按 val.auc_roc 最大取最佳 epoch。"""
    best_idx = 0
    best_val_auc = -1.0
    for i, item in enumerate(history):
        val_auc = safe_get(item.get("val", {}), "auc_roc", -1.0)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_idx = i
    return best_idx, history[best_idx]


def save_summary_csv(histories: Dict[str, List[Dict]], save_path: str) -> List[Dict]:
    rows = []
    for hospital, history in histories.items():
        _, best_item = get_best_epoch_record(history)
        val = best_item.get("val", {})
        row = {
            "hospital": hospital,
            "best_epoch": best_item.get("epoch"),
            "num_epochs": len(history),
        }
        for m in METRICS:
            row[m] = safe_get(val, m)
        rows.append(row)

    rows.sort(key=lambda x: x.get("auc_roc", float("nan")), reverse=True)
    fieldnames = ["hospital", "best_epoch", "num_epochs"] + METRICS
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def plot_auc_bar(rows: List[Dict], save_path: str):
    hospitals = [r["hospital"] for r in rows]
    auc_values = [r["auc_roc"] for r in rows]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(hospitals, auc_values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Best val AUC-ROC")
    plt.title("LOHO by hospital (best epoch by val AUC-ROC)")
    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, auc_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_val_auc_curves(histories: Dict[str, List[Dict]], save_path: str):
    plt.figure(figsize=(9, 5))
    for hospital, history in sorted(histories.items()):
        epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
        val_auc = [safe_get(h.get("val", {}), "auc_roc") for h in history]
        plt.plot(epochs, val_auc, label=hospital, linewidth=1.8)

    plt.ylim(0.0, 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC-ROC")
    plt.title("Validation AUC-ROC vs epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"【扫描目录】{OUTPUTS_DIR}")
    histories = load_histories(OUTPUTS_DIR)
    if not histories:
        raise FileNotFoundError("未在 outputs/*/logs/ 下找到 training_history.json，请确认已训练并保存日志。")

    summary_csv = os.path.join(SAVE_DIR, "summary.csv")
    auc_bar_png = os.path.join(SAVE_DIR, "auc_bar.png")
    val_auc_curve_png = os.path.join(SAVE_DIR, "val_auc_curves.png")

    rows = save_summary_csv(histories, summary_csv)
    plot_auc_bar(rows, auc_bar_png)
    plot_val_auc_curves(histories, val_auc_curve_png)

    print("【完成】已生成:")
    print(f"  - 汇总表: {summary_csv}")
    print(f"  - AUC 柱状图: {auc_bar_png}")
    print(f"  - 验证 AUC 曲线: {val_auc_curve_png}")
    print(f"【医院数】{len(histories)}")


if __name__ == "__main__":
    main()
