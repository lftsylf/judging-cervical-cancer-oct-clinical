"""
Figure 4：Baseline 辽宁中心阈值解耦动态分析（Development 标定 vs External 迁移）。

读取 development / external_sample_predictions.csv，扫描阈值 t∈[0,1] 步长 0.01，
绘制 Sensitivity、Specificity、MCC 随决策阈值变化的曲线。

用法:
    python scripts/figure4_threshold_decoupling_liaoning.py
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 学术经典配色（Tab10 系）
COLOR_SENS = "#D62728"
COLOR_SPEC = "#1F77B4"
COLOR_MCC = "#2CA02C"
CURVE_LW = 2.5
CURVE_ALPHA = 0.85

# 略增内边距，避免字与曲线“贴在一起”；右图建议配合 axes 坐标错位放置
TEXT_BBOX = dict(
    boxstyle="round,pad=0.45",
    facecolor="white",
    edgecolor="none",
    alpha=0.92,
)


def _set_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "figure.dpi": 150,
        }
    )


def style_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)


def compute_metrics_vs_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_min: float = 0.0,
    t_max: float = 1.0,
    step: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对阈值网格 t，若 prob >= t 判为阳性，返回 (thresholds, sensitivity, specificity, mcc)。
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = int(round((t_max - t_min) / step)) + 1
    thresholds = t_min + step * np.arange(n, dtype=np.float64)
    thresholds = np.clip(np.round(thresholds, 6), t_min, t_max)

    sens_list: list[float] = []
    spec_list: list[float] = []
    mcc_list: list[float] = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens_list.append(float(sens))
        spec_list.append(float(spec))
        mcc_list.append(float(matthews_corrcoef(y_true, y_pred)))

    return thresholds, np.array(sens_list), np.array(spec_list), np.array(mcc_list)


def load_predictions(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "y_true" not in df.columns or "prob_positive" not in df.columns:
        raise ValueError(f"{csv_path} 需包含 y_true, prob_positive")
    return df["y_true"].to_numpy(dtype=np.int32), df["prob_positive"].to_numpy(dtype=np.float64)


def plot_panel_metrics(
    ax: plt.Axes,
    thresholds: np.ndarray,
    sens: np.ndarray,
    spec: np.ndarray,
    mcc: np.ndarray,
    title: str,
    vlines: List[dict],
    annotations: List[dict],
) -> None:
    ax.set_axisbelow(True)

    y_hi = float(np.nanmax([sens.max(), spec.max(), mcc.max(), 1.0]))
    y_lo = float(np.nanmin([sens.min(), spec.min(), mcc.min(), 0.0]))
    pad = 0.04 * (y_hi - y_lo + 1e-6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.grid(True, linestyle="--", alpha=0.4, color="gray", zorder=0)

    ax.plot(
        thresholds,
        sens,
        color=COLOR_SENS,
        linewidth=CURVE_LW,
        alpha=CURVE_ALPHA,
        label="Sensitivity",
        zorder=2,
    )
    ax.plot(
        thresholds,
        spec,
        color=COLOR_SPEC,
        linewidth=CURVE_LW,
        alpha=CURVE_ALPHA,
        label="Specificity",
        zorder=2,
    )
    ax.plot(
        thresholds,
        mcc,
        color=COLOR_MCC,
        linewidth=CURVE_LW,
        alpha=CURVE_ALPHA,
        label="MCC",
        zorder=2,
    )

    for vl in vlines:
        ax.axvline(
            vl["x"],
            color=vl["color"],
            linestyle="--",
            linewidth=2.0,
            alpha=vl.get("alpha", 0.95),
            zorder=3,
        )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title(title, pad=8)
    style_spines(ax)

    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin + 1e-9
    for ann in annotations:
        if ann.get("kind") == "annotate":
            ax.annotate(
                ann["text"],
                xy=tuple(ann["xy"]),
                xytext=tuple(ann["xytext"]),
                textcoords="data",
                fontsize=ann.get("fontsize", 9),
                ha=ann.get("ha", "center"),
                va=ann.get("va", "center"),
                bbox=TEXT_BBOX,
                arrowprops=ann.get("arrowprops", {}),
                zorder=6,
                clip_on=False,
            )
            continue
        if ann.get("coords") == "axes":
            ax.text(
                float(ann["x_axes"]),
                float(ann["y_axes"]),
                ann["text"],
                transform=ax.transAxes,
                fontsize=ann.get("fontsize", 9),
                ha=ann.get("ha", "center"),
                va=ann.get("va", "center"),
                bbox=TEXT_BBOX,
                zorder=6,
                clip_on=False,
            )
            continue
        # 数据坐标：y_frac 为 y 轴可视范围内归一化高度
        y_frac = float(ann["y_frac"])
        y_text = ymin + y_frac * y_span
        x_text = float(ann["x_text"])
        ax.text(
            x_text,
            y_text,
            ann["text"],
            fontsize=ann.get("fontsize", 9),
            ha=ann.get("ha", "center"),
            va=ann.get("va", "center"),
            bbox=TEXT_BBOX,
            zorder=6,
            clip_on=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4: 辽宁 Baseline 阈值动态分析")
    parser.add_argument(
        "--baseline-root",
        default="all_outputs/baseline_outputs",
        help="Baseline 输出根（相对项目根）",
    )
    parser.add_argument("--fold", default="liaoning", help="医院子目录名")
    parser.add_argument("--t-star", type=float, default=0.491, dest="t_star", help="标定的 t*")
    parser.add_argument("--default-threshold", type=float, default=0.5, help="右图默认阈值线位置")
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--out-dir", default="figures/paper")
    parser.add_argument("--out-name", default="figure4_threshold_decoupling_liaoning.pdf")
    args = parser.parse_args()

    logs = os.path.join(PROJECT_ROOT, args.baseline_root, args.fold, "logs")
    dev_csv = os.path.join(logs, "development_sample_predictions.csv")
    ext_csv = os.path.join(logs, "external_sample_predictions.csv")
    if not os.path.isfile(dev_csv) or not os.path.isfile(ext_csv):
        raise FileNotFoundError(f"需要存在:\n{dev_csv}\n{ext_csv}")

    y_dev, p_dev = load_predictions(dev_csv)
    y_ext, p_ext = load_predictions(ext_csv)

    t_grid, s_dev, sp_dev, m_dev = compute_metrics_vs_threshold(
        y_dev, p_dev, step=args.threshold_step
    )
    _, s_ext, sp_ext, m_ext = compute_metrics_vs_threshold(
        y_ext, p_ext, step=args.threshold_step
    )

    _set_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.35))

    t_star = args.t_star
    t_def = args.default_threshold
    left_caption = rf"$t^*={t_star:.3f}$ (Sens $\geq 0.85$)"

    # 左图：箭头从说明框指向红竖线上的锚点，便于对应 t*
    plot_panel_metrics(
        axes[0],
        t_grid,
        s_dev,
        sp_dev,
        m_dev,
        "Development Set (Threshold Calibration)",
        vlines=[{"x": t_star, "color": COLOR_SENS}],
        annotations=[
            {
                "kind": "annotate",
                "text": left_caption,
                "xy": (t_star, 0.56),
                "xytext": (min(t_star + 0.16, 0.78), 0.86),
                "ha": "left",
                "va": "center",
                "arrowprops": {
                    "arrowstyle": "-",
                    "color": COLOR_SENS,
                    "lw": 1.0,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            }
        ],
    )

    # 右图：锚点落在各自竖线上，文字略偏开并带箭头，避免与阶跃曲线重叠又能看出对应关系
    plot_panel_metrics(
        axes[1],
        t_grid,
        s_ext,
        sp_ext,
        m_ext,
        "External Set (Threshold Migration)",
        vlines=[
            {"x": t_def, "color": "black"},
            {"x": t_star, "color": COLOR_SENS},
        ],
        annotations=[
            {
                "kind": "annotate",
                "text": "Default Threshold",
                "xy": (t_def, 0.52),
                "xytext": (min(t_def + 0.065, 0.92), 0.78),
                "ha": "left",
                "va": "center",
                "arrowprops": {
                    "arrowstyle": "-",
                    "color": "0.2",
                    "lw": 1.0,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            },
            {
                "kind": "annotate",
                "text": r"Calibrated $t^*$",
                "xy": (t_star, 0.48),
                "xytext": (max(t_star - 0.21, 0.04), 0.30),
                "ha": "left",
                "va": "center",
                "arrowprops": {
                    "arrowstyle": "-",
                    "color": COLOR_SENS,
                    "lw": 1.0,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
            },
        ],
    )

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=0.9,
        fontsize=10,
    )

    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.07, right=0.98, wspace=0.28)

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    fig.savefig(
        out_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=(leg,),
        pad_inches=0.05,
    )
    plt.close(fig)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
