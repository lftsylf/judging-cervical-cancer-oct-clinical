#!/usr/bin/env python3
"""
External multi-center PR curves (5 seeds × 3 centers pooled per seed).
Outputs: figures/paper/figure3_pr_external_highres.{pdf,png}
Panel label (B) matches combined figure scripts/plot_figure2_3_combined.py.

Legend shows AUPRC mean ± SD across seeds only (no DeLong p-values): pairwise
significance for ROC-AUC is reported on Figure 2, not duplicated here.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MODEL_CONFIGS = {
    "MUSE": {
        "dir": "outputs/消融实验/outputs_wma_ema_aux",
        "color": "#D62728",
        "linewidth": 3.0,
        "zorder": 10,
    },
    "Baseline": {"dir": "outputs/outputs_baseline", "color": "#7F7F7F", "linewidth": 2.0, "zorder": 5},
    "ViT-Small": {"dir": "outputs/对比试验/outputs_comparison_vit_small", "color": "#1F77B4", "linewidth": 2.0, "zorder": 4},
    "ViT-Base": {"dir": "outputs/对比试验/outputs_comparison_vit_base", "color": "#2CA02C", "linewidth": 2.0, "zorder": 3},
    "Swin-Small": {
        "dir": "outputs/对比试验/outputs_comparison_swin_small",
        "alt_dir": "outputs_comparison_swin_small",
        "color": "#9467BD",
        "linewidth": 2.0,
        "zorder": 2,
    },
    "ConvNeXt-S": {
        "dir": "outputs/对比试验/outputs_comparison_convnext_small",
        "alt_dir": "outputs_comparison_convnext_small",
        "color": "#E377C2",
        "linewidth": 2.0,
        "zorder": 1,
    },
}

CENTERS = ["huaxi", "liaoning", "xiangya"]
SEEDS = [42, 123, 2024, 3407, 114514]
MEAN_RECALL = np.linspace(0, 1, 300)


def resolve_model_dir(config):
    for key in ("dir", "alt_dir"):
        p = config.get(key)
        if p and os.path.isdir(p):
            return p
    return config["dir"]


def pick_columns(df):
    if "y_true" in df.columns:
        y_col = "y_true"
    elif "true_label" in df.columns:
        y_col = "true_label"
    else:
        raise ValueError("Missing y_true/true_label column")

    if "prob_positive" in df.columns:
        p_col = "prob_positive"
    elif "probability_1" in df.columns:
        p_col = "probability_1"
    elif "pred_prob" in df.columns:
        p_col = "pred_prob"
    else:
        raise ValueError("Missing probability column")
    return y_col, p_col


def draw_pr_curves(ax):
    baseline_ratio_values = []

    for model_name, config in MODEL_CONFIGS.items():
        precs = []
        aps = []
        model_dir = resolve_model_dir(config)

        for seed in SEEDS:
            y_true_all = []
            y_score_all = []

            for center in CENTERS:
                csv_path = os.path.join(
                    model_dir, center, f"seed_{seed}", "logs", "external_sample_predictions.csv"
                )
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path)
                y_col, p_col = pick_columns(df)
                y_true_all.extend(df[y_col].astype(int).tolist())
                y_score_all.extend(df[p_col].astype(float).tolist())

            if len(y_true_all) > 0 and len(set(y_true_all)) > 1:
                precision, recall, _ = precision_recall_curve(y_true_all, y_score_all)
                ap = average_precision_score(y_true_all, y_score_all)
                interp_prec = np.interp(MEAN_RECALL, recall[::-1], precision[::-1])
                precs.append(interp_prec)
                aps.append(ap)
                baseline_ratio_values.append(float(np.mean(y_true_all)))

        if len(precs) == 0:
            print(f"[WARN] No valid PR data for {model_name}")
            continue

        mean_prec = np.mean(precs, axis=0)
        std_prec = np.std(precs, axis=0, ddof=1) if len(precs) > 1 else np.zeros_like(mean_prec)
        mean_ap = float(np.mean(aps))
        std_ap = float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0

        precs_upper = np.minimum(mean_prec + std_prec, 1)
        precs_lower = np.maximum(mean_prec - std_prec, 0)

        label_text = f"{model_name} (AUPRC = {mean_ap:.3f} $\\pm$ {std_ap:.3f})"
        if model_name == "MUSE":
            label_text = f"$\\bf{{{model_name}}}$ (AUPRC = {mean_ap:.3f} $\\pm$ {std_ap:.3f})"

        ax.plot(
            MEAN_RECALL,
            mean_prec,
            color=config["color"],
            label=label_text,
            lw=config["linewidth"],
            zorder=config["zorder"],
        )

        if model_name == "MUSE":
            ax.fill_between(
                MEAN_RECALL, precs_lower, precs_upper, color=config["color"], alpha=0.20, zorder=config["zorder"] - 1
            )
        elif model_name == "ViT-Small":
            ax.fill_between(
                MEAN_RECALL, precs_lower, precs_upper, color=config["color"], alpha=0.08, zorder=config["zorder"] - 2
            )

        print(f"{model_name}: AUPRC = {mean_ap:.3f} ± {std_ap:.3f}")

    baseline_ratio = float(np.mean(baseline_ratio_values)) if baseline_ratio_values else 0.1
    ax.plot(
        [0, 1],
        [baseline_ratio, baseline_ratio],
        linestyle="--",
        lw=2,
        color="black",
        alpha=0.5,
        label=f"Random Guess (AUPRC = {baseline_ratio:.3f})",
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=14, fontweight="bold")
    ax.set_title(
        "(B) External Multi-Center Evaluation PR Curves",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(False)
    sns.despine(ax=ax)


def apply_pr_legend(ax):
    legend = ax.legend(loc="lower right", fontsize=10, frameon=True, edgecolor="black", fancybox=False)
    legend.get_frame().set_alpha(0.9)


def main():
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_facecolor("white")

    draw_pr_curves(ax)
    apply_pr_legend(ax)
    plt.tight_layout()

    os.makedirs("figures/paper", exist_ok=True)
    pdf_out = "figures/paper/figure3_pr_external_highres.pdf"
    png_out = "figures/paper/figure3_pr_external_highres.png"
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {pdf_out}")
    print(f"[OK] Saved: {png_out}")


if __name__ == "__main__":
    main()
