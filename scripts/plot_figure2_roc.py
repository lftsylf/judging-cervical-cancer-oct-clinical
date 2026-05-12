#!/usr/bin/env python3
"""
External multi-center ROC (5 seeds × 3 centers pooled per seed).
Outputs: figures/paper/figure2_roc_external_highres.{pdf,png} (legend lower right).
Panel label (A) matches combined figure scripts/plot_figure2_3_combined.py.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

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
        "p_val": "Ref.",
    },
    "Baseline": {
        "dir": "outputs/outputs_baseline",
        "color": "#7F7F7F",
        "linewidth": 2.0,
        "zorder": 5,
        "p_val": "p=0.002**",
    },
    "ViT-Small": {
        "dir": "outputs/对比试验/outputs_comparison_vit_small",
        "color": "#1F77B4",
        "linewidth": 2.0,
        "zorder": 4,
        "p_val": "p=0.047*",
    },
    "ViT-Base": {
        "dir": "outputs/对比试验/outputs_comparison_vit_base",
        "color": "#2CA02C",
        "linewidth": 2.0,
        "zorder": 3,
        "p_val": "p=0.044*",
    },
    "Swin-Small": {
        "dir": "outputs/对比试验/outputs_comparison_swin_small",
        "alt_dir": "outputs_comparison_swin_small",
        "color": "#9467BD",
        "linewidth": 2.0,
        "zorder": 2,
        "p_val": "p=0.116",
    },
    "ConvNeXt-S": {
        "dir": "outputs/对比试验/outputs_comparison_convnext_small",
        "alt_dir": "outputs_comparison_convnext_small",
        "color": "#E377C2",
        "linewidth": 2.0,
        "zorder": 1,
        "p_val": "p=0.090",
    },
}

CENTERS = ["huaxi", "liaoning", "xiangya"]
SEEDS = [42, 123, 2024, 3407, 114514]
MEAN_FPR = np.linspace(0, 1, 300)


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


def draw_curves(ax):
    for model_name, config in MODEL_CONFIGS.items():
        tprs = []
        aucs = []
        model_dir = resolve_model_dir(config)

        for seed in SEEDS:
            y_true_all = []
            y_score_all = []

            for center in CENTERS:
                csv_path = os.path.join(
                    model_dir,
                    center,
                    f"seed_{seed}",
                    "logs",
                    "external_sample_predictions.csv",
                )
                if not os.path.exists(csv_path):
                    continue

                df = pd.read_csv(csv_path)
                y_col, p_col = pick_columns(df)
                y_true_all.extend(df[y_col].astype(int).tolist())
                y_score_all.extend(df[p_col].astype(float).tolist())

            if len(y_true_all) > 0 and len(set(y_true_all)) > 1:
                fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
                roc_auc = auc(fpr, tpr)
                interp_tpr = np.interp(MEAN_FPR, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)

        if len(tprs) == 0:
            print(f"[WARN] No valid ROC data for {model_name}")
            continue

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0

        std_tpr = np.std(tprs, axis=0, ddof=1) if len(tprs) > 1 else np.zeros_like(mean_tpr)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        label_text = f"{model_name} (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f}) [{config['p_val']}]"
        if model_name == "MUSE":
            # Champion / reference: no pairwise p-value vs self; show Ref. for legend clarity.
            label_text = (
                f"$\\bf{{{model_name}}}$ (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f}) [{config['p_val']}]"
            )

        ax.plot(
            MEAN_FPR,
            mean_tpr,
            color=config["color"],
            label=label_text,
            lw=config["linewidth"],
            zorder=config["zorder"],
        )

        if model_name == "MUSE":
            ax.fill_between(
                MEAN_FPR,
                tprs_lower,
                tprs_upper,
                color=config["color"],
                alpha=0.20,
                zorder=config["zorder"] - 1,
            )
        elif model_name == "ViT-Small":
            ax.fill_between(
                MEAN_FPR,
                tprs_lower,
                tprs_upper,
                color=config["color"],
                alpha=0.08,
                zorder=config["zorder"] - 2,
            )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="black",
        alpha=0.5,
        label="Random Guess (AUC = 0.500)",
    )
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14, fontweight="bold")
    ax.set_title(
        "(A) External Multi-Center Evaluation ROC Curves",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(False)


def apply_legend(ax):
    kw = dict(fontsize=10, frameon=True, edgecolor="black", fancybox=False)
    leg = ax.legend(loc="lower right", **kw)
    leg.get_frame().set_alpha(0.9)


def render_and_save(stem: str):
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.set_facecolor("white")

    draw_curves(ax)
    apply_legend(ax)
    sns.despine(ax=ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(stem), exist_ok=True)
    plt.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {stem}.pdf / .png")


def main():
    render_and_save("figures/paper/figure2_roc_external_highres")


if __name__ == "__main__":
    main()
