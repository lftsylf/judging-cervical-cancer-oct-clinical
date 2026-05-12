#!/usr/bin/env python3
"""
Left–right panel: PR (left) + ROC (right).
Outputs: figures/paper/figure2_3_external_combined.{pdf,png}
Run from project root: python scripts/plot_figure2_3_combined.py
"""
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

# scripts/ as import root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import plot_figure2_roc as fig2  # noqa: E402
import plot_figure3_pr as fig3  # noqa: E402

PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def main():
    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    for ax in axes:
        ax.set_facecolor("white")

    # Left: PR (better visual emphasis as requested)
    fig3.draw_pr_curves(axes[0])
    fig3.apply_pr_legend(axes[0])
    axes[0].set_title(
        "(A) External Multi-Center Evaluation PR Curves",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Right: ROC
    fig2.draw_curves(axes[1])
    fig2.apply_legend(axes[1])
    axes[1].set_title(
        "(B) External Multi-Center Evaluation ROC Curves",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    stem = os.path.join(PROJECT_ROOT, "figures", "paper", "figure2_3_external_combined")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    plt.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {stem}.pdf / .png")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
