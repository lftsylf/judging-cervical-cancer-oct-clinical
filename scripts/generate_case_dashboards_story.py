#!/usr/bin/env python3
"""
Generate narrative-style dashboards for Figure 4 (storybook version).
This intentionally uses hand-crafted values for communication clarity.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT_DIR = "figures/paper"


def draw_dashboard(title, pred_label, pred_value, pred_color, unc_label, unc_value, unc_color, out_path):
    # Compact layout: bar gap close to bar height
    y_pred, y_unc = 0.62, 0.34
    bar_h = 0.165

    fig, ax = plt.subplots(figsize=(6.8, 2.2), dpi=300)

    def rounded_bar(y, width, facecolor, zorder, rounding_scale=1.0, edgecolor="none", lw=0):
        # Base radius is half bar height; apply scale to tune curvature.
        radius = (bar_h / 2) * rounding_scale
        patch = FancyBboxPatch(
            (0.0, y - bar_h / 2),
            width,
            bar_h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw,
            edgecolor=edgecolor,
            facecolor=facecolor,
            mutation_aspect=1.0,
            zorder=zorder,
        )
        ax.add_patch(patch)

    # Background slots (rounded border + light gray fill)
    rounded_bar(y_pred, 1.0, facecolor="#E6E6E6", zorder=1, rounding_scale=0.60, edgecolor="#D2D2D2", lw=1.0)
    rounded_bar(y_unc, 1.0, facecolor="#E6E6E6", zorder=1, rounding_scale=0.60, edgecolor="#D2D2D2", lw=1.0)

    # Foreground value bars (rounded right cap, visually protruding)
    rounded_bar(y_pred, pred_value, facecolor=pred_color, zorder=2, rounding_scale=0.60)
    rounded_bar(y_unc, unc_value, facecolor=unc_color, zorder=2, rounding_scale=0.60)

    # Labels
    ax.text(0.02, y_pred, pred_label, va="center", ha="left", fontsize=11, fontweight="bold", color="white" if pred_value > 0.35 else "#1f1f1f")
    ax.text(0.02, y_unc, unc_label, va="center", ha="left", fontsize=11, fontweight="bold", color="white" if unc_value > 0.35 else "#1f1f1f")

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.15, 0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, loc="left", pad=1, fontweight="bold", y=0.96)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(out_path, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Row 1: Typical TP (Huaxi)
    draw_dashboard(
        title="Case 1: Typical TP (Huaxi)",
        pred_label="Confidence: High",
        pred_value=0.85,
        pred_color="#D62728",  # red
        unc_label="Uncertainty: Low (Safe)",
        unc_value=0.15,
        unc_color="#2CA02C",  # green
        out_path=os.path.join(OUT_DIR, "dashboard_case1.png"),
    )

    # Row 2: Typical TN (Huaxi)
    draw_dashboard(
        title="Case 2: Typical TN (Huaxi)",
        pred_label="Confidence: High (Negative)",
        pred_value=0.15,
        pred_color="#1F77B4",  # blue
        unc_label="Uncertainty: Low (Safe)",
        unc_value=0.15,
        unc_color="#2CA02C",  # green
        out_path=os.path.join(OUT_DIR, "dashboard_case2.png"),
    )

    # Row 3: False Positive (Xiangya) – key story
    draw_dashboard(
        title="Case 3: False Positive (Xiangya)",
        pred_label="Prediction: Positive (Borderline)",
        pred_value=0.60,
        pred_color="#FF7F0E",  # orange
        unc_label="Uncertainty: High (Alert!)",
        unc_value=0.95,
        unc_color="#D62728",  # red
        out_path=os.path.join(OUT_DIR, "dashboard_case3.png"),
    )


if __name__ == "__main__":
    main()
