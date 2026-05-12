#!/usr/bin/env python3
import math
import os
import re
from typing import Dict

import matplotlib.pyplot as plt


MANIFESTS = [
    "case_study_4_panels/1_Huaxi_Easy_TP/candidate_manifest.txt",
    "case_study_4_panels/2_Huaxi_Easy_TN/candidate_manifest.txt",
    "case_study_4_panels/4_Xiangya_False_Positive/candidate_manifest.txt",
]

TARGETS = [
    ("M0008_2021_P0000199", "dashboard_case1.png"),
    ("M0008_2021_P0000314", "dashboard_case2.png"),
    ("M0008_2021_P0001623", "dashboard_case3.png"),
]

OUT_DIR = "figures/paper"


def parse_manifest_line(line: str):
    oct_match = re.search(r"oct_id=([^ ]+)", line)
    prob_match = re.search(r"prob=([0-9]*\.[0-9]+)", line)
    if not oct_match or not prob_match:
        return None
    return oct_match.group(1), float(prob_match.group(1))


def load_probs() -> Dict[str, float]:
    probs: Dict[str, float] = {}
    for path in MANIFESTS:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = parse_manifest_line(line)
                if parsed is None:
                    continue
                oct_id, prob = parsed
                probs[oct_id] = prob
    return probs


def uncertainty_entropy(prob: float) -> float:
    p = min(max(prob, 1e-7), 1 - 1e-7)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def prediction_color(prob: float) -> str:
    return "#D62728" if prob > 0.5 else "#1F77B4"


def uncertainty_color(unc: float) -> str:
    if unc > 0.6:
        return "#D62728"
    if unc >= 0.3:
        return "#FF7F0E"
    return "#2CA02C"


def plot_dashboard(oct_id: str, prob: float, out_path: str) -> None:
    unc = uncertainty_entropy(prob)

    labels = ["Prediction", "Uncertainty"]
    values = [prob, unc]
    colors = [prediction_color(prob), uncertainty_color(unc)]
    notes = [f"Prob: {prob*100:.1f}%", f"Unc: {unc:.2f}"]
    ypos = [1, 0]

    fig, ax = plt.subplots(figsize=(6.6, 2.2), dpi=300)

    # background slots
    for y in ypos:
        ax.barh(y, 1.0, color="#E8E8E8", height=0.35, edgecolor="none", zorder=1)

    # value bars
    for y, v, c in zip(ypos, values, colors):
        ax.barh(y, v, color=c, height=0.35, edgecolor="none", zorder=2)

    # labels on bars
    for y, v, txt in zip(ypos, values, notes):
        tx = min(max(v - 0.01, 0.02), 0.98)
        ha = "right" if v > 0.22 else "left"
        if ha == "left":
            tx = min(v + 0.02, 0.98)
        ax.text(tx, y, txt, va="center", ha=ha, fontsize=10, fontweight="bold", color="white" if v > 0.22 else "#222222")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    ax.set_title(oct_id, fontsize=11, loc="left", pad=6)

    # minimalist UI look
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    plt.tight_layout()
    plt.savefig(out_path, transparent=True, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    probs = load_probs()
    os.makedirs(OUT_DIR, exist_ok=True)

    for oct_id, out_name in TARGETS:
        if oct_id not in probs:
            raise RuntimeError(f"Cannot find prob for {oct_id} in manifests")
        out_path = os.path.join(OUT_DIR, out_name)
        plot_dashboard(oct_id, probs[oct_id], out_path)
        print(f"[OK] {oct_id}: prob={probs[oct_id]:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
