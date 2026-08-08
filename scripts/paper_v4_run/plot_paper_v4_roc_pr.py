#!/usr/bin/env python3
"""
paper_v4 / baseline2：外部 pooled ROC / PR 曲线（5-seed 均值±std 阴影）。

输出:
  figures/paper_v4/fig_ext_roc.{pdf,png}
  figures/paper_v4/fig_ext_pr.{pdf,png}
  figures/paper_v4/fig_ext_roc_pr_combined.{pdf,png}

用法:
  python scripts/paper_v4_run/plot_paper_v4_roc_pr.py
  python scripts/paper_v4_run/plot_paper_v4_roc_pr.py --which comparison
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.paper_v4_run.paper_v4_registry import (  # noqa: E402
    OUT_FIGURES,
    SEEDS,
    Experiment,
    experiments_for_tables,
)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MEAN_X = np.linspace(0, 1, 301)

# 固定配色：Ours 红粗线置顶
COLORS: Dict[str, str] = {
    "Ours (R50)": "#C0392B",
    "Baseline (12×page1)": "#7F8C8D",
    "ABMIL": "#2980B9",
    "DSMIL": "#27AE60",
    "UBIX": "#8E44AD",
    "WMA loss": "#D35400",
    "w/o Mean (→Max)": "#16A085",
    "w/o EMA": "#F39C12",
    "w/o UWA (equal)": "#1ABC9C",
    "w/o FrameAux": "#34495E",
    "Ours-ConvNeXt-T": "#95A5A6",
}


def load_external(run_dir: str, seed: int) -> Optional[pd.DataFrame]:
    path = os.path.join(
        run_dir, f"seed_{seed}", "logs", "external_sample_predictions.csv"
    )
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def mean_curve_roc(run_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    tprs: List[np.ndarray] = []
    aucs: List[float] = []
    for seed in SEEDS:
        df = load_external(run_dir, seed)
        if df is None:
            continue
        y = df["y_true"].to_numpy(dtype=np.int32)
        p = df["prob_positive"].to_numpy(dtype=np.float64)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        tprs.append(np.interp(MEAN_X, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(float(auc(fpr, tpr)))
    if not tprs:
        raise FileNotFoundError(f"No external ROC data in {run_dir}")
    stack = np.vstack(tprs)
    mean_tpr = stack.mean(axis=0)
    std_tpr = stack.std(axis=0, ddof=1) if len(tprs) > 1 else np.zeros_like(mean_tpr)
    mean_tpr[-1] = 1.0
    return MEAN_X, mean_tpr, std_tpr, float(np.mean(aucs)), float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0), len(aucs)


def mean_curve_pr(run_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    precs: List[np.ndarray] = []
    aps: List[float] = []
    for seed in SEEDS:
        df = load_external(run_dir, seed)
        if df is None:
            continue
        y = df["y_true"].to_numpy(dtype=np.int32)
        p = df["prob_positive"].to_numpy(dtype=np.float64)
        if len(np.unique(y)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        # interp on recall grid descending→ascending for np.interp
        rec_r = rec[::-1]
        prec_r = prec[::-1]
        precs.append(np.interp(MEAN_X, rec_r, prec_r))
        aps.append(float(average_precision_score(y, p)))
    if not precs:
        raise FileNotFoundError(f"No external PR data in {run_dir}")
    stack = np.vstack(precs)
    mean_p = stack.mean(axis=0)
    std_p = stack.std(axis=0, ddof=1) if len(precs) > 1 else np.zeros_like(mean_p)
    return MEAN_X, mean_p, std_p, float(np.mean(aps)), float(np.std(aps, ddof=1) if len(aps) > 1 else 0.0), len(aps)


def select_exps(which: str) -> List[Experiment]:
    all_exps = experiments_for_tables(include_appendix=True)
    if which == "all":
        return all_exps
    if which == "comparison":
        return [e for e in all_exps if e.paper_table in ("main_cmp",)]
    if which == "ablation":
        # include Ours
        return [e for e in all_exps if e.paper_table in ("main_ablation",) or e.key == "ours"]
    raise ValueError(which)


def plot_roc(ax, exps: List[Experiment], root: str) -> None:
    for exp in exps:
        run_dir = os.path.join(root, exp.rel_dir)
        try:
            x, y, s, m, sd, n = mean_curve_roc(run_dir)
        except FileNotFoundError:
            print(f"[SKIP ROC] {exp.display}")
            continue
        color = COLORS.get(exp.display, "#333333")
        lw = 3.0 if exp.key == "ours" else 1.8
        z = 10 if exp.key == "ours" else 5
        label = f"{exp.display} ({m:.3f}±{sd:.3f}, n={n})"
        ax.plot(x, y, color=color, lw=lw, zorder=z, label=label)
        ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), color=color, alpha=0.12, zorder=z - 1)
    ax.plot([0, 1], [0, 1], ls="--", color="#BDC3C7", lw=1.0, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("External pooled ROC (Huaxi + Liaoning)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.set_aspect("equal", adjustable="box")


def plot_pr(ax, exps: List[Experiment], root: str) -> None:
    # prevalence reference from first available Ours seed
    prev = None
    for exp in exps:
        if exp.key != "ours":
            continue
        df = load_external(os.path.join(root, exp.rel_dir), SEEDS[0])
        if df is not None:
            prev = float(df["y_true"].mean())
            break
    for exp in exps:
        run_dir = os.path.join(root, exp.rel_dir)
        try:
            x, y, s, m, sd, n = mean_curve_pr(run_dir)
        except FileNotFoundError:
            print(f"[SKIP PR] {exp.display}")
            continue
        color = COLORS.get(exp.display, "#333333")
        lw = 3.0 if exp.key == "ours" else 1.8
        z = 10 if exp.key == "ours" else 5
        label = f"{exp.display} ({m:.3f}±{sd:.3f}, n={n})"
        ax.plot(x, y, color=color, lw=lw, zorder=z, label=label)
        ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), color=color, alpha=0.12, zorder=z - 1)
    if prev is not None:
        ax.axhline(prev, ls="--", color="#BDC3C7", lw=1.0, label=f"Prevalence={prev:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("External pooled PR (Huaxi + Liaoning)")
    ax.legend(loc="lower left", fontsize=8, frameon=True)


def save_fig(fig, out_stem: str) -> None:
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
    print(f"[OK] {out_stem}.{{pdf,png}}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--which",
        choices=("comparison", "ablation", "all"),
        default="comparison",
        help="comparison=主对比表行；ablation=消融；all=全部",
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root
    exps = select_exps(args.which)
    out_dir = os.path.join(root, OUT_FIGURES)
    tag = args.which

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    plot_roc(ax, exps, root)
    save_fig(fig, os.path.join(out_dir, f"fig_ext_roc_{tag}"))

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    plot_pr(ax, exps, root)
    save_fig(fig, os.path.join(out_dir, f"fig_ext_pr_{tag}"))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6))
    plot_roc(axes[0], exps, root)
    plot_pr(axes[1], exps, root)
    axes[0].set_title("(A) External ROC")
    axes[1].set_title("(B) External PR")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"fig_ext_roc_pr_combined_{tag}"))


if __name__ == "__main__":
    main()
