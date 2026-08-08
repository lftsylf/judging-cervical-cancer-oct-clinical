#!/usr/bin/env python3
"""
paper_v4 / baseline2：外测 ECE + Brier（后处理，不重训）。

协议：只报 **华西 / 辽宁 / Ext pooled**（湘雅是内部 Val，不进本图主面板）。
概率列：``prob_positive``（患者级连续分）。

输出:
  outputs/paper_v4/tables/baseline2_ece_brier_*.csv
  outputs/paper_v4/tables/PAPER_ECE_BRIER_baseline2.md
  figures/paper_v4/fig_reliability_huaxi_liaoning.{pdf,png}
  figures/paper_v4/fig_ece_brier_bars.{pdf,png}

用法:
  python scripts/paper_v4_run/analyze_paper_v4_ece_brier.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.paper_v4_run.paper_v4_registry import (  # noqa: E402
    OUT_FIGURES,
    OUT_TABLES,
    SEEDS,
    Experiment,
    experiments_for_tables,
)
from scripts.youden_threshold_utils import fmt_mean_std, mean_std  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

SPLITS = (
    ("external", "Ext pooled"),
    ("external_huaxi", "Huaxi"),
    ("external_liaoning", "Liaoning"),
)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width ECE on positive-class probability (binary)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def load_split(run_dir: str, seed: int, split: str) -> Optional[pd.DataFrame]:
    path = os.path.join(run_dir, f"seed_{seed}", "logs", f"{split}_sample_predictions.csv")
    if not os.path.isfile(path):
        if split in ("external_huaxi", "external_liaoning"):
            pooled = os.path.join(
                run_dir, f"seed_{seed}", "logs", "external_sample_predictions.csv"
            )
            if not os.path.isfile(pooled):
                return None
            print(f"[WARN] missing {split}, fallback pooled+center ({run_dir} seed={seed})")
            df = pd.read_csv(pooled)
            key = "huaxi" if split.endswith("huaxi") else "liaoning"
            return df[df["center"].astype(str).str.lower() == key].copy()
        return None
    return pd.read_csv(path)


def collect_rows(exps: List[Experiment], root: str, n_bins: int) -> pd.DataFrame:
    rows: List[dict] = []
    for exp in exps:
        run_dir = os.path.join(root, exp.rel_dir)
        if not os.path.isdir(run_dir):
            print(f"[SKIP] {exp.display}")
            continue
        for seed in SEEDS:
            for split, label in SPLITS:
                df = load_split(run_dir, seed, split)
                if df is None or df.empty:
                    continue
                y = df["y_true"].to_numpy(dtype=np.int32)
                p = df["prob_positive"].to_numpy(dtype=np.float64)
                if len(np.unique(y)) < 2:
                    brier = float("nan")
                    ece = float("nan")
                else:
                    brier = float(brier_score_loss(y, p))
                    ece = expected_calibration_error(y, p, n_bins=n_bins)
                rows.append(
                    {
                        "model_key": exp.key,
                        "model": exp.display,
                        "role": exp.role,
                        "paper_table": exp.paper_table,
                        "seed": seed,
                        "split": split,
                        "split_label": label,
                        "n": int(len(df)),
                        "ece": ece,
                        "brier": brier,
                        "prob_mean": float(np.mean(p)),
                        "prob_std": float(np.std(p)),
                    }
                )
        print(f"[OK] {exp.display}")
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, split), g in detail.groupby(["model", "split"], sort=False):
        entry = {
            "model": model,
            "split": split,
            "split_label": g["split_label"].iloc[0],
            "n_seeds": int(g["seed"].nunique()),
            "role": g["role"].iloc[0],
            "paper_table": g["paper_table"].iloc[0],
        }
        for k in ("ece", "brier"):
            m, s = mean_std([float(x) for x in g[k].tolist()])
            entry[f"{k}_mean"] = m
            entry[f"{k}_std"] = s
            entry[f"{k}_fmt"] = fmt_mean_std(m, s)
        out.append(entry)
    return pd.DataFrame(out)


def markdown_table(df: pd.DataFrame, cols: List[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def pivot_metric(summary: pd.DataFrame, metric: str, models: List[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        sub = summary[summary["model"] == model]
        entry = {"Model": model}
        for split, label in SPLITS:
            r = sub[sub["split"] == split]
            entry[label] = r[f"{metric}_fmt"].iloc[0] if not r.empty else "—"
        rows.append(entry)
    return pd.DataFrame(rows)


def write_md(path: str, summary: pd.DataFrame, n_bins: int) -> None:
    cmp_models = summary.loc[
        summary["paper_table"].isin(["main_cmp"]), "model"
    ].drop_duplicates().tolist()
    # ensure Ours first
    if "Ours (R50)" in cmp_models:
        cmp_models = ["Ours (R50)"] + [m for m in cmp_models if m != "Ours (R50)"]

    ece_tbl = pivot_metric(summary[summary["model"].isin(cmp_models)], "ece", cmp_models)
    brier_tbl = pivot_metric(summary[summary["model"].isin(cmp_models)], "brier", cmp_models)

    text = "\n".join(
        [
            "# paper_v4 / baseline2：外测 ECE 与 Brier（后处理）",
            "",
            "> 协议：湘雅**内训**；本表/图仅含 **华西、辽宁、Ext pooled**（湘雅不进外测校准主结果）。",
            f"> ECE：等宽 {n_bins} bins；Brier：`mean((p−y)²)`；5 seeds mean±std。",
            "> **不重训**；用现有 `prob_positive`。越低越好。",
            "> 图：`figures/paper_v4/fig_reliability_huaxi_liaoning.*`，`fig_ece_brier_bars.*`",
            "",
            "## ECE（越低越好）",
            "",
            markdown_table(ece_tbl, list(ece_tbl.columns)),
            "",
            "## Brier score（越低越好）",
            "",
            markdown_table(brier_tbl, list(brier_tbl.columns)),
            "",
            "## 写进论文时的表述建议",
            "",
            "- 图2（案例/KDE）改协议标注后仍作示意；本表回应审稿人「补充 ECE/Brier」。",
            "- 主张收窄：EDL-u 用于 **UWA 帧聚合**；ECE/Brier 报告概率校准，**不**声称已完成完整 OOD benchmark。",
            "- **禁止**把湘雅难例图注改成华西/辽宁；中心标签必须与真实数据一致。",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_reliability_ours(root: str, ours: Experiment, out_stem: str, n_bins: int) -> None:
    run_dir = os.path.join(root, ours.rel_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    panels = [
        ("external_huaxi", "Huaxi (external)", axes[0]),
        ("external_liaoning", "Liaoning (external)", axes[1]),
    ]
    for split, title, ax in panels:
        # pool 5 seeds for denser reliability curve; also compute per-seed ECE for annotation
        ys, ps, eces = [], [], []
        for seed in SEEDS:
            df = load_split(run_dir, seed, split)
            if df is None:
                continue
            y = df["y_true"].to_numpy(dtype=np.int32)
            p = df["prob_positive"].to_numpy(dtype=np.float64)
            ys.append(y)
            ps.append(p)
            eces.append(expected_calibration_error(y, p, n_bins=n_bins))
        if not ys:
            ax.set_title(title + " (missing)")
            continue
        y_all = np.concatenate(ys)
        p_all = np.concatenate(ps)
        frac, mean_p = calibration_curve(y_all, p_all, n_bins=n_bins, strategy="uniform")
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Perfect")
        ax.plot(mean_p, frac, "o-", color="#C0392B", lw=2.0, label="Ours (pooled seeds)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(
            f"{title}\nECE={np.mean(eces):.3f}±{np.std(eces, ddof=1):.3f} (per-seed)"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
    fig.suptitle("Reliability diagrams (external only; Xiangya not shown)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_stem}.{{pdf,png}}")


def plot_bars(summary: pd.DataFrame, out_stem: str) -> None:
    models = summary.loc[
        summary["paper_table"] == "main_cmp", "model"
    ].drop_duplicates().tolist()
    if "Ours (R50)" in models:
        models = ["Ours (R50)"] + [m for m in models if m != "Ours (R50)"]

    # Ext pooled bars for ECE and Brier
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, metric, title in (
        (axes[0], "ece", "ECE (Ext pooled)"),
        (axes[1], "brier", "Brier (Ext pooled)"),
    ):
        means, stds, labels = [], [], []
        for m in models:
            r = summary[(summary["model"] == m) & (summary["split"] == "external")]
            if r.empty:
                continue
            means.append(float(r[f"{metric}_mean"].iloc[0]))
            stds.append(float(r[f"{metric}_std"].iloc[0]))
            labels.append(m)
        x = np.arange(len(labels))
        colors = ["#C0392B" if lb == "Ours (R50)" else "#7F8C8D" for lb in labels]
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title(title + " ↓ better")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Calibration metrics on external pooled (Huaxi+Liaoning)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_stem}.{{pdf,png}}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--include-appendix", action="store_true", default=False)
    args = parser.parse_args()
    root = args.project_root

    exps = [
        e
        for e in experiments_for_tables(include_appendix=args.include_appendix)
        if e.paper_table in ("main_cmp", "main_ablation")
        or (args.include_appendix and e.paper_table == "appendix")
    ]
    # focus main comparison + ours for figure
    detail = collect_rows(exps, root, args.n_bins)
    if detail.empty:
        raise SystemExit("No data.")
    summary = summarize(detail)

    out_dir = os.path.join(root, OUT_TABLES)
    fig_dir = os.path.join(root, OUT_FIGURES)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    detail_path = os.path.join(out_dir, "baseline2_ece_brier_detail.csv")
    summary_path = os.path.join(out_dir, "baseline2_ece_brier_summary.csv")
    md_path = os.path.join(out_dir, "PAPER_ECE_BRIER_baseline2.md")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    write_md(md_path, summary, args.n_bins)
    print(f"[OK] {detail_path}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {md_path}")

    ours = next(e for e in exps if e.key == "ours")
    plot_reliability_ours(
        root, ours, os.path.join(fig_dir, "fig_reliability_huaxi_liaoning"), args.n_bins
    )
    plot_bars(summary, os.path.join(fig_dir, "fig_ece_brier_bars"))


if __name__ == "__main__":
    main()
