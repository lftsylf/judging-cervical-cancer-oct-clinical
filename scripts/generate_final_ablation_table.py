#!/usr/bin/env python3
"""
生成最终消融实验汇总表（Table 2 - LOCO 版）。

处理流程：
1) 对每个变体、每个中心、每个 seed 的 external_sample_predictions.csv，
   使用 Youden Index 主目标 + 非退化解约束重算阈值与指标。
2) 对每个 seed，先在 huaxi/liaoning/xiangya 三中心做算术平均，得到该 seed 的 Overall 指标。
3) 对 5 个 seed 求 Mean ± Std，得到最终表格条目。
4) 以 OptiGenesis (Full) 的 Overall AUC 为基准，计算 Perf. Gap (ΔAUC)。

输出：
- 终端打印 Markdown 表格
- outputs/消融实验/final_ablation_table_loco.md
- outputs/消融实验/final_ablation_table_loco.tex
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSPITALS = ("huaxi", "liaoning", "xiangya")
SEEDS = ("42", "123", "2024", "3407", "114514")
EPS = 1e-10

VARIANTS: List[Tuple[str, str]] = [
    ("Baseline", "outputs/outputs_baseline"),
    ("Only WMA", "outputs/消融实验/outputs_wma"),
    ("w/o WMA", "outputs/消融实验/outputs_ablation_no_wma"),
    ("w/o EMA", "outputs/消融实验/outputs_ablation_no_ema"),
    ("w/o Aux", "outputs/消融实验/outputs_ablation_no_aux"),
    ("OptiGenesis (Full)", "outputs/消融实验/outputs_wma_ema_aux"),
    ("Failed CORAL", "outputs/消融实验/outputs_optigenesis_final（fail"),
]


def _candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=np.float64)
    u = np.sort(np.unique(y_prob))
    cands: List[float] = [0.0]
    for a, b in zip(u[:-1], u[1:]):
        cands.append(float((a + b) / 2.0))
    cands.extend(float(x) for x in u)
    cands.append(1.0)
    return np.array(sorted(set(cands)), dtype=np.float64)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def _sensitivity_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Tuple[float, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens), float(spec)


def _youden(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    sens, spec = _sensitivity_specificity(y_true, y_prob, threshold)
    return sens + spec - 1.0


def _is_valid_threshold(y_true: np.ndarray, y_prob: np.ndarray, t: float) -> bool:
    if t <= EPS or t >= 1.0 - EPS:
        return False
    sens, spec = _sensitivity_specificity(y_true, y_prob, float(t))
    return sens > EPS and spec > EPS


def _collect_valid_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    seen: set[float] = set()
    valid: List[float] = []
    for t in _candidate_thresholds(y_prob):
        t = float(t)
        if 0.0 < t < 1.0 and _is_valid_threshold(y_true, y_prob, t):
            key = round(t, 12)
            if key not in seen:
                seen.add(key)
                valid.append(t)
    if valid:
        return sorted(valid)

    for t in np.linspace(1e-4, 1.0 - 1e-4, 2001, dtype=np.float64):
        t = float(t)
        if _is_valid_threshold(y_true, y_prob, t):
            key = round(t, 8)
            if key not in seen:
                seen.add(key)
                valid.append(t)
    return sorted(valid)


def find_optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    prevalence = float(np.mean(y_true))

    valid = _collect_valid_thresholds(y_true, y_prob)
    if not valid:
        raise ValueError("No valid non-degenerate threshold found in (0,1).")

    scored: List[Tuple[float, float, float, float]] = []
    for t in valid:
        yd = _youden(y_true, y_prob, t)
        y_pred = (y_prob >= t).astype(np.int32)
        f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        dist = abs(float(t) - prevalence)
        scored.append((float(t), yd, f1, dist))

    best_yd = max(s[1] for s in scored)
    tier1 = [s for s in scored if abs(s[1] - best_yd) <= 1e-12]
    best_f1 = max(s[2] for s in tier1)
    tier2 = [s for s in tier1 if abs(s[2] - best_f1) <= 1e-12]
    min_dist = min(s[3] for s in tier2)
    tier3 = [s for s in tier2 if abs(s[3] - min_dist) <= 1e-12]
    return min(s[0] for s in tier3)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, t: float) -> dict:
    y_pred = (y_prob >= t).astype(np.int32)
    sens, spec = _sensitivity_specificity(y_true, y_prob, t)
    return {
        "auc": safe_auc(y_true, y_prob),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": sens,
        "specificity": spec,
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.array(vals, dtype=np.float64)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return m, s


def fmt_pm(m: float, s: float, nd: int = 4) -> str:
    if math.isnan(m):
        return "nan"
    return f"{m:.{nd}f}±{s:.{nd}f}"


def parse_variant(variant_name: str, variant_rel_dir: str) -> dict:
    center_seed_metrics: Dict[str, Dict[str, dict]] = {h: {} for h in HOSPITALS}

    for hosp in HOSPITALS:
        for seed in SEEDS:
            csv_path = os.path.join(
                PROJECT_ROOT,
                variant_rel_dir,
                hosp,
                f"seed_{seed}",
                "logs",
                "external_sample_predictions.csv",
            )
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f"[{variant_name}] missing file: {csv_path}")

            df = pd.read_csv(csv_path)
            if "y_true" not in df.columns or "prob_positive" not in df.columns:
                raise ValueError(f"[{variant_name}] invalid columns in: {csv_path}")

            y_true = df["y_true"].to_numpy(dtype=np.int32)
            y_prob = df["prob_positive"].to_numpy(dtype=np.float64)
            t_star = find_optimal_threshold_youden(y_true, y_prob)
            center_seed_metrics[hosp][seed] = metrics_at_threshold(y_true, y_prob, t_star)

    seed_overall: Dict[str, dict] = {}
    for seed in SEEDS:
        aucs = [center_seed_metrics[h][seed]["auc"] for h in HOSPITALS]
        bals = [center_seed_metrics[h][seed]["bal_acc"] for h in HOSPITALS]
        sens = [center_seed_metrics[h][seed]["sensitivity"] for h in HOSPITALS]
        specs = [center_seed_metrics[h][seed]["specificity"] for h in HOSPITALS]
        f1s = [center_seed_metrics[h][seed]["f1"] for h in HOSPITALS]
        seed_overall[seed] = {
            "auc": float(np.mean(aucs)),
            "bal_acc": float(np.mean(bals)),
            "sensitivity": float(np.mean(sens)),
            "specificity": float(np.mean(specs)),
            "f1": float(np.mean(f1s)),
        }

    overall_auc_m, overall_auc_s = mean_std([seed_overall[s]["auc"] for s in SEEDS])
    overall_bal_m, overall_bal_s = mean_std([seed_overall[s]["bal_acc"] for s in SEEDS])
    overall_sens_m, overall_sens_s = mean_std([seed_overall[s]["sensitivity"] for s in SEEDS])

    h_auc_m, h_auc_s = mean_std([center_seed_metrics["huaxi"][s]["auc"] for s in SEEDS])
    l_auc_m, l_auc_s = mean_std([center_seed_metrics["liaoning"][s]["auc"] for s in SEEDS])
    x_auc_m, x_auc_s = mean_std([center_seed_metrics["xiangya"][s]["auc"] for s in SEEDS])

    return {
        "variant": variant_name,
        "overall_auc_mean": overall_auc_m,
        "overall_auc_std": overall_auc_s,
        "overall_bal_mean": overall_bal_m,
        "overall_bal_std": overall_bal_s,
        "overall_sens_mean": overall_sens_m,
        "overall_sens_std": overall_sens_s,
        "huaxi_auc_mean": h_auc_m,
        "huaxi_auc_std": h_auc_s,
        "liaoning_auc_mean": l_auc_m,
        "liaoning_auc_std": l_auc_s,
        "xiangya_auc_mean": x_auc_m,
        "xiangya_auc_std": x_auc_s,
    }


def render_markdown(rows: List[dict]) -> str:
    headers = [
        "Variant",
        "Overall AUC",
        "Overall Bal Acc",
        "Overall Sens",
        "Huaxi AUC",
        "Liaoning AUC",
        "Xiangya AUC",
        "Perf. Gap",
    ]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    r["variant"],
                    fmt_pm(r["overall_auc_mean"], r["overall_auc_std"]),
                    fmt_pm(r["overall_bal_mean"], r["overall_bal_std"]),
                    fmt_pm(r["overall_sens_mean"], r["overall_sens_std"]),
                    fmt_pm(r["huaxi_auc_mean"], r["huaxi_auc_std"]),
                    fmt_pm(r["liaoning_auc_mean"], r["liaoning_auc_std"]),
                    fmt_pm(r["xiangya_auc_mean"], r["xiangya_auc_std"]),
                    f"{r['perf_gap']:+.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(out) + "\n"


def esc_tex(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def render_latex(rows: List[dict]) -> str:
    lines = []
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\hline")
    lines.append(
        "Variant & Overall AUC & Overall Bal Acc & Overall Sens & Huaxi AUC & Liaoning AUC & Xiangya AUC & Perf. Gap \\\\"
    )
    lines.append("\\hline")
    for r in rows:
        lines.append(
            f"{esc_tex(r['variant'])} & "
            f"{fmt_pm(r['overall_auc_mean'], r['overall_auc_std'])} & "
            f"{fmt_pm(r['overall_bal_mean'], r['overall_bal_std'])} & "
            f"{fmt_pm(r['overall_sens_mean'], r['overall_sens_std'])} & "
            f"{fmt_pm(r['huaxi_auc_mean'], r['huaxi_auc_std'])} & "
            f"{fmt_pm(r['liaoning_auc_mean'], r['liaoning_auc_std'])} & "
            f"{fmt_pm(r['xiangya_auc_mean'], r['xiangya_auc_std'])} & "
            f"{r['perf_gap']:+.4f} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [parse_variant(name, rel) for name, rel in VARIANTS]

    full_row = next(r for r in rows if r["variant"] == "OptiGenesis (Full)")
    full_auc = full_row["overall_auc_mean"]
    for r in rows:
        r["perf_gap"] = r["overall_auc_mean"] - full_auc

    md = render_markdown(rows)
    tex = render_latex(rows)

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "消融实验")
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "final_ablation_table_loco.md")
    tex_path = os.path.join(out_dir, "final_ablation_table_loco.tex")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    print(md)
    print(f"[Saved] Markdown: {md_path}")
    print(f"[Saved] LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
