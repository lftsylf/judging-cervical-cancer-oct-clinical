#!/usr/bin/env python3
"""
对比实验目录统一执行 Youden Index 阈值解耦分析，并输出与 ViT 一致格式汇总：

Hospital,AUC (Mean±Std),Adjusted Bal Acc (Mean±Std),Adjusted F1 (Mean±Std),Sensitivity (Mean±Std),Specificity (Mean±Std)

默认处理目录:
  - outputs_comparison_convnext_small
  - outputs_comparison_swin_small
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

COMPARISON_DIR_TO_SUMMARY = {
    "outputs_comparison_convnext_small": "outputs_comparison_convnext_small_summary.csv",
    "outputs_comparison_swin_small": "outputs_comparison_swin_small_summary.csv",
}

_EPS = 1e-10


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


def _is_valid_classifier_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> bool:
    t = float(threshold)
    if t <= _EPS or t >= 1.0 - _EPS:
        return False
    sens, spec = _sensitivity_specificity(y_true, y_prob, t)
    return sens > _EPS and spec > _EPS


def _collect_valid_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    seen: set[float] = set()
    valid: List[float] = []

    for t in _candidate_thresholds(y_prob):
        t = float(t)
        if 0.0 < t < 1.0 and _is_valid_classifier_threshold(y_true, y_prob, t):
            key = round(t, 12)
            if key not in seen:
                seen.add(key)
                valid.append(t)

    if valid:
        return sorted(valid)

    for t in np.linspace(1e-4, 1.0 - 1e-4, 2001, dtype=np.float64):
        t = float(t)
        if _is_valid_classifier_threshold(y_true, y_prob, t):
            key = round(t, 8)
            if key not in seen:
                seen.add(key)
                valid.append(t)

    return sorted(valid)


def find_optimal_threshold_youden_primary(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, dict]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    prevalence = float(np.mean(y_true))

    valid = _collect_valid_thresholds(y_true, y_prob)
    if not valid:
        raise ValueError("无法在 (0,1) 内找到 Sens>0 且 Spec>0 的阈值。")

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
    best_t = min(s[0] for s in tier3)

    meta = {
        "youden_at_chosen": _youden(y_true, y_prob, best_t),
        "prevalence": prevalence,
        "n_valid_candidates": len(valid),
    }
    return best_t, meta


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int32)
    sens, spec = _sensitivity_specificity(y_true, y_prob, threshold)
    return {
        "adjusted_bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "adjusted_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity": sens,
        "specificity": spec,
    }


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    x = [v for v in vals if not math.isnan(v)]
    if not x:
        return float("nan"), float("nan")
    m = float(np.mean(x))
    if len(x) < 2:
        return m, 0.0
    s = float(np.std(x, ddof=1))
    return m, s


def _fmt_mean_std(m: float, s: float, nd: int = 4) -> str:
    if math.isnan(m):
        return "nan"
    if math.isnan(s):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}±{s:.{nd}f}"


def _iter_external_csvs(model_dir: str):
    for hosp in HOSPITALS:
        for seed in SEEDS:
            path = os.path.join(
                PROJECT_ROOT,
                model_dir,
                hosp,
                f"seed_{seed}",
                "logs",
                "external_sample_predictions.csv",
            )
            if os.path.isfile(path):
                yield hosp, int(seed), path


def analyze_one_model_dir(model_dir: str) -> pd.DataFrame:
    rows: List[dict] = []
    for hosp, seed, path in _iter_external_csvs(model_dir):
        df = pd.read_csv(path)
        if "y_true" not in df.columns or "prob_positive" not in df.columns:
            raise ValueError(f"{path} 缺少 y_true 或 prob_positive 列")

        y_true = df["y_true"].to_numpy(dtype=np.int32)
        y_prob = df["prob_positive"].to_numpy(dtype=np.float64)

        auc = safe_roc_auc(y_true, y_prob)
        t_star, _ = find_optimal_threshold_youden_primary(y_true, y_prob)
        m = metrics_at_threshold(y_true, y_prob, t_star)

        rows.append(
            {
                "hospital": hosp,
                "seed": seed,
                "auc": auc,
                "optimal_threshold": t_star,
                "adjusted_bal_acc": m["adjusted_bal_acc"],
                "adjusted_f1": m["adjusted_f1"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
            }
        )

    if not rows:
        raise ValueError(f"{model_dir} 未找到 external_sample_predictions.csv")

    rows.sort(key=lambda r: (HOSPITALS.index(r["hospital"]), r["seed"]))

    summary_rows: List[Dict[str, object]] = []
    for hosp in HOSPITALS:
        rs = [r for r in rows if r["hospital"] == hosp]
        if not rs:
            continue

        aucs = [r["auc"] for r in rs]
        bals = [r["adjusted_bal_acc"] for r in rs]
        f1s = [r["adjusted_f1"] for r in rs]
        sens = [r["sensitivity"] for r in rs]
        specs = [r["specificity"] for r in rs]

        m_auc, s_auc = _mean_std(aucs)
        m_bal, s_bal = _mean_std(bals)
        m_f1, s_f1 = _mean_std(f1s)
        m_sens, s_sens = _mean_std(sens)
        m_spec, s_spec = _mean_std(specs)

        summary_rows.append(
            {
                "Hospital": hosp,
                "AUC (Mean±Std)": _fmt_mean_std(m_auc, s_auc),
                "Adjusted Bal Acc (Mean±Std)": _fmt_mean_std(m_bal, s_bal),
                "Adjusted F1 (Mean±Std)": _fmt_mean_std(m_f1, s_f1),
                "Sensitivity (Mean±Std)": _fmt_mean_std(m_sens, s_sens),
                "Specificity (Mean±Std)": _fmt_mean_std(m_spec, s_spec),
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    for model_dir, summary_name in COMPARISON_DIR_TO_SUMMARY.items():
        summary_df = analyze_one_model_dir(model_dir)

        root_out = os.path.join(PROJECT_ROOT, summary_name)
        summary_df.to_csv(root_out, index=False, encoding="utf-8-sig")

        legacy_parent = os.path.join(PROJECT_ROOT, "outputs", "对比试验")
        os.makedirs(legacy_parent, exist_ok=True)
        legacy_out = os.path.join(legacy_parent, summary_name)
        summary_df.to_csv(legacy_out, index=False, encoding="utf-8-sig")

        print(f"[OK] {model_dir} -> {root_out}")
        print(f"[OK] {model_dir} -> {legacy_out}")


if __name__ == "__main__":
    main()
