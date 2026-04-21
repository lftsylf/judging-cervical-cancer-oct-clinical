#!/usr/bin/env python3
"""
OptiGenesis V2 外部集阈值解耦（Threshold Decoupling）后处理分析。

遍历 outputs_optigenesis_v2 下各中心 × seed 的 external_sample_predictions.csv，
在**非退化解**阈值上**优先最大化 Youden Index**（等价于最大化 Balanced Accuracy）；
Youden 并列时取**阳性 F1** 更高者；再并列则取**阈值更接近阳性患病率**者；仍并列取**较小阈值**。

退化解约束：剔除 t∈{0,1}，且剔除 Sens=0 或 Spec=0 的阈值。

用法:
    python scripts/analyze_v2_optimal_thresholds.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HOSPITALS = ("huaxi", "liaoning", "xiangya")
SEEDS = ("42", "123", "2024", "3407", "114514")
DEFAULT_THRESHOLD = 0.5

# 浮点比较与退化解判定
_EPS = 1e-10


def _candidate_thresholds(y_prob: np.ndarray) -> np.ndarray:
    """产生可使预测发生变化的阈值候选（含相邻概率中点；端点 0/1 由上游剔除）。"""
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
    """排除 t=0/1 及 Sens 或 Spec 为 0 的退化解。"""
    t = float(threshold)
    if t <= _EPS or t >= 1.0 - _EPS:
        return False
    sens, spec = _sensitivity_specificity(y_true, y_prob, t)
    return sens > _EPS and spec > _EPS


def _collect_valid_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    """先用语义候选；若无有效阈值，再在 (0,1) 内密网格搜索。"""
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
    """
    Primary: 最大化 Youden Index（= Sens + Spec - 1，与 Balanced Accuracy 同单调）。
    Tie-1: 最大化阳性 F1。
    Tie-2: |t - prevalence| 最小（prevalence = 样本阳性比例）。
    Tie-3: 取较小阈值。
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    prevalence = float(np.mean(y_true))

    valid = _collect_valid_thresholds(y_true, y_prob)
    if not valid:
        raise ValueError(
            "无法在 (0,1) 内找到 Sens>0 且 Spec>0 的阈值；请检查标签与预测概率是否可分离。"
        )

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
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity": sens,
        "specificity": spec,
    }


def iter_external_csvs(root: str) -> Iterable[Tuple[str, str, str]]:
    for hosp in HOSPITALS:
        for seed in SEEDS:
            path = os.path.join(root, hosp, f"seed_{seed}", "logs", "external_sample_predictions.csv")
            if os.path.isfile(path):
                yield hosp, seed, path


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
    if math.isnan(s) or s == 0.0:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}±{s:.{nd}f}"


def print_table_a(rows: List[dict]) -> None:
    headers = [
        "Hospital",
        "Seed",
        "Original AUC",
        "Default(0.5) Bal Acc",
        "Optimal Threshold",
        "Adjusted Bal Acc",
        "Adjusted F1",
        "Sensitivity",
        "Specificity",
    ]
    col_w = [10, 8, 14, 20, 18, 18, 14, 12, 12]
    line = " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        vals = [
            r["hospital"],
            str(r["seed"]),
            f"{r['original_auc']:.4f}" if not math.isnan(r["original_auc"]) else "nan",
            f"{r['default_bal_acc']:.4f}",
            f"{r['optimal_threshold']:.6f}",
            f"{r['adjusted_bal_acc']:.4f}",
            f"{r['adjusted_f1']:.4f}",
            f"{r['sensitivity']:.4f}",
            f"{r['specificity']:.4f}",
        ]
        print(" | ".join(v.ljust(col_w[i]) for i, v in enumerate(vals)))


def print_table_b(summary: List[dict]) -> None:
    headers = [
        "Hospital",
        "AUC (Mean±Std)",
        "Optimal Threshold (Mean)",
        "Adjusted Bal Acc (Mean±Std)",
        "Adjusted F1 (Mean±Std)",
    ]
    col_w = [12, 22, 24, 28, 28]
    line = " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in summary:
        vals = [
            r["hospital"],
            r["auc_mean_std"],
            f"{r['optimal_threshold_mean']:.6f}",
            r["adj_bal_mean_std"],
            r["adj_f1_mean_std"],
        ]
        print(" | ".join(str(v).ljust(col_w[i]) for i, v in enumerate(vals)))


def main() -> None:
    root = os.path.join(PROJECT_ROOT, "outputs_optigenesis_v2")
    out_csv = os.path.join(root, "v2_threshold_decoupling_summary.csv")

    rows: List[dict] = []
    for hosp, seed, path in iter_external_csvs(root):
        df = pd.read_csv(path)
        if "y_true" not in df.columns or "prob_positive" not in df.columns:
            raise ValueError(f"{path} 需包含列 y_true, prob_positive")
        y_true = df["y_true"].to_numpy(dtype=np.int32)
        y_prob = df["prob_positive"].to_numpy(dtype=np.float64)

        original_auc = safe_roc_auc(y_true, y_prob)
        default_pred = (y_prob >= DEFAULT_THRESHOLD).astype(np.int32)
        default_bal_acc = float(balanced_accuracy_score(y_true, default_pred))

        t_star, _ = find_optimal_threshold_youden_primary(y_true, y_prob)
        adj = metrics_at_threshold(y_true, y_prob, t_star)

        rows.append(
            {
                "hospital": hosp,
                "seed": int(seed),
                "original_auc": original_auc,
                "default_bal_acc": default_bal_acc,
                "optimal_threshold": t_star,
                "adjusted_bal_acc": adj["balanced_accuracy"],
                "adjusted_f1": adj["f1_positive"],
                "sensitivity": adj["sensitivity"],
                "specificity": adj["specificity"],
            }
        )

    rows.sort(key=lambda r: (HOSPITALS.index(r["hospital"]), r["seed"]))

    print("\n" + "=" * 100)
    print("表 A — 明细（外部集，Youden 主目标 + 非退化解）")
    print("=" * 100)
    print_table_a(rows)

    by_h: dict = {h: [] for h in HOSPITALS}
    for r in rows:
        by_h[r["hospital"]].append(r)

    summary: List[dict] = []
    for hosp in HOSPITALS:
        rs = by_h[hosp]
        aucs = [r["original_auc"] for r in rs]
        thrs = [r["optimal_threshold"] for r in rs]
        bals = [r["adjusted_bal_acc"] for r in rs]
        f1s = [r["adjusted_f1"] for r in rs]

        m_auc, s_auc = _mean_std(aucs)
        m_thr = float(np.mean(thrs)) if thrs else float("nan")
        m_bal, s_bal = _mean_std(bals)
        m_f1, s_f1 = _mean_std(f1s)

        summary.append(
            {
                "hospital": hosp,
                "auc_mean_std": _fmt_mean_std(m_auc, s_auc),
                "optimal_threshold_mean": m_thr,
                "adj_bal_mean_std": _fmt_mean_std(m_bal, s_bal),
                "adj_f1_mean_std": _fmt_mean_std(m_f1, s_f1),
            }
        )

    print("\n" + "=" * 100)
    print("表 B — 按中心汇总（5 seeds，Mean ± Std）")
    print("=" * 100)
    print_table_b(summary)

    lines: List[str] = []
    lines.append(
        "# 表A-明细（N=15）：主目标=约登指数(Youden)，剔除 t=0/1 及 Sens 或 Spec 为 0 的退化解"
    )
    lines.append(
        "hospital,seed,original_auc,default_bal_acc,optimal_threshold,"
        "adjusted_bal_acc,adjusted_f1,sensitivity,specificity"
    )
    for r in rows:
        oa = "nan" if math.isnan(r["original_auc"]) else f"{r['original_auc']:.8f}"
        lines.append(
            f"{r['hospital']},{r['seed']},{oa},{r['default_bal_acc']:.8f},"
            f"{r['optimal_threshold']:.8f},{r['adjusted_bal_acc']:.8f},{r['adjusted_f1']:.8f},"
            f"{r['sensitivity']:.8f},{r['specificity']:.8f}"
        )
    lines.append("# 表B-按医院汇总（每中心 5 个 seed，Mean±Std）")
    lines.append(
        "hospital,auc_mean_std,optimal_threshold_mean,adjusted_bal_acc_mean_std,adjusted_f1_mean_std"
    )
    for s in summary:
        lines.append(
            f"{s['hospital']},{s['auc_mean_std']},{s['optimal_threshold_mean']:.8f},"
            f"{s['adj_bal_mean_std']},{s['adj_f1_mean_std']}"
        )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 100)
    print(f"已保存: {out_csv}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
