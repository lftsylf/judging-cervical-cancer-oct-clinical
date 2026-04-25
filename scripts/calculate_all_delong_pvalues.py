#!/usr/bin/env python3
"""
Calculate all DeLong p-values for paper Table 1 and Table 2 (Xiangya center).

Champion:
  outputs/消融实验/outputs_wma_ema_aux/xiangya/seed_2024/logs/external_sample_predictions.csv

Challengers:
  [Table 1]
    - outputs_comparison_vit_base
    - outputs_comparison_vit_small
  [Table 2]
    - outputs_baseline
    - outputs_wma
    - outputs_ablation_no_wma
    - outputs_ablation_no_ema
    - outputs_ablation_no_aux
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREFERRED_SEED = "2024"
SUPPORTED_SEEDS = ("42", "123", "2024", "3407", "114514")
# 优先 2024，再 42 / 114514 等（与论文主设定一致）
SEED_TRY_ORDER = ("2024", "42", "123", "3407", "114514")

CHAMPION_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "消融实验",
    "outputs_wma_ema_aux",
    "xiangya",
    f"seed_{PREFERRED_SEED}",
    "logs",
    "external_sample_predictions.csv",
)

MODEL_ROOT_CANDIDATES: Dict[str, Sequence[str]] = {
    "outputs_comparison_vit_base": (
        os.path.join(PROJECT_ROOT, "outputs", "对比试验", "outputs_comparison_vit_base"),
    ),
    "outputs_comparison_vit_small": (
        os.path.join(PROJECT_ROOT, "outputs", "对比试验", "outputs_comparison_vit_small"),
    ),
    "outputs_comparison_convnext_small": (
        os.path.join(PROJECT_ROOT, "outputs_comparison_convnext_small"),
        os.path.join(PROJECT_ROOT, "outputs", "对比试验", "outputs_comparison_convnext_small"),
    ),
    "outputs_comparison_swin_small": (
        os.path.join(PROJECT_ROOT, "outputs_comparison_swin_small"),
        os.path.join(PROJECT_ROOT, "outputs", "对比试验", "outputs_comparison_swin_small"),
    ),
    "outputs_baseline": (
        os.path.join(PROJECT_ROOT, "outputs", "outputs_baseline"),
        os.path.join(PROJECT_ROOT, "outputs", "消融实验", "outputs_baseline"),
    ),
    "outputs_wma": (
        os.path.join(PROJECT_ROOT, "outputs", "消融实验", "outputs_wma"),
    ),
    "outputs_ablation_no_wma": (
        os.path.join(PROJECT_ROOT, "outputs", "消融实验", "outputs_ablation_no_wma"),
    ),
    "outputs_ablation_no_ema": (
        os.path.join(PROJECT_ROOT, "outputs", "消融实验", "outputs_ablation_no_ema"),
    ),
    "outputs_ablation_no_aux": (
        os.path.join(PROJECT_ROOT, "outputs", "消融实验", "outputs_ablation_no_aux"),
    ),
}

TABLE_1_MODELS: List[Tuple[str, str]] = [
    ("outputs_comparison_vit_base", "ViT-Base"),
    ("outputs_comparison_vit_small", "ViT-Small"),
    ("outputs_comparison_convnext_small", "ConvNeXt-Small"),
    ("outputs_comparison_swin_small", "Swin-Small"),
]

TABLE_2_MODELS: List[Tuple[str, str]] = [
    ("outputs_baseline", "Baseline"),
    ("outputs_wma", "With WMA (No EMA + Aux)"),
    ("outputs_ablation_no_wma", "No WMA"),
    ("outputs_ablation_no_ema", "No EMA"),
    ("outputs_ablation_no_aux", "No Aux"),
]


@dataclass
class ModelData:
    key: str
    display_name: str
    seed: str
    csv_path: str
    df: pd.DataFrame


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1.0
        midranks[i:j] = mid
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = midranks
    return out


def _fast_delong(y_true: np.ndarray, y_score_1: np.ndarray, y_score_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if len(y_true) != len(y_score_1) or len(y_true) != len(y_score_2):
        raise ValueError("y_true and score arrays must have same length")

    order = np.argsort(-y_true)
    y_true_sorted = y_true[order]
    if not np.array_equal(np.unique(y_true_sorted), np.array([0, 1])):
        raise ValueError("y_true must be binary with both classes present")

    scores = np.vstack([y_score_1[order], y_score_2[order]])
    m = int(np.sum(y_true_sorted == 1))
    n = int(np.sum(y_true_sorted == 0))

    pos = scores[:, :m]
    neg = scores[:, m:]

    tx = np.empty((2, m), dtype=np.float64)
    ty = np.empty((2, n), dtype=np.float64)
    tz = np.empty((2, m + n), dtype=np.float64)
    for r in range(2):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(scores[r])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01, bias=False)
    sy = np.cov(v10, bias=False)
    covariance = sx / m + sy / n
    return aucs, covariance


def delong_pvalue(y_true: np.ndarray, y_score_1: np.ndarray, y_score_2: np.ndarray) -> Tuple[float, float, float]:
    aucs, covariance = _fast_delong(y_true, y_score_1, y_score_2)
    diff = float(aucs[0] - aucs[1])
    var = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if var <= 0:
        z = math.inf if diff != 0 else 0.0
        p = 0.0 if diff != 0 else 1.0
        return p, float(aucs[0]), float(aucs[1])
    z = abs(diff) / math.sqrt(var)
    p = 2.0 * norm.sf(z)
    return float(p), float(aucs[0]), float(aucs[1])


def _resolve_model_root(model_key: str) -> str:
    for candidate in MODEL_ROOT_CANDIDATES[model_key]:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot locate root directory for {model_key}: {MODEL_ROOT_CANDIDATES[model_key]}")


def _csv_path_for(root: str, seed: str) -> str:
    return os.path.join(root, "xiangya", f"seed_{seed}", "logs", "external_sample_predictions.csv")


def _collect_available_seeds(root: str) -> List[str]:
    seeds: List[str] = []
    for seed in SUPPORTED_SEEDS:
        if os.path.isfile(_csv_path_for(root, seed)):
            seeds.append(seed)
    return seeds


def _pick_common_seed(champion_seed: str, challenger_root: str) -> str:
    challenger_seeds = set(_collect_available_seeds(challenger_root))
    if champion_seed in challenger_seeds:
        return champion_seed
    if not challenger_seeds:
        raise FileNotFoundError(f"No Xiangya seed CSV found in {challenger_root}")
    for seed in SEED_TRY_ORDER:
        if seed in challenger_seeds:
            return seed
    for seed in SUPPORTED_SEEDS:
        if seed in challenger_seeds:
            return seed
    return sorted(challenger_seeds)[0]


def _resolve_champion_xiangya_csv() -> Tuple[str, str]:
    """优先 seed_2024，不存在则按 SEED_TRY_ORDER 回退。"""
    base = os.path.join(PROJECT_ROOT, "outputs", "消融实验", "outputs_wma_ema_aux", "xiangya")
    for seed in SEED_TRY_ORDER:
        path = os.path.join(base, f"seed_{seed}", "logs", "external_sample_predictions.csv")
        if os.path.isfile(path):
            return seed, path
    raise FileNotFoundError(
        f"Champion CSV not found under {base} for seeds {SEED_TRY_ORDER}"
    )


def run_sota_convnext_swin_delong_only() -> None:
    """仅输出 OptiGenesis vs ConvNeXt-Small / Swin-Small 两行 DeLong（xiangya，oct_id 内连接）。"""
    champion_seed, champion_path = _resolve_champion_xiangya_csv()
    champion_df = _read_prediction_csv(champion_path)

    pairs: List[Tuple[str, str]] = [
        ("outputs_comparison_convnext_small", "ConvNeXt-Small"),
        ("outputs_comparison_swin_small", "Swin-Small"),
    ]
    for model_key, display_name in pairs:
        challenger = _load_challenger_data(model_key, display_name, champion_seed=champion_seed)
        y_true, s_champion, s_challenger, _label_mismatch = _align_on_oct_id(
            champion_df=champion_df,
            challenger_df=challenger.df,
        )
        p, _auc_champion, _auc_challenger = delong_pvalue(y_true, s_champion, s_challenger)
        print(f"OptiGenesis vs {display_name}: p = {_format_p(p)}")


def _read_prediction_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"oct_id", "y_true", "prob_positive"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.loc[:, ["oct_id", "y_true", "prob_positive"]].copy()


def _align_on_oct_id(
    champion_df: pd.DataFrame, challenger_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    merged = champion_df.merge(
        challenger_df,
        on="oct_id",
        how="inner",
        suffixes=("_champion", "_challenger"),
    )
    if merged.empty:
        raise ValueError("No overlapping oct_id between champion and challenger")

    y_champion = merged["y_true_champion"].to_numpy(dtype=np.int32)
    y_challenger = merged["y_true_challenger"].to_numpy(dtype=np.int32)
    label_mismatch = int(np.sum(y_champion != y_challenger))

    y_true = y_champion
    y_score_champion = merged["prob_positive_champion"].to_numpy(dtype=np.float64)
    y_score_challenger = merged["prob_positive_challenger"].to_numpy(dtype=np.float64)
    return y_true, y_score_champion, y_score_challenger, label_mismatch


def _format_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.6f}"


def _load_challenger_data(model_key: str, display_name: str, champion_seed: str) -> ModelData:
    root = _resolve_model_root(model_key)
    seed = _pick_common_seed(champion_seed=champion_seed, challenger_root=root)
    csv_path = _csv_path_for(root, seed)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = _read_prediction_csv(csv_path)
    return ModelData(key=model_key, display_name=display_name, seed=seed, csv_path=csv_path, df=df)


def _print_section(
    title: str,
    models: Iterable[Tuple[str, str]],
    champion_name: str,
    champion_seed: str,
    champion_df: pd.DataFrame,
) -> None:
    print(f"\n=== {title} ===")
    for model_key, display_name in models:
        challenger = _load_challenger_data(model_key, display_name, champion_seed=champion_seed)
        y_true, s_champion, s_challenger, label_mismatch = _align_on_oct_id(
            champion_df=champion_df,
            challenger_df=challenger.df,
        )
        p, auc_champion, auc_challenger = delong_pvalue(y_true, s_champion, s_challenger)
        n = len(y_true)
        mismatch_note = f", y_true_mismatch={label_mismatch}" if label_mismatch > 0 else ""
        print(
            f"{champion_name} vs {challenger.display_name}: "
            f"p = {_format_p(p)} "
            f"(AUC: {auc_champion:.4f} vs {auc_challenger:.4f}, n={n}, seed={challenger.seed}{mismatch_note})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeLong tests (Xiangya external ROC-AUC).")
    parser.add_argument(
        "--sota-convnext-swin-only",
        action="store_true",
        help="Only print OptiGenesis vs ConvNeXt-Small and vs Swin-Small (strict oct_id inner join).",
    )
    args = parser.parse_args()

    if args.sota_convnext_swin_only:
        run_sota_convnext_swin_delong_only()
        return

    if not os.path.isfile(CHAMPION_PATH):
        raise FileNotFoundError(f"Champion CSV not found: {CHAMPION_PATH}")

    champion_seed = PREFERRED_SEED
    champion_df = _read_prediction_csv(CHAMPION_PATH)
    champion_auc = roc_auc_score(
        champion_df["y_true"].to_numpy(dtype=np.int32),
        champion_df["prob_positive"].to_numpy(dtype=np.float64),
    )

    print("DeLong's Test for Xiangya ROC-AUC (two-sided)")
    print("-" * 72)
    print(f"Champion: OptiGenesis (outputs_wma_ema_aux)")
    print(f"Path: {CHAMPION_PATH}")
    print(f"Seed: {champion_seed}")
    print(f"AUC: {champion_auc:.4f}")

    _print_section(
        title="Table 1: SOTA Comparisons",
        models=TABLE_1_MODELS,
        champion_name="OptiGenesis",
        champion_seed=champion_seed,
        champion_df=champion_df,
    )
    _print_section(
        title="Table 2: Ablation Studies",
        models=TABLE_2_MODELS,
        champion_name="OptiGenesis",
        champion_seed=champion_seed,
        champion_df=champion_df,
    )


if __name__ == "__main__":
    main()
