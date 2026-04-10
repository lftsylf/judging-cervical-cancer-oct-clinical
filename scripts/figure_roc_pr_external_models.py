"""
Figure 2 (ROC) 与 Figure 3 (PR)：读取三折 external 预测，绘制 1×3 子图对比。

用法示例:
    python scripts/figure_roc_pr_external_models.py
    python scripts/figure_roc_pr_external_models.py --out-dir figures/paper
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 默认与仓库 all_outputs 布局一致；可用 CLI 覆盖
DEFAULT_MODEL_ROOTS: Dict[str, str] = {
    "OptiGenesis": "all_outputs/baseline_outputs",
    "ViT-Small": "all_outputs/Comparisons_vit_loho_outputs/vit_small_patch16_224",
    "ViT-Base": "all_outputs/Comparisons_vit_loho_outputs/vit_base_patch16_224",
}

FOLDS: List[Tuple[str, str, str]] = [
    ("xiangya", "湘雅", "Xiangya"),
    ("huaxi", "华西", "Huaxi"),
    ("liaoning", "辽宁", "Liaoning"),
]

# OptiGenesis 红色加粗；对比模型浅蓝 / 翠绿
STYLE: Dict[str, Dict] = {
    "OptiGenesis": {"color": "#C41E3A", "linewidth": 2.5, "linestyle": "-"},
    "ViT-Small": {"color": "#00BFFF", "linewidth": 1.8, "linestyle": "-"},  # DeepSkyBlue
    "ViT-Base": {"color": "#00C957", "linewidth": 1.8, "linestyle": "-"},  # 翠绿
}


def _set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "0.8",
            "figure.dpi": 150,
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def resolve_fold_titles(use_chinese_titles: bool, blind_centers: bool = False) -> List[Tuple[str, str]]:
    """返回 (fold_key, subplot_title)。中文标题需系统 CJK 字体，否则回退英文。"""
    global _TITLE_FONT_PROPERTIES
    if blind_centers:
        _TITLE_FONT_PROPERTIES = None
        blind_map = {
            "xiangya": "Center B",
            "huaxi": "Center A",
            "liaoning": "Center C",
        }
        return [(k, blind_map.get(k, k)) for k, _cn, _en in FOLDS]

    if not use_chinese_titles:
        _TITLE_FONT_PROPERTIES = None
        return [(k, en) for k, _cn, en in FOLDS]

    cjk_candidates = [
        "Noto Serif CJK SC",
        "Source Han Serif SC",
        "SimSun",
        "STSong",
        "Microsoft YaHei",
        "WenQuanYi Zen Hei",
    ]
    title_fp = None
    for fam in cjk_candidates:
        fp = fm.FontProperties(family=fam)
        try:
            path = fm.findfont(fp, fallback_to_default=False)
        except (TypeError, ValueError):
            continue
        if path and "dejavu" not in path.lower():
            title_fp = fp
            break

    out: List[Tuple[str, str]] = []
    for k, cn, en in FOLDS:
        if title_fp is not None:
            out.append((k, cn))
        else:
            out.append((k, en))
    _TITLE_FONT_PROPERTIES = title_fp
    return out


_TITLE_FONT_PROPERTIES: fm.FontProperties | None = None


def set_subplot_title(ax: plt.Axes, title: str) -> None:
    """中文标题使用 CJK 字体；英文标题沿用全局 serif (Times New Roman)。"""
    if _TITLE_FONT_PROPERTIES is not None and any(ord(c) > 127 for c in title):
        ax.set_title(title, fontproperties=_TITLE_FONT_PROPERTIES)
    else:
        ax.set_title(title)


def load_external_preds(model_root: str, fold: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(PROJECT_ROOT, model_root, fold, "logs", "external_sample_predictions.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少文件: {path}")
    df = pd.read_csv(path)
    if "y_true" not in df.columns or "prob_positive" not in df.columns:
        raise ValueError(f"{path} 需包含列 y_true, prob_positive")
    y = df["y_true"].to_numpy(dtype=np.int32)
    p = df["prob_positive"].to_numpy(dtype=np.float64)
    return y, p


def plot_roc_figure(
    model_roots: Dict[str, str],
    out_path: str,
    fold_titles: List[Tuple[str, str]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    for ax, (fold_key, fold_title) in zip(axes, fold_titles):
        for name, root in model_roots.items():
            y, p = load_external_preds(root, fold_key)
            fpr, tpr, _ = roc_curve(y, p)
            roc_auc = roc_auc_score(y, p)
            st = STYLE[name]
            ax.plot(
                fpr,
                tpr,
                label=f"{name} (AUC={roc_auc:.2f})",
                color=st["color"],
                linewidth=st["linewidth"],
                linestyle=st["linestyle"],
            )
        ax.plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        set_subplot_title(ax, fold_title)
        despine(ax)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)

    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr_figure(
    model_roots: Dict[str, str],
    out_path: str,
    fold_titles: List[Tuple[str, str]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    for ax, (fold_key, fold_title) in zip(axes, fold_titles):
        y, _ = load_external_preds(next(iter(model_roots.values())), fold_key)
        pos_rate = float(np.mean(y)) if len(y) else 0.0

        for name, root in model_roots.items():
            y, p = load_external_preds(root, fold_key)
            prec, rec, _ = precision_recall_curve(y, p)
            # sklearn PR 曲线 x 为 recall，y 为 precision；AUC 用 average_precision（与 PR-AUC 一致）
            ap = average_precision_score(y, p)
            st = STYLE[name]
            ax.plot(
                rec,
                prec,
                label=f"{name} (AUC={ap:.2f})",
                color=st["color"],
                linewidth=st["linewidth"],
                linestyle=st["linestyle"],
            )

        ax.axhline(pos_rate, color="0.55", linestyle=":", linewidth=1.0, alpha=0.9, label=None)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        set_subplot_title(ax, fold_title)
        despine(ax)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)

    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 External 集 ROC / PR 对比图 (Figure 2 & 3)")
    parser.add_argument(
        "--baseline-root",
        default=DEFAULT_MODEL_ROOTS["OptiGenesis"],
        help="Baseline (OptiGenesis) 输出根目录（相对项目根）",
    )
    parser.add_argument(
        "--vit-small-root",
        default=DEFAULT_MODEL_ROOTS["ViT-Small"],
        help="ViT-Small 输出根目录（相对项目根）",
    )
    parser.add_argument(
        "--vit-base-root",
        default=DEFAULT_MODEL_ROOTS["ViT-Base"],
        help="ViT-Base 输出根目录（相对项目根）",
    )
    parser.add_argument(
        "--out-dir",
        default="figures/paper",
        help="PDF 输出目录（相对项目根）",
    )
    parser.add_argument("--roc-name", default="figure2_roc_external.pdf")
    parser.add_argument("--pr-name", default="figure3_pr_external.pdf")
    parser.add_argument(
        "--english-titles",
        action="store_true",
        help="子图标题强制使用英文 (Xiangya / Huaxi / Liaoning)，与 Times New Roman 完全一致",
    )
    parser.add_argument(
        "--blind-centers",
        action="store_true",
        help="盲审模式：标题使用 Center A/B/C（Xiangya->B, Huaxi->A, Liaoning->C）",
    )
    args = parser.parse_args()

    model_roots = {
        "OptiGenesis": args.baseline_root,
        "ViT-Small": args.vit_small_root,
        "ViT-Base": args.vit_base_root,
    }

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    roc_path = os.path.join(out_dir, args.roc_name)
    pr_path = os.path.join(out_dir, args.pr_name)

    _set_ieee_style()
    fold_titles = resolve_fold_titles(
        use_chinese_titles=not args.english_titles,
        blind_centers=args.blind_centers,
    )
    plot_roc_figure(model_roots, roc_path, fold_titles)
    plot_pr_figure(model_roots, pr_path, fold_titles)

    print("已保存:")
    print(f"  ROC: {roc_path}")
    print(f"  PR:  {pr_path}")


if __name__ == "__main__":
    main()
