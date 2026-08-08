#!/usr/bin/env python3
"""
paper_v4 / baseline2：阈值后处理 + ROC/PR 汇总 + 论文用 Markdown/CSV 表。

协议：湘雅内训 / 华西+辽宁外测；主报 Ext pooled ROC；阈值相关指标单独解耦。

阈值策略（--threshold-policy）:
  youden_on_split  — 与 v2 Table1/2 一致：在**当前评估划分**上 per-seed Youden
                     （Val / Ext / Hx / Ln 各自标定；乐观操作点，文中写清）
  youden_on_val    — 在湘雅 Val 上标定 t*，应用到各外部划分（更贴近部署）
  fixed0.5         — 固定 0.5（日志默认；小样本 BA 常不稳）

用法:
  export PATH="/home/amax/anaconda3/bin:$PATH"
  python scripts/paper_v4_run/analyze_paper_v4_metrics.py
  python scripts/paper_v4_run/analyze_paper_v4_metrics.py --threshold-policy youden_on_val
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.paper_v4_run.paper_v4_registry import (  # noqa: E402
    OUT_TABLES,
    PRED_SPLITS,
    SEEDS,
    SPLIT_LABEL,
    Experiment,
    champion,
    experiments_for_tables,
)
from scripts.youden_threshold_utils import (  # noqa: E402
    find_optimal_threshold_youden_primary,
    fmt_mean_std,
    mean_std,
    metrics_at_threshold,
)

try:
    from scipy.stats import wilcoxon
except ImportError:  # pragma: no cover
    wilcoxon = None  # type: ignore


def _pred_path(run_dir: str, seed: int, split: str) -> str:
    return os.path.join(
        run_dir, f"seed_{seed}", "logs", f"{split}_sample_predictions.csv"
    )


def load_split_df(run_dir: str, seed: int, split: str) -> Optional[pd.DataFrame]:
    """Load predictions for a split.

    **重要**：华西/辽宁必须优先读 ``external_{huaxi,liaoning}_sample_predictions.csv``。
    勿默认从 pooled ``external_*.csv`` 按 ``center`` 列切片——该文件与分中心文件的
    ``y_true`` / ``prob_positive`` 可能不一致，会导致 Hx/Ln AUC 偏离训练日志与简报。
    仅当缺少分中心 CSV（如部分旧 baseline 只写了 pooled）时才回退切片，并打警告。
    """
    path = _pred_path(run_dir, seed, split)
    if os.path.isfile(path):
        df = pd.read_csv(path)
    elif split in ("external_huaxi", "external_liaoning"):
        pooled = _pred_path(run_dir, seed, "external")
        if not os.path.isfile(pooled):
            return None
        print(
            f"[WARN] {run_dir} seed={seed} 缺 {split} CSV，回退 pooled+center 切片"
            "（Hx/Ln 可能与正式分中心评不一致）"
        )
        df = pd.read_csv(pooled)
        center_key = "huaxi" if split.endswith("huaxi") else "liaoning"
        if "center" not in df.columns:
            return None
        df = df[df["center"].astype(str).str.lower() == center_key].copy()
        if df.empty:
            return None
    else:
        return None
    if "y_true" not in df.columns or "prob_positive" not in df.columns:
        raise ValueError(f"{path or split} 缺少 y_true / prob_positive")
    return df


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def resolve_threshold(
    policy: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    val_df: Optional[pd.DataFrame],
) -> Tuple[float, str]:
    if policy == "fixed0.5":
        return 0.5, "fixed0.5"
    if policy == "youden_on_split":
        t, _ = find_optimal_threshold_youden_primary(y_true, y_prob)
        return float(t), "youden_on_split"
    if policy == "youden_on_val":
        if val_df is None:
            raise ValueError("youden_on_val 需要 val_sample_predictions.csv")
        yt = val_df["y_true"].to_numpy(dtype=np.int32)
        yp = val_df["prob_positive"].to_numpy(dtype=np.float64)
        t, _ = find_optimal_threshold_youden_primary(yt, yp)
        return float(t), "youden_on_val"
    raise ValueError(f"未知 threshold policy: {policy}")


def collect_seed_rows(
    exp: Experiment,
    project_root: str,
    threshold_policy: str,
) -> List[dict]:
    run_dir = os.path.join(project_root, exp.rel_dir)
    rows: List[dict] = []
    for seed in SEEDS:
        val_df = load_split_df(run_dir, seed, "val")
        for split in PRED_SPLITS:
            df = load_split_df(run_dir, seed, split)
            if df is None:
                continue
            y_true = df["y_true"].to_numpy(dtype=np.int32)
            y_prob = df["prob_positive"].to_numpy(dtype=np.float64)
            auc = safe_auc(y_true, y_prob)
            ap = safe_ap(y_true, y_prob)
            try:
                t_star, t_src = resolve_threshold(
                    threshold_policy, y_true, y_prob, val_df
                )
                m = metrics_at_threshold(y_true, y_prob, t_star)
            except Exception as exc:  # noqa: BLE001 — 单 seed 失败不拖垮整表
                t_star, t_src = float("nan"), f"error:{exc}"
                m = {
                    "adjusted_bal_acc": float("nan"),
                    "adjusted_f1": float("nan"),
                    "sensitivity": float("nan"),
                    "specificity": float("nan"),
                    "ppv": float("nan"),
                    "npv": float("nan"),
                }
            youden = (
                float(m["sensitivity"] + m["specificity"] - 1.0)
                if not math.isnan(m["sensitivity"])
                else float("nan")
            )
            rows.append(
                {
                    "model_key": exp.key,
                    "model": exp.display,
                    "role": exp.role,
                    "paper_table": exp.paper_table,
                    "seed": seed,
                    "split": split,
                    "split_label": SPLIT_LABEL[split],
                    "n": int(len(df)),
                    "auc": auc,
                    "pr_auc": ap,
                    "threshold": t_star,
                    "threshold_source": t_src,
                    "youden": youden,
                    **m,
                }
            )
    return rows


def summarize_group(df: pd.DataFrame, metric_keys: List[str]) -> pd.DataFrame:
    out_rows = []
    for (model, split), g in df.groupby(["model", "split"], sort=False):
        entry = {
            "model": model,
            "split": split,
            "split_label": g["split_label"].iloc[0],
            "n_seeds": int(g["seed"].nunique()),
            "role": g["role"].iloc[0],
            "paper_table": g["paper_table"].iloc[0],
        }
        for k in metric_keys:
            m, s = mean_std([float(x) for x in g[k].tolist()])
            entry[f"{k}_mean"] = m
            entry[f"{k}_std"] = s
            entry[f"{k}_fmt"] = fmt_mean_std(m, s)
        out_rows.append(entry)
    return pd.DataFrame(out_rows)


def pivot_auc_table(summary: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    """Rows=model, cols=Val/Ext/Hx/Ln."""
    col_order = ["val", "external", "external_huaxi", "external_liaoning"]
    models = summary["model"].drop_duplicates().tolist()
    rows = []
    for model in models:
        sub = summary[summary["model"] == model]
        role = sub["role"].iloc[0] if len(sub) else ""
        # Prefer Ext pooled seed count for the table's n_seeds column
        ext_rows = sub[sub["split"] == "external"]
        n_seeds = (
            int(ext_rows["n_seeds"].iloc[0])
            if not ext_rows.empty
            else (int(sub["n_seeds"].max()) if len(sub) else 0)
        )
        entry = {"Model": model, "Role": role, "n_seeds": n_seeds}
        for split in col_order:
            r = sub[sub["split"] == split]
            if r.empty:
                entry[SPLIT_LABEL[split]] = "—"
            else:
                entry[SPLIT_LABEL[split]] = r[f"{metric}_fmt"].iloc[0]
        # Δ Ext vs Ours
        if metric == "auc" and "Ours (R50)" in models:
            ours = summary[
                (summary["model"] == "Ours (R50)") & (summary["split"] == "external")
            ]
            cur = sub[sub["split"] == "external"]
            if not ours.empty and not cur.empty:
                delta = float(cur["auc_mean"].iloc[0] - ours["auc_mean"].iloc[0])
                entry["ΔExt ROC"] = f"{delta:+.3f}"
            else:
                entry["ΔExt ROC"] = "—"
        rows.append(entry)
    return pd.DataFrame(rows)


def paired_wilcoxon_vs_ours(detail: pd.DataFrame) -> pd.DataFrame:
    if wilcoxon is None:
        return pd.DataFrame()
    ours_key = champion().display
    ext = detail[detail["split"] == "external"].copy()
    if ext.empty:
        return pd.DataFrame()
    ours = (
        ext[ext["model"] == ours_key]
        .set_index("seed")["auc"]
        .astype(float)
        .sort_index()
    )
    out = []
    for model, g in ext.groupby("model", sort=False):
        if model == ours_key:
            continue
        chall = g.set_index("seed")["auc"].astype(float).sort_index()
        common = ours.index.intersection(chall.index)
        if len(common) < 3:
            continue
        a = ours.loc[common].to_numpy()
        b = chall.loc[common].to_numpy()
        diff = a - b
        # one-sided: Ours > challenger
        try:
            if np.allclose(diff, 0):
                p = 1.0
            else:
                stat = wilcoxon(diff, alternative="greater", zero_method="wilcox")
                p = float(stat.pvalue)
        except ValueError:
            p = float("nan")
        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff < 0))
        out.append(
            {
                "challenger": model,
                "n_pairs": int(len(common)),
                "mean_delta_ours_minus_chall": float(np.mean(diff)),
                "wins_ours": wins,
                "losses_ours": losses,
                "wilcoxon_onesided_p": p,
                "sig": (
                    "**"
                    if p < 0.01
                    else ("*" if p < 0.05 else "ns")
                    if not math.isnan(p)
                    else "—"
                ),
            }
        )
    return pd.DataFrame(out)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def filter_table_models(summary: pd.DataFrame, paper_table: str) -> pd.DataFrame:
    # Always include Ours in ablation table for Δ
    keys = set(
        summary.loc[summary["paper_table"] == paper_table, "model"].tolist()
    )
    if paper_table == "main_ablation":
        keys.add("Ours (R50)")
    return summary[summary["model"].isin(keys)].copy()


def write_paper_md(
    out_dir: str,
    summary: pd.DataFrame,
    youden_ext: pd.DataFrame,
    paired: pd.DataFrame,
    threshold_policy: str,
    incomplete: List[str],
    suffix: str = "",
) -> str:
    path = os.path.join(out_dir, f"PAPER_TABLES_baseline2{suffix}.md")
    cmp_sum = filter_table_models(summary, "main_cmp")
    abl_sum = filter_table_models(summary, "main_ablation")
    app_sum = filter_table_models(summary, "appendix")

    roc_cmp = pivot_auc_table(cmp_sum, "auc")
    pr_cmp = pivot_auc_table(cmp_sum, "pr_auc")
    roc_abl = pivot_auc_table(abl_sum, "auc")
    roc_app = pivot_auc_table(app_sum, "auc") if not app_sum.empty else pd.DataFrame()

    # Youden operating-point on Ext pooled (and centers) for main_cmp + ours
    op_models = set(cmp_sum["model"].tolist()) | {"Ours (R50)"}
    op = youden_ext[youden_ext["model"].isin(op_models)].copy()

    lines = [
        "# paper_v4 / baseline2 论文用表（自动生成）",
        "",
        "> 协议：湘雅内训 · 华西+辽宁外测 · OCT-only · 5 seeds（42/123/2024/3407/114514）。",
        "> **主指标：Ext pooled ROC-AUC（已锁定）**。勿与旧 LOHO / Attn 数字横比。",
        f"> 阈值策略：`{threshold_policy}`（与 v2/v3 共用 `youden_threshold_utils.py`："
        "Youden 主目标，平局 F1 → |t−患病率|；仅影响 Sens/Spec/PPV/NPV/Youden；**AUC/PR 不依赖阈值**）。",
        "> **草稿声明**：操作点列（Sens/Spec/…）算法已对齐 v3，但**论文表最终指标集合尚未由作者定稿**。",
        "> 华西/辽宁来自分中心预测 CSV，非 pooled 切片。",
        "> 生成：`python scripts/paper_v4_run/analyze_paper_v4_metrics.py`",
        "",
    ]
    if incomplete:
        lines.append("## ⚠ 不完整 run（表中 n_seeds < 5）")
        lines.append("")
        for msg in incomplete:
            lines.append(f"- {msg}")
        lines.append("")

    lines += [
        "## Table · 主对比（ROC）",
        "",
        markdown_table(roc_cmp),
        "",
        "## Table · 主对比（PR-AUC）",
        "",
        markdown_table(pr_cmp),
        "",
        f"## Table · 操作点（{threshold_policy} · Ext pooled）",
        "",
    ]
    op_ext = op[op["split"] == "external"][
        [
            "model",
            "n_seeds",
            "auc_fmt",
            "sensitivity_fmt",
            "specificity_fmt",
            "ppv_fmt",
            "npv_fmt",
            "youden_fmt",
            "threshold_fmt",
        ]
    ].rename(
        columns={
            "model": "Model",
            "auc_fmt": "ROC-AUC",
            "sensitivity_fmt": "Sens",
            "specificity_fmt": "Spec",
            "ppv_fmt": "PPV",
            "npv_fmt": "NPV",
            "youden_fmt": "Youden",
            "threshold_fmt": "t*",
        }
    )
    lines += [markdown_table(op_ext), ""]

    lines += [
        "## Table · 消融（ROC）",
        "",
        markdown_table(roc_abl),
        "",
    ]
    if not roc_app.empty:
        lines += [
            "## Appendix · 骨干变体（ROC）",
            "",
            markdown_table(roc_app),
            "",
        ]

    if not paired.empty:
        lines += [
            "## Ours vs 对照（Ext pooled · 配对 Wilcoxon 单侧 Ours>challenger）",
            "",
            markdown_table(paired),
            "",
            "注：种子数少（≤5）时 p 值仅作参考，主文可以 mean±std + Δ 为主。",
            "",
        ]

    lines += [
        "## 表述提醒",
        "",
        "- UWA = 窗**内**帧不确定加权；Mean = 窗**间**平均 —— 勿混。",
        "- EMA 是训练稳定，不是图像处理支路。",
        "- AUC/PR 扫全阈值，与单一硬阈值无关。",
        "- ConvNeXt 变体：Val 高、Ext 低 → 过拟合敏感性，不作 Method-B。",
        "",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="paper_v4 baseline2 metrics tables")
    parser.add_argument(
        "--threshold-policy",
        choices=("youden_on_split", "youden_on_val", "fixed0.5"),
        default="youden_on_split",
        help="硬分类阈值策略（默认与 v2 表一致：评估集 Youden）",
    )
    parser.add_argument(
        "--project-root",
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--include-appendix",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--tag",
        default="",
        help="输出文件名后缀，例如 youden_on_val → *_youden_on_val.csv",
    )
    args = parser.parse_args()
    root = args.project_root
    tag = args.tag.strip()
    if not tag:
        tag = args.threshold_policy
    suffix = f"_{tag}" if tag else ""

    experiments = experiments_for_tables(include_appendix=args.include_appendix)
    detail_parts: List[pd.DataFrame] = []
    incomplete: List[str] = []

    for exp in experiments:
        run_dir = os.path.join(root, exp.rel_dir)
        if not os.path.isdir(run_dir):
            incomplete.append(f"{exp.display}: 目录不存在 → {exp.rel_dir}")
            print(f"[SKIP] missing dir {exp.rel_dir}")
            continue
        rows = collect_seed_rows(exp, root, args.threshold_policy)
        if not rows:
            incomplete.append(f"{exp.display}: 无任何预测 CSV")
            print(f"[SKIP] no preds {exp.display}")
            continue
        df = pd.DataFrame(rows)
        n_ext = df[df["split"] == "external"]["seed"].nunique()
        if n_ext < len(SEEDS):
            missing = sorted(set(SEEDS) - set(df[df["split"] == "external"]["seed"]))
            incomplete.append(
                f"{exp.display}: Ext seeds={n_ext}/5，缺 {missing}"
            )
            print(f"[PARTIAL] {exp.display} ext {n_ext}/5 missing={missing}")
        else:
            print(f"[OK] {exp.display} ext 5/5")
        detail_parts.append(df)

    if not detail_parts:
        raise SystemExit("No experiments processed.")

    detail = pd.concat(detail_parts, ignore_index=True)
    metric_keys = [
        "auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "youden",
        "adjusted_bal_acc",
        "adjusted_f1",
        "threshold",
    ]
    summary = summarize_group(detail, metric_keys)

    out_dir = os.path.join(root, OUT_TABLES)
    os.makedirs(out_dir, exist_ok=True)

    detail_path = os.path.join(out_dir, f"baseline2_metrics_detail{suffix}.csv")
    summary_path = os.path.join(out_dir, f"baseline2_metrics_summary{suffix}.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[OK] detail → {detail_path}")
    print(f"[OK] summary → {summary_path}")

    # Youden-focused Ext views
    youden_like = summary.copy()
    paired = paired_wilcoxon_vs_ours(detail)
    paired_path = os.path.join(out_dir, f"baseline2_paired_auc_wilcoxon{suffix}.csv")
    if not paired.empty:
        paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
        print(f"[OK] paired → {paired_path}")

    md_path = write_paper_md(
        out_dir,
        summary,
        youden_like,
        paired,
        args.threshold_policy,
        incomplete,
        suffix=suffix,
    )
    print(f"[OK] markdown → {md_path}")

    # Canonical alias for default policy (easy for paper writers)
    if args.threshold_policy == "youden_on_split":
        alias = os.path.join(out_dir, "PAPER_TABLES_baseline2.md")
        with open(md_path, "r", encoding="utf-8") as src, open(
            alias, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())
        print(f"[OK] alias → {alias}")

    # policy stamp
    meta = pd.DataFrame(
        [
            {
                "threshold_policy": args.threshold_policy,
                "tag": tag,
                "champion": champion().display,
                "n_experiments": len(detail["model"].unique()),
                "incomplete": "; ".join(incomplete) if incomplete else "",
            }
        ]
    )
    meta.to_csv(
        os.path.join(out_dir, f"baseline2_run_meta{suffix}.csv"),
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
