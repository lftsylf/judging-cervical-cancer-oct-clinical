#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 帧辅损探路：提取外部 ROC，并与 edl@0.2 对照，决定是否建议跑 edl@0.3。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

HOSPITALS = ("huaxi", "liaoning", "xiangya")
SEED = "42"


def _metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    y = df["y_true"].astype(int).values
    p = df["prob_positive"].astype(float).values
    pred = (p > 0.5).astype(int)
    fw = df[[c for c in df.columns if c.startswith("frame_w_")]].values
    fu = df[[c for c in df.columns if c.startswith("frame_u_")]].values
    return {
        "roc": float(roc_auc_score(y, p)),
        "pr": float(average_precision_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "neff": float(np.mean(1.0 / (fw**2).sum(1))),
        "fu_std": float(np.mean(fu.std(1))),
    }


def collect(root: Path) -> list[dict]:
    rows = []
    for h in HOSPITALS:
        for split in ("val", "external"):
            p = root / h / f"seed_{SEED}" / "logs" / f"{split}_sample_predictions.csv"
            if not p.exists():
                raise FileNotFoundError(str(p))
            m = _metrics(p)
            rows.append({"hospital": h, "split": split, **m})
    return rows


def mean_ext_roc(rows: list[dict]) -> float:
    return float(np.mean([r["roc"] for r in rows if r["split"] == "external"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-root", required=True, help="刚跑完的 T2 输出根目录")
    ap.add_argument(
        "--ref-root",
        default="outputs/paper_v4/baseline/测试t2/ours_uw_frameaux_t2",
        help="对照：edl@0.2 Full T2（CSV 在 测试t2/ 下）",
    )
    ap.add_argument(
        "--gap-threshold",
        type=float,
        default=0.02,
        help="若 ref_ext_mean - cand_ext_mean >= 该值，则判定「差得远」并建议跑 edl@0.3",
    )
    ap.add_argument("--report", required=True, help="写入 markdown 报告路径")
    ap.add_argument("--decision-json", required=True, help="写入决策 JSON（供 shell 读取）")
    ap.add_argument("--candidate-name", default="candidate")
    args = ap.parse_args()

    project = Path(__file__).resolve().parents[2]
    cand_root = Path(args.candidate_root)
    if not cand_root.is_absolute():
        cand_root = project / cand_root
    ref_root = Path(args.ref_root)
    if not ref_root.is_absolute():
        ref_root = project / ref_root

    cand = collect(cand_root)
    ref = collect(ref_root)
    cand_mean = mean_ext_roc(cand)
    ref_mean = mean_ext_roc(ref)
    gap = ref_mean - cand_mean
    # 「差得远」：明显低于 edl@0.2；若接近或更好则不跑 0.3
    run_edl03 = gap >= float(args.gap_threshold)
    neff_mean = float(np.mean([r["neff"] for r in cand if r["split"] == "external"]))

    lines = [
        f"# 自动探路报告 · {args.candidate_name}",
        "",
        f"- candidate: `{cand_root}`",
        f"- 对照 edl@0.2: `{ref_root}`",
        f"- 判定阈值: ref − cand ≥ **{args.gap_threshold}** → 启动 edl@0.3",
        "",
        "## 外部 ROC（seed=42）",
        "",
        "| 中心 | 对照 edl@0.2 | candidate | Δ(cand−ref) |",
        "|------|-------------:|----------:|------------:|",
    ]
    ref_by = {r["hospital"]: r for r in ref if r["split"] == "external"}
    cand_by = {r["hospital"]: r for r in cand if r["split"] == "external"}
    for h in HOSPITALS:
        a, b = ref_by[h]["roc"], cand_by[h]["roc"]
        lines.append(f"| {h} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    lines += [
        "",
        f"- **对照三折均值**: {ref_mean:.4f}",
        f"- **candidate 三折均值**: {cand_mean:.4f}",
        f"- **gap (ref−cand)**: {gap:+.4f}",
        f"- **candidate 外部 n_eff 均值**: {neff_mean:.3f}（≈12 表示加权仍近等权）",
        "",
        "## 判决",
        "",
    ]
    if run_edl03:
        lines.append(
            f"**启动 edl@0.3**：candidate 相对 edl@0.2 仍差 ≥ {args.gap_threshold}。"
        )
    else:
        lines.append(
            f"**不启动 edl@0.3**：candidate 与 edl@0.2 差距 < {args.gap_threshold}"
            f"（接近或更好）。"
        )
    lines.append("")

    report = Path(args.report)
    if not report.is_absolute():
        report = project / report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")

    decision = {
        "candidate_name": args.candidate_name,
        "candidate_ext_mean": cand_mean,
        "ref_edl02_ext_mean": ref_mean,
        "gap_ref_minus_cand": gap,
        "gap_threshold": float(args.gap_threshold),
        "candidate_neff_mean": neff_mean,
        "run_edl_w03": bool(run_edl03),
        "report": str(report),
    }
    dec = Path(args.decision_json)
    if not dec.is_absolute():
        dec = project / dec
    dec.parent.mkdir(parents=True, exist_ok=True)
    dec.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"REPORT={report}")
    if run_edl03:
        print("DECISION=RUN_EDL_W03")
    else:
        print("DECISION=STOP")


if __name__ == "__main__":
    main()
