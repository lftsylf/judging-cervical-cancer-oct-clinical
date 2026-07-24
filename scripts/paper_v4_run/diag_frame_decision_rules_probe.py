#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
决策/聚合规则探针（不重训）：固定 checkpoint，取出帧级 p，对比多种患者级分数。

规则：
  - patient_uw     : 模型默认 UW 聚合后的患者 p（与训练 CSV 对齐校验）
  - mean_p         : 有效帧 p 均值
  - max_p          : 有效帧 p 最大值（软 OR）
  - topk3_mean_p   : 最高 3 帧 p 均值
  - topk5_mean_p   : 最高 5 帧 p 均值
  - hard_or_05     : 任一帧 p>0.5 → 1 否则 0（硬 OR；ROC 用此二值会退化）
  - hybrid_or_uw   : 任一帧 p>0.5 → 用 max_p，否则用 patient_uw（你提的规则的连续版）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    recall_score,
    precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.lancet_config import Config  # noqa: E402
from data.dataset_lancet import get_dataloader, unpack_loader_batch  # noqa: E402
from models.optigenesis_model import OptiGenesis  # noqa: E402

HOSPITALS = ("huaxi", "liaoning", "xiangya")


def _metrics_cont(y, p) -> dict:
    pred = (p > 0.5).astype(int)
    out = {
        "roc": float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "pr": float(average_precision_score(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "sens": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "spec": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "ppv": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "pos_rate": float(pred.mean()),
    }
    return out


def _masked_frame_p(alpha_frame: torch.Tensor, mask: torch.Tensor | None) -> tuple[np.ndarray, np.ndarray]:
    """alpha [N,F,K] → p_pos [N,F], mask [N,F] float"""
    s = alpha_frame.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    p = (alpha_frame / s)[..., 1].cpu().numpy()
    if mask is None:
        m = np.ones_like(p, dtype=np.float32)
    else:
        m = mask.cpu().numpy().astype(np.float32)
    return p, m


def aggregate_scores(p_frame: np.ndarray, mask: np.ndarray, p_uw: np.ndarray) -> dict[str, np.ndarray]:
    """p_frame/mask: [N,F]"""
    n, f = p_frame.shape
    # 无效帧置 -inf 便于 max/topk
    p_eff = np.where(mask > 0.5, p_frame, -np.inf)
    n_eff = mask.sum(1).clip(min=1.0)

    mean_p = np.where(mask > 0.5, p_frame, 0.0).sum(1) / n_eff
    max_p = p_eff.max(1)
    # topk
    def topk_mean(k):
        # 对 -inf 的帧，partition 仍可用；用 masked count
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            vals = p_frame[i][mask[i] > 0.5]
            if len(vals) == 0:
                out[i] = 0.0
                continue
            kk = min(k, len(vals))
            out[i] = np.sort(vals)[-kk:].mean()
        return out

    any_pos = (p_eff > 0.5).any(axis=1)
    hard_or = any_pos.astype(np.float64)
    hybrid = np.where(any_pos, max_p, p_uw)

    return {
        "patient_uw": p_uw.astype(np.float64),
        "mean_p": mean_p.astype(np.float64),
        "max_p": max_p.astype(np.float64),
        "topk3_mean_p": topk_mean(3),
        "topk5_mean_p": topk_mean(5),
        "hard_or_05": hard_or,
        "hybrid_or_uw": hybrid.astype(np.float64),
    }


def collect_frames(model, loader, device):
    model.eval()
    alphas, masks, ys, p_uws = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, clinical, labels, frame_mask = unpack_loader_batch(batch)
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            frame_mask = frame_mask.to(device)
            alpha, details = model(
                imgs, clinical, return_frame_details=True, frame_mask=frame_mask
            )
            s = alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
            p_uw = (alpha / s)[:, 1].cpu().numpy()
            fa = details["frame_alpha"]
            if fa is None:
                raise RuntimeError("frame_alpha is None — 需要 equal/uw 模式")
            alphas.append(fa.cpu())
            masks.append(frame_mask.cpu())
            ys.append(labels.numpy())
            p_uws.append(p_uw)
    return (
        torch.cat(alphas, 0),
        torch.cat(masks, 0),
        np.concatenate(ys),
        np.concatenate(p_uws),
    )


def load_model(ckpt: Path, device, weight_signal: str, u_base=0.5, u_scale=10.0, chunk=16):
    model = OptiGenesis(
        model_name=Config.BACKBONE,
        use_clinical=False,
        num_classes=2,
        frame_agg_mode="uncertainty_weighted",
        agg_temperature=0.5,
        review_top_k=3,
        weight_signal=weight_signal,
        u_score_base=u_base,
        u_score_scale=u_scale,
        frame_encode_chunk=chunk,
    ).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def csv_roc(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return float(roc_auc_score(df["y_true"].astype(int), df["prob_positive"].astype(float)))


def run_setting(
    name: str,
    run_root: Path,
    expand: bool,
    max_pages: int,
    weight_signal: str,
    batch_size: int,
    chunk: int,
    device: torch.device,
    splits: list[str],
) -> list[dict]:
    # 覆盖 Config（dataloader 会读这些）
    Config.USE_CLINICAL = False
    Config.BACKBONE = "resnet50"
    Config.EXPAND_TIFF_PAGES = bool(expand)
    Config.MAX_PAGES_PER_TIFF = int(max_pages)
    Config.FRAME_ENCODE_CHUNK = int(chunk)
    Config.BATCH_SIZE = int(batch_size)
    # get_dataloader 内部用 Config.BATCH_SIZE？检查
    rows = []
    for hosp in HOSPITALS:
        ckpt = run_root / hosp / "seed_42" / "checkpoints" / "best_model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"\n=== [{name}] {hosp} | {ckpt} ===")
        model = load_model(
            ckpt,
            device,
            weight_signal=weight_signal,
            chunk=chunk,
        )
        for split in splits:
            csv_path = Path(Config.DATA_ROOT) / f"{split}_{hosp}.csv"
            # 临时改 batch：get_dataloader 看 Config
            loader = get_dataloader(str(csv_path), mode="val")
            print(f"  {split}: collect frame α … (expand={expand}, max_pages={max_pages})")
            alpha_f, mask, y, p_uw = collect_frames(model, loader, device)
            p_frame, m_np = _masked_frame_p(alpha_f, mask)
            scores = aggregate_scores(p_frame, m_np, p_uw)

            # 帧差异诊断
            p_std = []
            for i in range(len(y)):
                vals = p_frame[i][m_np[i] > 0.5]
                p_std.append(float(vals.std()) if len(vals) > 1 else 0.0)
            any_pos_rate = float((scores["hard_or_05"] > 0.5).mean())

            pred_csv = run_root / hosp / "seed_42" / "logs" / f"{split}_sample_predictions.csv"
            ref = csv_roc(pred_csv)
            uw_m = _metrics_cont(y, scores["patient_uw"])
            ok = ref is not None and abs(uw_m["roc"] - ref) < 1e-4
            print(
                f"    sanity UW ROC={uw_m['roc']:.4f} vs CSV={ref} → {'OK' if ok else 'MISMATCH'}"
                f" | frame_p std(mean)={np.mean(p_std):.4f} | hard_or pos_rate={any_pos_rate:.3f}"
            )

            for rule, p in scores.items():
                m = _metrics_cont(y, p)
                rows.append(
                    {
                        "setting": name,
                        "hospital": hosp,
                        "split": split,
                        "rule": rule,
                        **m,
                        "frame_p_std_mean": float(np.mean(p_std)),
                        "n_frames_mean": float(m_np.sum(1).mean()),
                        "sanity_ok": bool(ok) if rule == "patient_uw" else None,
                        "csv_roc": ref if rule == "patient_uw" else None,
                    }
                )
                print(
                    f"    {rule:16s} ROC={m['roc']:.4f} PR={m['pr']:.4f} "
                    f"Sens={m['sens']:.3f} Spec={m['spec']:.3f} pos_rate={m['pos_rate']:.3f}"
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--splits", default="val,external")
    ap.add_argument(
        "--report",
        default="scripts/paper_v4_run/DIAG_frame_decision_rules_probe.md",
    )
    ap.add_argument(
        "--json-out",
        default="outputs/paper_v4/baseline/DIAG_frame_decision_rules_probe.json",
    )
    ap.add_argument("--skip-pages5", action="store_true")
    ap.add_argument("--skip-n12", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OPTIGENESIS_USE_CLINICAL", "0")
    Config.USE_CLINICAL = False
    Config.BACKBONE = "resnet50"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    base = PROJECT_ROOT / "outputs/paper_v4/baseline"
    all_rows: list[dict] = []

    if not args.skip_n12:
        all_rows += run_setting(
            name="N12_edl03",
            run_root=base / "ours_uw_frameaux_t2_edl_w0.3",
            expand=False,
            max_pages=0,
            weight_signal="edl_u",
            batch_size=4,
            chunk=16,
            device=device,
            splits=splits,
        )

    if not args.skip_pages5:
        all_rows += run_setting(
            name="pages5_uamp",
            run_root=base / "ours_uw_frameaux_t2_edl_w0.3_expand_uamp10_pages5",
            expand=True,
            max_pages=5,
            weight_signal="edl_u_amp",
            batch_size=2,
            chunk=8,
            device=device,
            splits=splits,
        )

    df = pd.DataFrame(all_rows)
    report = Path(args.report)
    if not report.is_absolute():
        report = PROJECT_ROOT / report
    json_path = Path(args.json_out)
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path
    report.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 帧决策/聚合规则探针（不重训）",
        "",
        "固定 checkpoint，离线对比患者级分数规则。",
        "",
        "## 规则说明",
        "",
        "| 规则 | 含义 |",
        "|------|------|",
        "| patient_uw | 训练时默认 UW 聚合患者 p（应对齐 CSV） |",
        "| mean_p | 有效帧阳性概率均值 |",
        "| max_p | 有效帧 max p（软 OR） |",
        "| topk3/5_mean_p | 最高 k 帧 p 均值 |",
        "| hard_or_05 | 任一帧 p>0.5 → 阳（二值；ROC 参考价值有限） |",
        "| hybrid_or_uw | **你提的规则连续版**：任一帧>0.5 用 max_p，否则用 UW p |",
        "",
    ]

    for setting in df["setting"].unique():
        lines.append(f"## {setting}")
        lines.append("")
        sub = df[df["setting"] == setting]
        # sanity
        san = sub[sub["rule"] == "patient_uw"]
        lines.append("### 对齐校验（patient_uw vs CSV）")
        lines.append("")
        lines.append("| 中心 | split | UW ROC | CSV ROC | OK |")
        lines.append("|------|-------|-------:|--------:|:--:|")
        for _, r in san.iterrows():
            lines.append(
                f"| {r['hospital']} | {r['split']} | {r['roc']:.4f} | "
                f"{r['csv_roc'] if r['csv_roc'] is not None else 'NA'} | "
                f"{'✅' if r['sanity_ok'] else '❌'} |"
            )
        lines.append("")

        ext = sub[sub["split"] == "external"]
        # 三折均值表
        summ = (
            ext.groupby("rule", sort=False)
            .agg(
                roc=("roc", "mean"),
                pr=("pr", "mean"),
                sens=("sens", "mean"),
                spec=("spec", "mean"),
                pos_rate=("pos_rate", "mean"),
                frame_p_std=("frame_p_std_mean", "mean"),
            )
            .reset_index()
        )
        base_roc = float(summ.loc[summ["rule"] == "patient_uw", "roc"].iloc[0])
        summ["d_roc"] = summ["roc"] - base_roc
        summ = summ.sort_values("roc", ascending=False)
        lines.append("### 外部三折均值")
        lines.append("")
        lines.append("| 规则 | ROC | Δ vs UW | PR | Sens | Spec | 预测阳性率 |")
        lines.append("|------|----:|--------:|---:|-----:|-----:|-----------:|")
        for _, r in summ.iterrows():
            lines.append(
                f"| {r['rule']} | {r['roc']:.4f} | {r['d_roc']:+.4f} | {r['pr']:.4f} | "
                f"{r['sens']:.3f} | {r['spec']:.3f} | {r['pos_rate']:.3f} |"
            )
        lines.append("")
        lines.append(
            f"- 帧 p 患者内 std（三折均值）≈ **{float(ext['frame_p_std_mean'].mean()):.4f}**"
            f"（越小说明 OR/max 越接近 UW）"
        )
        lines.append("")

        # 分中心 external ROC 透视
        lines.append("### 分中心 external ROC")
        lines.append("")
        piv = ext.pivot_table(index="rule", columns="hospital", values="roc", aggfunc="first")
        piv["_mean"] = piv.mean(axis=1)
        piv = piv.sort_values("_mean", ascending=False)
        cols = [c for c in ("huaxi", "liaoning", "xiangya") if c in piv.columns]
        lines.append("| 规则 | " + " | ".join(cols) + " | 均值 |")
        lines.append("|------|" + "|".join(["-------:"] * len(cols)) + "|------:|")
        for rule, r in piv.iterrows():
            cells = " | ".join(f"{r[c]:.4f}" for c in cols)
            lines.append(f"| {rule} | {cells} | {r['_mean']:.4f} |")
        lines.append("")

        # val 均值（诚实参考）
        val = sub[sub["split"] == "val"]
        vsum = val.groupby("rule")["roc"].mean().sort_values(ascending=False)
        lines.append("### 内部 val ROC 均值（参考）")
        lines.append("")
        lines.append("| 规则 | val ROC |")
        lines.append("|------|--------:|")
        for rule, v in vsum.items():
            lines.append(f"| {rule} | {v:.4f} |")
        lines.append("")

    # 总结论
    lines += [
        "## 结论草稿",
        "",
        "- 若 `max_p` / `hybrid_or_uw` 相对 `patient_uw` 外部 ROC **无明显提升**（或伤 val），"
        "则「任一帧阳性即判阳」在当前模型上**不可行**（帧 p 太同质，OR≈患者级）。",
        "- `hard_or_05` 常抬高阳性率、伤特异度；湘雅若已近全阳则更无增益。",
        "- 本探针**不重训**；若某规则显著更好，再考虑是否写进推理或重训。",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps({"rows": all_rows, "report": str(report)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nREPORT={report}")
    print(f"JSON={json_path}")


if __name__ == "__main__":
    main()
