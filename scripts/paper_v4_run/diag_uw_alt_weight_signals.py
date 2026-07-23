#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B · 换权重信号的推理后处理诊断（固定 edl@0.3 权重，不重训）。

一次前向缓存帧级 α 与融合特征，再用多种权重公式离线重聚合。
必须与现有 CSV（τ=0.5 EDL-u UW）对齐校验，再与 equal / 锐化对照。
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
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.lancet_config import Config  # noqa: E402
from data.dataset_lancet import get_dataloader, unpack_loader_batch  # noqa: E402
from models.optigenesis_model import OptiGenesis  # noqa: E402


def _dirichlet_u(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[-1]
    s = torch.sum(alpha, dim=-1).clamp_min(1e-8)
    return float(k) / s


def _frame_probs(alpha: torch.Tensor) -> torch.Tensor:
    s = torch.sum(alpha, dim=-1, keepdim=True).clamp_min(1e-8)
    return alpha / s


def _softmax_scores(scores: torch.Tensor, tau: float) -> torch.Tensor:
    t = max(float(tau), 1e-8)
    return F.softmax(scores / t, dim=1)


def _sparsemax(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Martins & Astudillo sparsemax（按 dim 归一，支持 [N,F]）。"""
    # 排序
    z = logits
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    k = k.view(*view)
    support = (1 + k * z_sorted) > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1).to(z.dtype)
    # τ = (cumsum_at_kz - 1) / kz
    # gather cumsum at k_z-1
    idx = (k_z.long() - 1).clamp_min(0)
    tau = (z_cumsum.gather(dim, idx) - 1) / k_z
    return torch.clamp(z - tau, min=0.0)


def weights_equal(n: int, f: int, device, dtype) -> torch.Tensor:
    return torch.full((n, f), 1.0 / float(f), device=device, dtype=dtype)


def weights_edl_u(alpha: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    conf = (1.0 - _dirichlet_u(alpha)).clamp(min=0.0)
    return _softmax_scores(conf, tau)


def weights_max_prob(alpha: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    p = _frame_probs(alpha)
    return _softmax_scores(p.max(dim=-1).values, tau)


def weights_neg_entropy(alpha: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    p = _frame_probs(alpha).clamp_min(1e-8)
    ent = -(p * p.log()).sum(dim=-1)  # [N,F]，越小越确定
    # 用 -H 作分数；再按帧内标准化减轻尺度问题
    score = -ent
    score = score - score.mean(dim=1, keepdim=True)
    return _softmax_scores(score, tau)


def weights_evidence(alpha: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    # S=Σα 越大 → 证据越强（与 1/u 同向，但用 S 本身）
    s = torch.sum(alpha, dim=-1)
    score = s - s.mean(dim=1, keepdim=True)
    return _softmax_scores(score, tau)


def weights_feat_close(feat: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    # 越靠近患者均值 → 权重越高（抑制离群帧）
    mean = feat.mean(dim=1, keepdim=True)
    dist = torch.norm(feat - mean, dim=-1)  # [N,F]
    score = -dist
    score = score - score.mean(dim=1, keepdim=True)
    return _softmax_scores(score, tau)


def weights_feat_far(feat: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    # 越远离均值 → 权重越高（强调特异帧）
    mean = feat.mean(dim=1, keepdim=True)
    dist = torch.norm(feat - mean, dim=-1)
    score = dist - dist.mean(dim=1, keepdim=True)
    return _softmax_scores(score, tau)


def weights_sparsemax_edl(alpha: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    conf = (1.0 - _dirichlet_u(alpha)).clamp(min=0.0)
    # scale 放大微小差异，使 sparsemax 有机会稀疏
    return _sparsemax(conf * float(scale), dim=1)


def weights_topk_maxprob(alpha: torch.Tensor, k: int = 3) -> torch.Tensor:
    p = _frame_probs(alpha)
    conf = p.max(dim=-1).values
    n, f = conf.shape
    kk = min(int(k), f)
    idx = torch.topk(conf, k=kk, dim=1, largest=True).indices
    w = torch.zeros_like(conf)
    ones = torch.full((n, kk), 1.0 / float(kk), device=conf.device, dtype=conf.dtype)
    w.scatter_(1, idx, ones)
    return w


def aggregate(alpha_frame: torch.Tensor, frame_w: torch.Tensor) -> torch.Tensor:
    return torch.sum(frame_w.unsqueeze(-1) * alpha_frame, dim=1)


def metrics(alpha: torch.Tensor, y: np.ndarray, frame_w: torch.Tensor) -> dict:
    s = torch.sum(alpha, dim=1, keepdim=True).clamp_min(1e-8)
    p = (alpha / s)[:, 1].detach().cpu().numpy()
    fw = frame_w.detach().cpu().numpy()
    return {
        "roc": float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "pr": float(average_precision_score(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "neff": float(np.mean(1.0 / np.clip((fw**2).sum(1), 1e-12, None))),
        "w_max_mean": float(fw.max(1).mean()),
        "w_std_mean": float(fw.std(1).mean()),
    }


def collect(model, loader, device):
    model.eval()
    alphas, feats, ys = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs, clinical, labels, frame_mask = unpack_loader_batch(batch)
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            frame_mask = frame_mask.to(device)
            b, n_images, c, h, w = imgs.shape
            v_feat = model.vision_backbone(imgs.view(b * n_images, c, h, w)).view(b, n_images, -1)
            alpha_frame, feat_fused, _, _ = model._frame_alphas_and_features(
                v_feat, clinical, frame_mask=frame_mask
            )
            alphas.append(alpha_frame.cpu())
            feats.append(feat_fused.cpu())
            ys.append(labels.numpy())
    return torch.cat(alphas, 0), torch.cat(feats, 0), np.concatenate(ys, 0)


def load_model(ckpt: Path, device) -> OptiGenesis:
    model = OptiGenesis(
        model_name=Config.BACKBONE,
        use_clinical=bool(Config.USE_CLINICAL),
        num_classes=Config.NUM_CLASSES,
        frame_agg_mode="uncertainty_weighted",
        agg_temperature=0.5,
        review_top_k=3,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def csv_roc(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return float(roc_auc_score(df["y_true"].astype(int), df["prob_positive"].astype(float)))


def build_modes(alpha: torch.Tensor, feat: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """返回 (mode_name, frame_w)。含基线与若干替代信号。"""
    modes = [
        ("equal", weights_equal(alpha.shape[0], alpha.shape[1], alpha.device, alpha.dtype)),
        ("edl_u_tau0.5", weights_edl_u(alpha, 0.5)),  # 训练默认，须对齐 CSV
        ("edl_u_tau0.01", weights_edl_u(alpha, 0.01)),  # 锐化对照（已知伤 AUC）
        ("maxprob_tau0.5", weights_max_prob(alpha, 0.5)),
        ("maxprob_tau0.1", weights_max_prob(alpha, 0.1)),
        ("maxprob_tau0.01", weights_max_prob(alpha, 0.01)),
        ("negent_tau0.5", weights_neg_entropy(alpha, 0.5)),
        ("negent_tau0.1", weights_neg_entropy(alpha, 0.1)),
        ("negent_tau0.01", weights_neg_entropy(alpha, 0.01)),
        ("evidence_tau0.5", weights_evidence(alpha, 0.5)),
        ("evidence_tau0.01", weights_evidence(alpha, 0.01)),
        ("feat_close_tau0.5", weights_feat_close(feat, 0.5)),
        ("feat_close_tau0.1", weights_feat_close(feat, 0.1)),
        ("feat_close_tau0.01", weights_feat_close(feat, 0.01)),
        ("feat_far_tau0.5", weights_feat_far(feat, 0.5)),
        ("feat_far_tau0.1", weights_feat_far(feat, 0.1)),
        ("feat_far_tau0.01", weights_feat_far(feat, 0.01)),
        ("sparsemax_edl_x20", weights_sparsemax_edl(alpha, 20.0)),
        ("sparsemax_edl_x50", weights_sparsemax_edl(alpha, 50.0)),
        ("topk3_maxprob", weights_topk_maxprob(alpha, 3)),
        ("topk1_maxprob", weights_topk_maxprob(alpha, 1)),
    ]
    return modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        default="outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3",
    )
    ap.add_argument("--seed", default="42")
    ap.add_argument("--hospitals", default="huaxi,liaoning,xiangya")
    ap.add_argument("--splits", default="val,external")
    ap.add_argument(
        "--report",
        default="scripts/paper_v4_run/DIAG_uw_alt_weight_signals_edl03.md",
    )
    ap.add_argument(
        "--json-out",
        default="outputs/paper_v4/baseline/DIAG_uw_alt_weight_signals_edl03.json",
    )
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OPTIGENESIS_USE_CLINICAL", "0")
    Config.USE_CLINICAL = False
    Config.BACKBONE = os.getenv("OPTIGENESIS_BACKBONE", "resnet50")

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    hospitals = [h.strip() for h in args.hospitals.split(",") if h.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict] = []
    sanity: list[dict] = []

    for hosp in hospitals:
        ckpt = run_root / hosp / f"seed_{args.seed}" / "checkpoints" / "best_model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"\n=== {hosp} | {ckpt} ===")
        model = load_model(ckpt, device)

        for split in splits:
            csv_path = Path(Config.DATA_ROOT) / f"{split}_{hosp}.csv"
            loader = get_dataloader(str(csv_path), mode="val")
            print(f"  collect α+feat: {split} …")
            alpha, feat, y = collect(model, loader, device)
            print(f"    N={len(y)} F={alpha.shape[1]} D={feat.shape[2]}")

            pred_csv = (
                run_root / hosp / f"seed_{args.seed}" / "logs" / f"{split}_sample_predictions.csv"
            )
            roc_ref = csv_roc(pred_csv)

            for mode, fw in build_modes(alpha, feat):
                ap_pat = aggregate(alpha, fw)
                m = metrics(ap_pat, y, fw)
                row = {
                    "hospital": hosp,
                    "split": split,
                    "mode": mode,
                    **m,
                }
                rows.append(row)
                if mode == "edl_u_tau0.5":
                    delta = None if roc_ref is None else m["roc"] - roc_ref
                    sanity.append(
                        {
                            "hospital": hosp,
                            "split": split,
                            "reagg_roc": m["roc"],
                            "csv_roc": roc_ref,
                            "delta": delta,
                            "ok": roc_ref is not None and abs(delta) < 1e-6,
                        }
                    )
                    tag = "OK" if sanity[-1]["ok"] else f"Δ={delta}"
                    print(
                        f"    [sanity] edl_u_tau0.5 reagg={m['roc']:.6f} "
                        f"csv={roc_ref} → {tag}"
                    )
                print(
                    f"    {mode:<22} ROC={m['roc']:.4f} PR={m['pr']:.4f} "
                    f"n_eff={m['neff']:.3f} w_max={m['w_max_mean']:.3f}"
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    ext = df[df["split"] == "external"].copy()
    summary = (
        ext.groupby("mode", sort=False)
        .agg(
            roc_mean=("roc", "mean"),
            roc_std=("roc", "std"),
            pr_mean=("pr", "mean"),
            neff_mean=("neff", "mean"),
            w_max_mean=("w_max_mean", "mean"),
        )
        .reset_index()
    )
    base = float(summary.loc[summary["mode"] == "edl_u_tau0.5", "roc_mean"].iloc[0])
    eq = float(summary.loc[summary["mode"] == "equal", "roc_mean"].iloc[0])
    summary["delta_vs_edl_u"] = summary["roc_mean"] - base
    summary["delta_vs_equal"] = summary["roc_mean"] - eq

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    json_path = Path(args.json_out)
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# B · 换权重信号推理后处理 · edl@0.3 固定权重",
        "",
        f"- 权重根: `{run_root}`",
        "- 对照：训练默认 `edl_u_tau0.5`（须与 `*_sample_predictions.csv` ROC 对齐）",
        "- 另对照：`equal`（等权）与 `edl_u_tau0.01`（锐化，已知伤 AUC）",
        "",
        "## 对齐校验（edl_u_tau0.5 vs 现有 CSV）",
        "",
        "| 中心 | split | 重聚合 ROC | CSV ROC | |Δ| < 1e-6 |",
        "|------|-------|-----------:|--------:|:---------:|",
    ]
    all_ok = True
    for s in sanity:
        ok = bool(s["ok"])
        all_ok = all_ok and ok
        lines.append(
            f"| {s['hospital']} | {s['split']} | {s['reagg_roc']:.6f} | "
            f"{s['csv_roc']:.6f} | {'✅' if ok else '❌'} |"
        )
    lines += [
        "",
        f"**对齐结论**: {'全部通过' if all_ok else '存在偏差，结果不可信，需先查管线'}",
        "",
        "## 外部三折汇总（相对默认 EDL-u）",
        "",
        "| 模式 | ROC 均值 | Δ vs edl_u | Δ vs equal | n_eff | w_max |",
        "|------|---------:|-----------:|-----------:|------:|------:|",
    ]
    # 按 ROC 降序展示更易读，但保留 baseline 靠前：先按 delta 再按 roc
    show = summary.sort_values("roc_mean", ascending=False)
    for _, r in show.iterrows():
        lines.append(
            f"| {r['mode']} | {r['roc_mean']:.4f} | {r['delta_vs_edl_u']:+.4f} | "
            f"{r['delta_vs_equal']:+.4f} | {r['neff_mean']:.3f} | {r['w_max_mean']:.3f} |"
        )

    best = show.iloc[0]
    lines += [
        "",
        "## 分中心 external ROC",
        "",
    ]
    pivot = ext.pivot_table(index="mode", columns="hospital", values="roc", aggfunc="first")
    # 按均值排序
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False)
    hosp_cols = [c for c in ("huaxi", "liaoning", "xiangya") if c in pivot.columns]
    lines.append("| 模式 | " + " | ".join(hosp_cols) + " | 均值 |")
    lines.append("|------|" + "|".join(["-------:"] * len(hosp_cols)) + "|------:|")
    for mode, r in pivot.iterrows():
        cells = " | ".join(f"{r[h]:.4f}" for h in hosp_cols)
        lines.append(f"| {mode} | {cells} | {r['_mean']:.4f} |")

    # 自动结论
    beat_base = summary[summary["delta_vs_edl_u"] > 0.005]
    lines += [
        "",
        "## 结论",
        "",
        f"- 对齐校验: **{'通过' if all_ok else '失败'}**",
        f"- 默认 EDL-u τ=0.5 外部 ROC = **{base:.4f}**；equal = **{eq:.4f}**"
        f"（Δ(edl−equal)={base - eq:+.4f}）",
        f"- 本轮最高: **{best['mode']}** = {best['roc_mean']:.4f}"
        f"（Δ vs edl_u = {best['delta_vs_edl_u']:+.4f}）",
    ]
    if len(beat_base) == 0:
        lines.append(
            "- **无**模式相对默认 EDL-u 提升 >0.005 → 换权重公式的后处理**未能**救 UW；"
            "下一优先考虑可学习注意力（D）或改监督（C）。"
        )
    else:
        names = ", ".join(beat_base.sort_values("delta_vs_edl_u", ascending=False)["mode"].tolist())
        lines.append(f"- 相对默认有提升（>0.005）的模式: {names} → 可考虑写入训练/默认推理。")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "run_root": str(run_root),
        "sanity": sanity,
        "all_sanity_ok": all_ok,
        "baseline_edl_u_roc": base,
        "equal_roc": eq,
        "external_summary": summary.to_dict(orient="records"),
        "rows": rows,
        "report": str(report_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nREPORT={report_path}")
    print(f"JSON={json_path}")
    print(summary.sort_values("roc_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
