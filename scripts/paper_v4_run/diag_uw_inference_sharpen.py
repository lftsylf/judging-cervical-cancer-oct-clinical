#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理锐化诊断：固定已训好的 UW+FrameAux 权重，只改推理 τ / top-k。

一次前向取出帧级 α，再离线重聚合，避免重复跑骨干。
用于判断：帧 u 是否含可被锐化放大的信息。若 n_eff 能动但 AUC 不动（或仍近 12），
说明 u 本身信息量不足，需改监督/特征而非再扫 FrameAux weight。
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
from data.dataset_lancet import get_dataloader  # noqa: E402
from models.optigenesis_model import OptiGenesis  # noqa: E402

HOSPITALS = ("huaxi", "liaoning", "xiangya")
SPLITS = ("val", "external")


def _dirichlet_u(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[-1]
    s = torch.sum(alpha, dim=-1).clamp_min(1e-8)
    return float(k) / s


def reaggregate(
    alpha_frame: torch.Tensor,
    tau: float | None = None,
    topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    alpha_frame: [N, F, K]
    返回: alpha_patient [N,K], frame_u [N,F], frame_w [N,F]
    """
    frame_u = _dirichlet_u(alpha_frame)
    n, f, _ = alpha_frame.shape
    conf = (1.0 - frame_u).clamp(min=0.0)

    if topk is not None and topk > 0:
        k = min(int(topk), f)
        # 置信最高的 k 帧等权；其余 0
        idx = torch.topk(conf, k=k, dim=1, largest=True).indices  # [N,k]
        frame_w = torch.zeros_like(conf)
        ones = torch.full((n, k), 1.0 / float(k), device=conf.device, dtype=conf.dtype)
        frame_w.scatter_(1, idx, ones)
    else:
        t = max(float(tau if tau is not None else 0.5), 1e-8)
        frame_w = F.softmax(conf / t, dim=1)

    alpha = torch.sum(frame_w.unsqueeze(-1) * alpha_frame, dim=1)
    return alpha, frame_u, frame_w


def metrics_from_alpha(alpha: torch.Tensor, y: np.ndarray, frame_w: torch.Tensor) -> dict:
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


def collect_frame_alphas(model, loader, device) -> tuple[torch.Tensor, np.ndarray]:
    model.eval()
    alphas = []
    ys = []
    with torch.no_grad():
        for imgs, clinical, labels in loader:
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            # 直接走帧级路径拿 alpha_frame
            b, n_images, c, h, w = imgs.shape
            img_flat = imgs.view(b * n_images, c, h, w)
            v_feat = model.vision_backbone(img_flat).view(b, n_images, -1)
            alpha_frame, _, _, _ = model._frame_alphas_and_features(v_feat, clinical)
            alphas.append(alpha_frame.cpu())
            ys.append(labels.numpy())
    return torch.cat(alphas, dim=0), np.concatenate(ys, axis=0)


def load_model(ckpt: Path, device, tau: float = 0.5) -> OptiGenesis:
    model = OptiGenesis(
        model_name=Config.BACKBONE,
        use_clinical=bool(Config.USE_CLINICAL),
        num_classes=Config.NUM_CLASSES,
        frame_agg_mode="uncertainty_weighted",
        agg_temperature=float(tau),
        review_top_k=3,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def parse_list_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="UW 推理锐化诊断（固定权重，扫 τ / top-k）")
    ap.add_argument(
        "--run-root",
        default="outputs/paper_v4/baseline/ours_uw_frameaux_t2_edl_w0.3",
        help="含各中心 seed_*/checkpoints/best_model.pth 的运行根目录",
    )
    ap.add_argument("--seed", default="42")
    ap.add_argument("--taus", default="0.5,0.1,0.05,0.01,0.001")
    ap.add_argument("--topks", default="6,3,1", help="置信 top-k 硬选等权；空串则跳过")
    ap.add_argument("--hospitals", default="huaxi,liaoning,xiangya")
    ap.add_argument("--splits", default="val,external")
    ap.add_argument(
        "--report",
        default="scripts/paper_v4_run/DIAG_uw_inference_sharpen_edl03.md",
    )
    ap.add_argument(
        "--json-out",
        default="outputs/paper_v4/baseline/DIAG_uw_inference_sharpen_edl03.json",
    )
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # 与 T2 训练一致：OCT-only
    os.environ.setdefault("OPTIGENESIS_USE_CLINICAL", "0")
    # 重新读 Config 字段需在 import 后手动覆盖（Config 已在 import 时读过 env）
    Config.USE_CLINICAL = False
    Config.BACKBONE = os.getenv("OPTIGENESIS_BACKBONE", "resnet50")

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    hospitals = [h.strip() for h in args.hospitals.split(",") if h.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    taus = parse_list_floats(args.taus)
    topks = parse_list_ints(args.topks) if args.topks.strip() else []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    for hosp in hospitals:
        ckpt = run_root / hosp / f"seed_{args.seed}" / "checkpoints" / "best_model.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"缺少权重: {ckpt}")
        print(f"\n=== {hosp} | load {ckpt} ===")
        model = load_model(ckpt, device)

        for split in splits:
            csv_path = Path(Config.DATA_ROOT) / f"{split}_{hosp}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(str(csv_path))
            loader = get_dataloader(str(csv_path), mode="val")
            print(f"  collect frame α: {split} ({csv_path.name}) …")
            alpha_frame, y = collect_frame_alphas(model, loader, device)
            print(f"    N={len(y)} frames={alpha_frame.shape[1]} classes={alpha_frame.shape[2]}")

            # 基线 τ 与 CSV 对齐检查（仅 external/val 有现成 CSV 时）
            pred_csv = (
                run_root / hosp / f"seed_{args.seed}" / "logs" / f"{split}_sample_predictions.csv"
            )
            if pred_csv.exists() and 0.5 in taus:
                alpha0, _, fw0 = reaggregate(alpha_frame, tau=0.5)
                m0 = metrics_from_alpha(alpha0, y, fw0)
                df0 = pd.read_csv(pred_csv)
                roc_csv = float(roc_auc_score(df0["y_true"].astype(int), df0["prob_positive"].astype(float)))
                print(f"    sanity τ=0.5: reagg ROC={m0['roc']:.4f} vs CSV ROC={roc_csv:.4f}")

            for tau in taus:
                alpha, fu, fw = reaggregate(alpha_frame, tau=tau)
                m = metrics_from_alpha(alpha, y, fw)
                fu_np = fu.numpy()
                rows.append(
                    {
                        "hospital": hosp,
                        "split": split,
                        "mode": f"tau={tau}",
                        "tau": tau,
                        "topk": None,
                        "roc": m["roc"],
                        "pr": m["pr"],
                        "neff": m["neff"],
                        "w_max_mean": m["w_max_mean"],
                        "fu_std_mean": float(fu_np.std(1).mean()),
                        "fu_range_mean": float((fu_np.max(1) - fu_np.min(1)).mean()),
                    }
                )
                print(
                    f"    tau={tau:<7} ROC={m['roc']:.4f} PR={m['pr']:.4f} "
                    f"n_eff={m['neff']:.3f} w_max={m['w_max_mean']:.3f}"
                )

            for k in topks:
                alpha, fu, fw = reaggregate(alpha_frame, topk=k)
                m = metrics_from_alpha(alpha, y, fw)
                fu_np = fu.numpy()
                rows.append(
                    {
                        "hospital": hosp,
                        "split": split,
                        "mode": f"topk={k}",
                        "tau": None,
                        "topk": k,
                        "roc": m["roc"],
                        "pr": m["pr"],
                        "neff": m["neff"],
                        "w_max_mean": m["w_max_mean"],
                        "fu_std_mean": float(fu_np.std(1).mean()),
                        "fu_range_mean": float((fu_np.max(1) - fu_np.min(1)).mean()),
                    }
                )
                print(
                    f"    topk={k:<6} ROC={m['roc']:.4f} PR={m['pr']:.4f} "
                    f"n_eff={m['neff']:.3f} w_max={m['w_max_mean']:.3f}"
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # 外部三折均值表
    ext = df[df["split"] == "external"]
    summary = (
        ext.groupby("mode", sort=False)
        .agg(roc_mean=("roc", "mean"), roc_std=("roc", "std"), neff_mean=("neff", "mean"))
        .reset_index()
    )
    base = summary.loc[summary["mode"] == "tau=0.5", "roc_mean"]
    base_roc = float(base.iloc[0]) if len(base) else float("nan")
    summary["delta_vs_tau0.5"] = summary["roc_mean"] - base_roc

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    json_path = Path(args.json_out)
    if not json_path.is_absolute():
        json_path = PROJECT_ROOT / json_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# UW 推理锐化诊断 · edl@0.3 固定权重",
        "",
        f"- 权重根: `{run_root}`",
        f"- seed={args.seed}；USE_CLINICAL=0；backbone={Config.BACKBONE}",
        f"- τ 网格: {taus}",
        f"- top-k 硬选: {topks}",
        "",
        "## 外部 ROC / n_eff（三折）",
        "",
        "| 模式 | ROC 均值 | ROC std | Δ vs τ=0.5 | n_eff 均值 |",
        "|------|---------:|--------:|-----------:|-----------:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['mode']} | {r['roc_mean']:.4f} | {r['roc_std']:.4f} | "
            f"{r['delta_vs_tau0.5']:+.4f} | {r['neff_mean']:.3f} |"
        )

    lines += ["", "## 分中心明细（external）", ""]
    for mode in ext["mode"].unique():
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| 中心 | ROC | PR | n_eff | w_max | 帧u std | 帧u range |")
        lines.append("|------|----:|---:|------:|------:|--------:|----------:|")
        sub = ext[ext["mode"] == mode]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['hospital']} | {r['roc']:.4f} | {r['pr']:.4f} | {r['neff']:.3f} | "
                f"{r['w_max_mean']:.3f} | {r['fu_std_mean']:.4f} | {r['fu_range_mean']:.4f} |"
            )
        lines.append("")

    # 结论自动草拟
    best = summary.loc[summary["roc_mean"].idxmax()]
    neff_at_sharp = float(
        summary.loc[summary["mode"] == "tau=0.01", "neff_mean"].iloc[0]
    ) if (summary["mode"] == "tau=0.01").any() else float("nan")
    lines += [
        "## 自动结论草稿",
        "",
        f"- 外部最佳模式: **{best['mode']}**（ROC={best['roc_mean']:.4f}，"
        f"Δ={best['delta_vs_tau0.5']:+.4f}，n_eff={best['neff_mean']:.3f}）",
        f"- τ=0.01 时 n_eff≈{neff_at_sharp:.3f}（相对 12 是否明显下降）",
        "- 判读：若锐化后 n_eff≪12 但 ROC 几乎不动/下降 → **u 可拉开权重但无判别信息**；"
        "若 n_eff 仍≈12 → **u 差太小，连锐化也救不了**；"
        "若 ROC 明显升且 n_eff 降 → 可把更尖的 τ 作为推理默认，再考虑训练侧改动。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "run_root": str(run_root),
        "seed": args.seed,
        "taus": taus,
        "topks": topks,
        "rows": rows,
        "external_summary": summary.to_dict(orient="records"),
        "report": str(report_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nREPORT={report_path}")
    print(f"JSON={json_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
