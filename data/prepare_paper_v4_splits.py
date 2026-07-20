#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 v3 的 development_{hospital}.csv 生成 v4 的 train/val（患者/病例级 8:2 分层）。

协议：
  - 折名仍 = 外部测试医院（huaxi / liaoning / xiangya）
  - development = 其余两家医院（与 v3 相同）
  - 在 development 内按「中心 × 病理标签」分层，约 80% train / 20% val
    → 尽量使 train、val 的阴阳比例都接近 development 整体比例，
      且两家医院各自的阴阳比例也尽量保留
  - external 不参与划分，原样保留，仅作终评

安全措施：
  - 不覆盖 data/snapshots/paper_v3_tsy_loho/
  - 划分前校验源 CSV；划分后校验并集/交集/比例/图像目录
  - 默认 dry-run 可先看报告；正式写入需 --write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSPITALS = ("huaxi", "liaoning", "xiangya")
REQUIRED_COLS = (
    "oct_id",
    "image_folder",
    "age",
    "hpv_status",
    "tct_result",
    "pathology_class",
)
# 固定划分种子（与训练 SEED 无关；写入 README，保证可复现）
DEFAULT_SPLIT_SEED = 20260720
DEFAULT_VAL_RATIO = 0.2


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_center(image_folder: str) -> str:
    """从 image_folder 解析中心名，例如 .../development_octData/LiaoNing/<oct_id>。"""
    parts = str(image_folder).replace("\\", "/").split("/")
    for key in ("development_octData", "external_octData"):
        if key in parts:
            i = parts.index(key)
            if i + 1 < len(parts):
                return parts[i + 1]
    return "UNKNOWN"


def class_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    n_pos = int((df["pathology_class"] == 1).sum())
    n_neg = int((df["pathology_class"] == 0).sum())
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": (n_pos / n) if n else float("nan"),
        "neg_rate": (n_neg / n) if n else float("nan"),
        "pos_neg": f"{n_pos}:{n_neg}",
    }


def stratified_indices_by_group(
    groups: list[str],
    labels: list[int],
    val_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在每个 (group, label) 桶内按 val_ratio 抽样到 val，其余进 train。
    桶太小（<2）时整桶进 train，避免空桶或无法分层。
    """
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (g, y) in enumerate(zip(groups, labels)):
        buckets[(g, int(y))].append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for key in sorted(buckets.keys(), key=lambda x: (x[0], x[1])):
        idxs = np.array(buckets[key], dtype=np.int64)
        rng.shuffle(idxs)
        n = len(idxs)
        if n < 2:
            train_idx.extend(idxs.tolist())
            continue
        n_val = int(round(n * val_ratio))
        # 保证两侧都非空
        n_val = max(1, min(n - 1, n_val))
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())

    train_arr = np.array(sorted(train_idx), dtype=np.int64)
    val_arr = np.array(sorted(val_idx), dtype=np.int64)
    return train_arr, val_arr


def load_and_check_development(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少源文件: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺列: {missing}")
    if df["oct_id"].duplicated().any():
        dups = df.loc[df["oct_id"].duplicated(keep=False), "oct_id"].tolist()
        raise ValueError(f"{path} 存在重复 oct_id: {dups[:10]}...")
    if not set(df["pathology_class"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} pathology_class 含非 0/1 值")
    df = df.copy()
    df["center"] = df["image_folder"].map(extract_center)
    if (df["center"] == "UNKNOWN").any():
        bad = df.loc[df["center"] == "UNKNOWN", "image_folder"].head(3).tolist()
        raise ValueError(f"{path} 无法解析中心名，示例: {bad}")
    return df


def verify_split(
    hospital: str,
    dev: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    external_df: pd.DataFrame | None,
    data_root: str,
    pos_rate_tol: float = 0.05,
) -> list[str]:
    """返回警告列表；遇硬错误直接 raise。"""
    warnings: list[str] = []

    # 列一致
    for name, part in ("train", train_df), ("val", val_df):
        miss = [c for c in REQUIRED_COLS if c not in part.columns]
        if miss:
            raise ValueError(f"[{hospital}] {name} 缺列 {miss}")

    train_ids = set(train_df["oct_id"].astype(str))
    val_ids = set(val_df["oct_id"].astype(str))
    dev_ids = set(dev["oct_id"].astype(str))

    if train_ids & val_ids:
        raise ValueError(
            f"[{hospital}] train∩val 非空: {sorted(train_ids & val_ids)[:5]}"
        )
    if train_ids | val_ids != dev_ids:
        only_dev = sorted(dev_ids - (train_ids | val_ids))
        only_parts = sorted((train_ids | val_ids) - dev_ids)
        raise ValueError(
            f"[{hospital}] train∪val ≠ development；"
            f"仅在dev={only_dev[:5]} 仅在划分={only_parts[:5]}"
        )
    if len(train_df) + len(val_df) != len(dev):
        raise ValueError(
            f"[{hospital}] 行数之和 {len(train_df)+len(val_df)} != development {len(dev)}"
        )

    # 与 external 互斥
    if external_df is not None and len(external_df):
        ext_ids = set(external_df["oct_id"].astype(str))
        leak = (train_ids | val_ids) & ext_ids
        if leak:
            raise ValueError(f"[{hospital}] development 划分与 external 重叠: {sorted(leak)[:5]}")

    # 内容与 development 一致（按 oct_id 对齐后比较关键列；NaN 视为相等）
    key = "oct_id"
    for part_name, part in ("train", train_df), ("val", val_df):
        left = part[list(REQUIRED_COLS)].copy()
        right = dev[list(REQUIRED_COLS)].copy()
        if set(left[key].astype(str)) - set(right[key].astype(str)):
            raise ValueError(f"[{hospital}] {part_name} 含 development 中不存在的 oct_id")
        merged = left.merge(right, on=key, suffixes=("_new", "_dev"), how="left")
        for col in REQUIRED_COLS:
            if col == key:
                continue
            a = merged[f"{col}_new"]
            b = merged[f"{col}_dev"]
            # 两边同为 NA 视为一致；其余要求相等
            both_na = a.isna() & b.isna()
            equal = (a == b) | both_na
            if not bool(equal.all()):
                bad = merged.loc[~equal, [key, f"{col}_new", f"{col}_dev"]].head(5)
                raise ValueError(
                    f"[{hospital}] {part_name}.{col} 与 development 不一致:\n{bad}"
                )

    # 阴阳比例
    s_dev = class_stats(dev)
    s_tr = class_stats(train_df)
    s_va = class_stats(val_df)
    for name, s in ("train", s_tr), ("val", s_va):
        if abs(s["pos_rate"] - s_dev["pos_rate"]) > pos_rate_tol:
            warnings.append(
                f"[{hospital}] {name} 阳性率 {s['pos_rate']:.3f} 与 development "
                f"{s_dev['pos_rate']:.3f} 差 > {pos_rate_tol}"
            )

    # 图像目录存在
    for name, part in ("train", train_df), ("val", val_df):
        missing_dirs = []
        for folder in part["image_folder"]:
            abs_dir = os.path.join(data_root, folder)
            if not os.path.isdir(abs_dir):
                missing_dirs.append(folder)
        if missing_dirs:
            raise ValueError(
                f"[{hospital}] {name} 有 {len(missing_dirs)} 个 image_folder 不存在，"
                f"示例: {missing_dirs[:3]}"
            )

    return warnings


def split_one_hospital(
    hospital: str,
    data_root: str,
    split_seed: int,
    val_ratio: float,
) -> dict:
    dev_path = os.path.join(data_root, f"development_{hospital}.csv")
    ext_path = os.path.join(data_root, f"external_{hospital}.csv")
    dev = load_and_check_development(dev_path)
    external_df = pd.read_csv(ext_path) if os.path.isfile(ext_path) else None

    rng = np.random.default_rng(split_seed + sum(ord(c) for c in hospital))
    groups = dev["center"].astype(str).tolist()
    labels = dev["pathology_class"].astype(int).tolist()
    train_i, val_i = stratified_indices_by_group(groups, labels, val_ratio, rng)

    train_df = dev.iloc[train_i].drop(columns=["center"]).reset_index(drop=True)
    val_df = dev.iloc[val_i].drop(columns=["center"]).reset_index(drop=True)

    warnings = verify_split(
        hospital, dev.drop(columns=["center"]), train_df, val_df, external_df, data_root
    )

    # 按中心报告
    center_report = {}
    dev_c = load_and_check_development(dev_path)
    for center in sorted(dev_c["center"].unique()):
        d_sub = dev_c[dev_c["center"] == center]
        # train/val 用 image_folder 反查中心
        tr_sub = train_df[train_df["image_folder"].map(extract_center) == center]
        va_sub = val_df[val_df["image_folder"].map(extract_center) == center]
        center_report[center] = {
            "development": class_stats(d_sub),
            "train": class_stats(tr_sub),
            "val": class_stats(va_sub),
        }

    report = {
        "hospital_fold": hospital,
        "source_development": os.path.abspath(dev_path),
        "source_development_sha256": _file_sha256(dev_path),
        "source_external": os.path.abspath(ext_path) if external_df is not None else None,
        "source_external_sha256": _file_sha256(ext_path) if external_df is not None else None,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "development": class_stats(dev.drop(columns=["center"])),
        "train": class_stats(train_df),
        "val": class_stats(val_df),
        "external": class_stats(external_df) if external_df is not None else None,
        "by_center": center_report,
        "warnings": warnings,
    }
    return {
        "train_df": train_df,
        "val_df": val_df,
        "external_df": external_df,
        "dev_df": dev.drop(columns=["center"]),
        "report": report,
    }


def write_readme(snapshot_dir: str, reports: list[dict], split_seed: int, val_ratio: float) -> None:
    lines = [
        "# Paper v4 LOHO 划分快照（train / val / external）",
        "",
        f"> **生成日期**：{date.today().isoformat()}",
        f"> **划分种子 split_seed**：`{split_seed}`",
        f"> **验证集比例 val_ratio**：`{val_ratio}`（约 8:2）",
        "> **源数据**：各折 `development_*.csv` / `external_*.csv`（与运行时 `dataset/` 一致）",
        "",
        "## 协议",
        "",
        "1. 仍 **3 折**，折名 = **外部测试医院**（huaxi / liaoning / xiangya）。",
        "2. `development` = 其余两家医院（与 v3 相同，本快照中保留副本）。",
        "3. 在 development 内按 **中心 × pathology_class** 分层抽样：",
        "   - 约 80% → `train_{hospital}.csv`",
        "   - 约 20% → `val_{hospital}.csv`",
        "   - 目标：train / val 的阴阳比例都接近 development 整体比例，并尽量保留各中心自身比例。",
        "4. `external_{hospital}.csv` **不参与训练与早停**，仅最终评估。",
        "",
        "## 与 v3 的关系",
        "",
        "- **不覆盖** `data/snapshots/paper_v3_tsy_loho/`。",
        "- v3 曾用 external 做验证选模；v4 改为内部 val 选模，external 只测一次。",
        "",
        "## 各折统计",
        "",
    ]
    for r in reports:
        h = r["hospital_fold"]
        lines.append(f"### 折 `{h}`（测试 = external_{h}）")
        lines.append("")
        lines.append(
            f"| 划分 | n | 阳:阴 | 阳性率 |\n|------|--:|------|--------|"
        )
        for split_key, label in (
            ("development", "development（源）"),
            ("train", "train"),
            ("val", "val"),
            ("external", "external（测试）"),
        ):
            s = r.get(split_key)
            if not s:
                continue
            lines.append(
                f"| {label} | {s['n']} | {s['pos_neg']} | {s['pos_rate']:.3f} |"
            )
        lines.append("")
        lines.append("按中心：")
        lines.append("")
        for center, cr in r.get("by_center", {}).items():
            d, t, v = cr["development"], cr["train"], cr["val"]
            lines.append(
                f"- **{center}**：dev {d['pos_neg']}（阳率 {d['pos_rate']:.3f}）→ "
                f"train {t['pos_neg']}（{t['pos_rate']:.3f}），"
                f"val {v['pos_neg']}（{v['pos_rate']:.3f}）"
            )
        lines.append("")
        if r.get("warnings"):
            lines.append("警告：")
            for w in r["warnings"]:
                lines.append(f"- {w}")
            lines.append("")
        lines.append(f"- development SHA256: `{r['source_development_sha256']}`")
        if r.get("source_external_sha256"):
            lines.append(f"- external SHA256: `{r['source_external_sha256']}`")
        lines.append("")

    lines.extend(
        [
            "## 复现命令",
            "",
            "```bash",
            "export PATH=\"/home/amax/anaconda3/bin:$PATH\"",
            "cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet",
            f"python data/prepare_paper_v4_splits.py --write --split-seed {split_seed} --val-ratio {val_ratio}",
            "```",
            "",
            "## 训练时 CSV",
            "",
            "脚本会把 `train_*.csv` / `val_*.csv` 复制到 `dataset/`（即 `tsy_loho/`），",
            "`main.py` 使用：",
            "",
            "- 训练：`train_{hospital}.csv`",
            "- 验证/早停：`val_{hospital}.csv`",
            "- 终评：`external_{hospital}.csv`",
            "",
            f"*文档生成日期：{date.today().isoformat()}*",
            "",
        ]
    )
    with open(os.path.join(snapshot_dir, "README_2026-07-20.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="生成 paper v4 train/val 划分")
    parser.add_argument(
        "--data-root",
        default=os.path.join(PROJECT_ROOT, "dataset"),
        help="含 development_*.csv / external_*.csv 的目录",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=os.path.join(PROJECT_ROOT, "data", "snapshots", "paper_v4_tsy_loho"),
        help="快照输出目录（勿指向 paper_v3）",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument(
        "--write",
        action="store_true",
        help="真正写入文件；不加此开关只做划分与校验（不落盘）",
    )
    parser.add_argument(
        "--install-to-data-root",
        action="store_true",
        default=True,
        help="写入快照后，同步 train/val 到 data-root（默认开）",
    )
    parser.add_argument(
        "--no-install-to-data-root",
        action="store_true",
        help="只写快照，不复制到 dataset/",
    )
    args = parser.parse_args()

    if "paper_v3" in os.path.abspath(args.snapshot_dir):
        raise SystemExit("拒绝写入：snapshot-dir 不可指向 paper_v3")

    install = args.install_to_data_root and not args.no_install_to_data_root
    results = []
    for h in HOSPITALS:
        print(f"\n==== 处理折 {h} ====")
        out = split_one_hospital(h, args.data_root, args.split_seed, args.val_ratio)
        r = out["report"]
        print(
            f"  development {r['development']['n']} "
            f"阳:阴={r['development']['pos_neg']} ({r['development']['pos_rate']:.3f})"
        )
        print(
            f"  train       {r['train']['n']} "
            f"阳:阴={r['train']['pos_neg']} ({r['train']['pos_rate']:.3f})"
        )
        print(
            f"  val         {r['val']['n']} "
            f"阳:阴={r['val']['pos_neg']} ({r['val']['pos_rate']:.3f})"
        )
        for w in r["warnings"]:
            print(f"  ⚠️ {w}")
        results.append(out)

    if not args.write:
        print("\n【dry-run】校验通过，未写入。加上 --write 才会落盘。")
        return

    os.makedirs(args.snapshot_dir, exist_ok=True)
    reports = []
    for out in results:
        h = out["report"]["hospital_fold"]
        # 保留 development / external 副本 + 新 train / val
        out["dev_df"].to_csv(
            os.path.join(args.snapshot_dir, f"development_{h}.csv"), index=False
        )
        out["train_df"].to_csv(
            os.path.join(args.snapshot_dir, f"train_{h}.csv"), index=False
        )
        out["val_df"].to_csv(
            os.path.join(args.snapshot_dir, f"val_{h}.csv"), index=False
        )
        if out["external_df"] is not None:
            out["external_df"].to_csv(
                os.path.join(args.snapshot_dir, f"external_{h}.csv"), index=False
            )
        if install:
            out["train_df"].to_csv(
                os.path.join(args.data_root, f"train_{h}.csv"), index=False
            )
            out["val_df"].to_csv(
                os.path.join(args.data_root, f"val_{h}.csv"), index=False
            )
        reports.append(out["report"])
        print(f"  ✅ 已写入折 {h}")

    write_readme(args.snapshot_dir, reports, args.split_seed, args.val_ratio)
    report_json = os.path.join(args.snapshot_dir, "split_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    # 二次从磁盘读回再校验，防止写坏
    print("\n==== 落盘后二次校验 ====")
    for h in HOSPITALS:
        dev = pd.read_csv(os.path.join(args.snapshot_dir, f"development_{h}.csv"))
        train_df = pd.read_csv(os.path.join(args.snapshot_dir, f"train_{h}.csv"))
        val_df = pd.read_csv(os.path.join(args.snapshot_dir, f"val_{h}.csv"))
        ext_path = os.path.join(args.snapshot_dir, f"external_{h}.csv")
        external_df = pd.read_csv(ext_path) if os.path.isfile(ext_path) else None
        warns = verify_split(h, dev, train_df, val_df, external_df, args.data_root)
        # 与 data-root 安装副本一致
        if install:
            t2 = pd.read_csv(os.path.join(args.data_root, f"train_{h}.csv"))
            v2 = pd.read_csv(os.path.join(args.data_root, f"val_{h}.csv"))
            if not train_df.equals(t2) or not val_df.equals(v2):
                raise RuntimeError(f"[{h}] 快照与 data-root 安装副本不一致")
        print(f"  ✅ {h} 二次校验通过" + (f"（警告 {len(warns)}）" if warns else ""))

    print(f"\n快照目录: {args.snapshot_dir}")
    print(f"README: {os.path.join(args.snapshot_dir, 'README_2026-07-20.md')}")
    if install:
        print(f"已同步 train_*.csv / val_*.csv → {args.data_root}")


if __name__ == "__main__":
    main()
