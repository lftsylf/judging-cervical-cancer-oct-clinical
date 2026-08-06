#!/usr/bin/env python3
"""
湘雅内部 site-bag 切分（协议 xyi）：

  - 内部：XiangYa_dataset.csv（有 OCT 的 97 例）8:2 → train_xyi / val_xyi
  - 外部终评：HuaXi_dataset / LiaoNing_dataset（仅保留磁盘有 octData 的例）
  - 解析病理阳点；阳患者兄妹 ID±1 共享阳点
  - 写出 pos_sites 列供 dataloader 组袋

用法:
  python data/prepare_xyi_sitebag_splits.py --write --split-seed 20260731 --val-ratio 0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sitebag_utils import (
    canon_oct_id,
    encode_age,
    encode_hpv,
    encode_tct,
    parse_huaxi_positive_sites,
    parse_liaoning_positive_sites,
    parse_xiangya_positive_sites,
    share_sibling_pos_sites,
    sites_to_str,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET = ROOT / "dataset"
SNAP = ROOT / "data" / "snapshots" / "paper_v4_xyi_sitebag"


def _load_xiangya_rows() -> pd.DataFrame:
    xy = pd.read_csv(DATASET / "XiangYa" / "XiangYa_dataset.csv")
    raw = pd.read_csv(DATA_DIR / "副本湘雅二肖潇.csv")
    xy = xy.copy()
    raw = raw.copy()
    xy["cid"] = xy["OCT_ID"].map(canon_oct_id)
    raw["cid"] = raw["光超编号"].map(canon_oct_id)
    raw = raw.drop_duplicates("cid", keep="first")
    m = xy.merge(raw, on="cid", how="left")

    pos_list = []
    notes = []
    for _, r in m.iterrows():
        path = r.get("病理结果")
        sites, note = parse_xiangya_positive_sites(path) if pd.notna(path) else ([], "empty")
        pos_list.append(sites)
        notes.append(note)

    labels = m["Final_Label"].astype(int).tolist()
    oct_ids = m["OCT_ID"].astype(str).tolist()
    pos_shared = share_sibling_pos_sites(oct_ids, labels, pos_list)

    rows = []
    for idx, (_, r) in enumerate(m.iterrows()):
        oid = str(r["OCT_ID"])
        folder = f"XiangYa/octData/{oid}"
        abs_folder = DATASET / folder
        if not abs_folder.is_dir():
            raise FileNotFoundError(f"缺少 OCT 目录: {abs_folder}")
        rows.append(
            {
                "oct_id": oid,
                "image_folder": folder,
                "age": encode_age(r.get("Age")),
                "hpv_status": encode_hpv(r.get("HPV_Result")),
                "tct_result": encode_tct(r.get("TCT_Result")),
                "pathology_class": int(r["Final_Label"]),
                "pos_sites": sites_to_str(pos_shared[idx]),
                "pos_sites_raw": sites_to_str(pos_list[idx]),
                "parse_note": notes[idx],
                "sibling_shared": int(bool(pos_shared[idx]) and not pos_list[idx]),
                "center": "xiangya",
            }
        )
    return pd.DataFrame(rows)


def _load_huaxi_rows() -> pd.DataFrame:
    hx = pd.read_csv(DATASET / "HuaXi" / "HuaXi_dataset.csv")
    raw = pd.read_csv(DATA_DIR / "华西二院.csv")
    hx = hx.copy()
    raw = raw.copy()
    hx["cid"] = hx["OCT_ID"].map(canon_oct_id)
    raw["cid"] = raw["光超编号"].map(lambda x: canon_oct_id(str(x).split("/")[0]))
    raw = raw.drop_duplicates("cid", keep="first")
    m = hx.merge(raw, on="cid", how="left")

    rows = []
    for _, r in m.iterrows():
        oid = str(r["OCT_ID"])
        folder = f"HuaXi/octData/{oid}"
        if not (DATASET / folder).is_dir():
            continue
        path = r.get("病理")
        biopsy = r.get("活检点") if pd.notna(r.get("活检点")) else ""
        sites, note = parse_huaxi_positive_sites(
            str(path) if pd.notna(path) else "",
            str(biopsy) if biopsy is not None else "",
        )
        # 外部终评不需要组袋阳点，但保留列以统一 schema
        rows.append(
            {
                "oct_id": oid,
                "image_folder": folder,
                "age": encode_age(r.get("Age") if "Age" in r else r.get("年龄")),
                "hpv_status": encode_hpv(r.get("HPV_Result") if "HPV_Result" in r else r.get("HPV")),
                "tct_result": encode_tct(r.get("TCT_Result") if "TCT_Result" in r else r.get("TCT")),
                "pathology_class": int(r["Final_Label"]),
                "pos_sites": sites_to_str(sites),
                "parse_note": note,
                "center": "huaxi",
            }
        )
    return pd.DataFrame(rows)


def _load_liaoning_rows() -> pd.DataFrame:
    ln = pd.read_csv(DATASET / "LiaoNing" / "LiaoNing_dataset.csv")
    raw = pd.read_csv(DATA_DIR / "辽宁省肿瘤.csv")
    ln = ln.copy()
    raw = raw.copy()
    ln["cid"] = ln["OCT_ID"].map(canon_oct_id)
    raw["cid"] = raw["OCT编号"].map(canon_oct_id)
    raw = raw.drop_duplicates("cid", keep="first")
    m = ln.merge(raw, on="cid", how="left")

    rows = []
    for _, r in m.iterrows():
        oid = str(r["OCT_ID"])
        folder = f"LiaoNing/octData/{oid}"
        if not (DATASET / folder).is_dir():
            continue
        path = r.get("病理")
        ihc = r.get("免疫组化")
        s1, n1 = parse_liaoning_positive_sites(path) if pd.notna(path) else ([], "empty")
        s2, n2 = parse_liaoning_positive_sites(ihc) if pd.notna(ihc) else ([], "empty")
        sites = sorted(set(s1) | set(s2))
        note = n1 if sites or n1 != "empty" else n2
        rows.append(
            {
                "oct_id": oid,
                "image_folder": folder,
                "age": encode_age(r.get("Age") if "Age" in r else r.get("年龄")),
                "hpv_status": encode_hpv(r.get("HPV_Result") if "HPV_Result" in r else r.get("HPV")),
                "tct_result": encode_tct(r.get("TCT_Result") if "TCT_Result" in r else r.get("TCT")),
                "pathology_class": int(r["Final_Label"]),
                "pos_sites": sites_to_str(sites),
                "parse_note": note,
                "center": "liaoning",
            }
        )
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, val_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for label in sorted(df["pathology_class"].unique()):
        idxs = df.index[df["pathology_class"] == label].to_numpy()
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) > 1 else 0
        # keep at least 1 train if possible
        if len(idxs) - n_val < 1 and len(idxs) > 1:
            n_val = len(idxs) - 1
        val_idx.extend(idxs[:n_val].tolist())
        train_idx.extend(idxs[n_val:].tolist())
    return df.loc[sorted(train_idx)].reset_index(drop=True), df.loc[sorted(val_idx)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--split-seed", type=int, default=20260731)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--install-to-data-root", action="store_true", default=True)
    args = ap.parse_args()

    xy = _load_xiangya_rows()
    hx = _load_huaxi_rows()
    ln = _load_liaoning_rows()

    train_df, val_df = stratified_split(xy, args.val_ratio, args.split_seed)
    external_all = pd.concat([hx, ln], ignore_index=True)

    def _stats(name, d):
        n = len(d)
        pos = int((d["pathology_class"] == 1).sum())
        with_sites = int(((d["pathology_class"] == 1) & (d["pos_sites"].astype(str).str.len() > 0)).sum())
        print(f"{name}: n={n} pos={pos} ({100*pos/max(n,1):.1f}%) pos_with_sites={with_sites}")

    print("=== xyi site-bag splits ===")
    _stats("xiangya_all", xy)
    _stats("train_xyi", train_df)
    _stats("val_xyi", val_df)
    _stats("external_huaxi", hx)
    _stats("external_liaoning", ln)
    _stats("external_xyi(pooled)", external_all)
    n_share = int(xy.get("sibling_shared", pd.Series(dtype=int)).sum()) if "sibling_shared" in xy else 0
    print(f"sibling_shared pos_sites rows: {n_share}")

    core_cols = [
        "oct_id",
        "image_folder",
        "age",
        "hpv_status",
        "tct_result",
        "pathology_class",
        "pos_sites",
        "center",
    ]

    if not args.write:
        print("(dry-run，加 --write 落盘)")
        return

    SNAP.mkdir(parents=True, exist_ok=True)
    files = {
        "train_xyi.csv": train_df[core_cols],
        "val_xyi.csv": val_df[core_cols],
        "external_xyi.csv": external_all[core_cols],
        "external_xyi_huaxi.csv": hx[core_cols],
        "external_xyi_liaoning.csv": ln[core_cols],
        "annotations_xiangya_pos_sites.csv": xy,
    }
    for name, d in files.items():
        path = SNAP / name
        d.to_csv(path, index=False)
        print("wrote", path)
        if args.install_to_data_root and name.startswith(("train_", "val_", "external_")):
            dst = DATASET / name
            d.to_csv(dst, index=False)
            print("installed", dst)


if __name__ == "__main__":
    main()
