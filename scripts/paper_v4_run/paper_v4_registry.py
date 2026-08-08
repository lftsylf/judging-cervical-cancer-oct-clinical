"""baseline2 / paper_v4 实验注册表（路径 + 角色）。

目录布局（与旧 LOHO v2 不同）::
    <run_dir>/seed_<seed>/logs/{val,external,external_huaxi,external_liaoning}_sample_predictions.csv

禁止把旧 ``outputs/paper_v4/baseline/`` Attn/LOHO 数字写入本表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

SEEDS: Tuple[int, ...] = (42, 123, 2024, 3407, 114514)

ABLATION_ROOT = "outputs/paper_v4/baseline2和消融"
CMP_ROOT = "outputs/paper_v4/对比实验"
OUT_TABLES = "outputs/paper_v4/tables"
OUT_FIGURES = "figures/paper_v4"


@dataclass(frozen=True)
class Experiment:
    key: str
    display: str
    rel_dir: str
    role: str  # final | baseline | ablation | comparison | variant | appendix
    paper_table: str  # main_cmp | main_ablation | appendix | skip
    note: str = ""


# 主文建议行顺序：对比表 / 消融表分开
EXPERIMENTS: Tuple[Experiment, ...] = (
    Experiment(
        key="ours",
        display="Ours (R50)",
        rel_dir=f"{ABLATION_ROOT}/4[final] xyi_sitebag_n2_uw_mil_aggmean_ema099",
        role="final",
        paper_table="main_cmp",
        note="sitebag n=2×全页 + UW + MIL@0.3 + 六窗 mean + EMA0.99",
    ),
    Experiment(
        key="baseline_page1",
        display="Baseline (12×page1)",
        rel_dir=f"{ABLATION_ROOT}/3[baseline] xyi_n12_page1_uw_mil_t0",
        role="baseline",
        paper_table="main_cmp",
        note="无 sitebag；12 点×首页；无 EMA",
    ),
    Experiment(
        key="cmp_abmil",
        display="ABMIL",
        rel_dir=f"{CMP_ROOT}/10[对比] abmil_sitebag_n2_aggmean_r50",
        role="comparison",
        paper_table="main_cmp",
    ),
    Experiment(
        key="cmp_dsmil",
        display="DSMIL",
        rel_dir=f"{CMP_ROOT}/12[对比] dsmil_sitebag_n2_aggmean_r50",
        role="comparison",
        paper_table="main_cmp",
    ),
    Experiment(
        key="cmp_ubix",
        display="UBIX",
        rel_dir=f"{CMP_ROOT}/11[对比] ubix_sitebag_n2_equal_mcdrop_r50",
        role="comparison",
        paper_table="main_cmp",
    ),
    Experiment(
        key="cmp_wma",
        display="WMA loss",
        rel_dir=f"{CMP_ROOT}/5[对比] xyi_sitebag_n2_uw_mil_aggmean_wmaC01",
        role="comparison",
        paper_table="main_cmp",
        note="主损失换 WMA(C=0.1)；无 EMA",
    ),
    Experiment(
        key="abl_max",
        display="w/o Mean (→Max)",
        rel_dir=f"{ABLATION_ROOT}/1[消融mean->max] xyi_sitebag_n2_uw_mil_aggmax_ema099",
        role="ablation",
        paper_table="main_ablation",
        note="相对 Ours 仅六窗 mean→max（含 EMA）",
    ),
    Experiment(
        key="abl_no_ema",
        display="w/o EMA",
        rel_dir=f"{ABLATION_ROOT}/2[消融无ema] xyi_sitebag_n2_uw_mil_aggmean_t0",
        role="ablation",
        paper_table="main_ablation",
    ),
    Experiment(
        key="abl_equal",
        display="w/o UWA (equal)",
        rel_dir=f"{ABLATION_ROOT}/10[消融关uw] xyi_sitebag_n2_equal_mil_aggmean_ema099",
        role="ablation",
        paper_table="main_ablation",
    ),
    Experiment(
        key="abl_noaux",
        display="w/o FrameAux",
        rel_dir=f"{ABLATION_ROOT}/11[消融 无aux] xyi_sitebag_n2_uw_noaux_aggmean_ema099",
        role="ablation",
        paper_table="main_ablation",
    ),
    Experiment(
        key="var_convnext",
        display="Ours-ConvNeXt-T",
        rel_dir=f"{CMP_ROOT}/13[变体] final_uw_mil_aggmean_ema099_convnext_tiny",
        role="variant",
        paper_table="appendix",
        note="骨干敏感性；Val↑ Ext↓，不宜作 Method-B",
    ),
)


def experiments_for_tables(
    include_appendix: bool = True,
) -> List[Experiment]:
    out: List[Experiment] = []
    for e in EXPERIMENTS:
        if e.paper_table == "skip":
            continue
        if e.paper_table == "appendix" and not include_appendix:
            continue
        out.append(e)
    return out


def find_experiment(key: str) -> Optional[Experiment]:
    for e in EXPERIMENTS:
        if e.key == key:
            return e
    return None


def champion() -> Experiment:
    return EXPERIMENTS[0]


PRED_SPLITS: Sequence[str] = (
    "val",
    "external",
    "external_huaxi",
    "external_liaoning",
)

SPLIT_LABEL = {
    "val": "Val (Xiangya)",
    "external": "Ext pooled",
    "external_huaxi": "Huaxi",
    "external_liaoning": "Liaoning",
}
