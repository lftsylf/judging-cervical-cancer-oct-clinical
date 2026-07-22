# 自动探路报告 · ce@0.2 可删

- candidate: `/ssd_data/tsy_study_venv/OptiGenesis_Lancet/outputs/paper_v4/baseline/ours_uw_frameaux_t2_ce_w0.2`
- 对照 edl@0.2: `/ssd_data/tsy_study_venv/OptiGenesis_Lancet/outputs/paper_v4/baseline/ours_uw_frameaux_t2`
- 判定阈值: ref − cand ≥ **0.02** → 启动 edl@0.3

## 外部 ROC（seed=42）

| 中心 | 对照 edl@0.2 | candidate | Δ(cand−ref) |
|------|-------------:|----------:|------------:|
| huaxi | 0.5562 | 0.5415 | -0.0148 |
| liaoning | 0.5344 | 0.5440 | +0.0096 |
| xiangya | 0.5756 | 0.4134 | -0.1622 |

- **对照三折均值**: 0.5554
- **candidate 三折均值**: 0.4996
- **gap (ref−cand)**: +0.0558
- **candidate 外部 n_eff 均值**: 11.988（≈12 表示加权仍近等权）

## 判决

**启动 edl@0.3**：candidate 相对 edl@0.2 仍差 ≥ 0.02。
