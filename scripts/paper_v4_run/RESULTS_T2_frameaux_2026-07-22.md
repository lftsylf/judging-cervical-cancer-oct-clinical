# Paper v4 · T2 探路结果摘要（2026-07-22）

> 对应代码：`65d45c1`（帧级弱监督）+ `run_ours_uw_frameaux_t2_oct_only.sh`  
> 产物目录（本地，不入库）：`outputs/paper_v4/baseline/ours_uw_frameaux_t2/`

## 实验设置

- Full T2：`FRAME_AGG=uncertainty_weighted` + `ENABLE_FRAME_AUX=1`（type=**edl**，weight=**0.2**）
- 3 折 × seed **42**；对照同 seed 的 B1（`b1_mean_edl_t0`）与 Ours 无帧损（`ours_uw_agg_t0`）

## 外部 ROC（seed=42）

| 中心 | B1 | Ours（无帧损） | Full（+帧弱监督） |
|------|-----:|---------------:|------------------:|
| huaxi | 0.578 | 0.552 | 0.556 |
| liaoning | 0.640 | 0.551 | 0.534 |
| xiangya | 0.380 | 0.400 | 0.576 |
| **三折均值** | 0.533 | 0.501 | **0.555** |

## 帧不确定 / 加权诊断（external）

- 帧 \(u\) 几乎全挤在约 **0.20–0.33**；患者内标准差 ≈ 0.006–0.012
- \(n_{\mathrm{eff}}\approx 12\)（等权），`max_w≈1/12`
- **结论**：当前 edl、w=0.2 的帧弱监督**未拉开**帧 \(u\)，不确定加权仍近似等权；湘雅 ROC 上升不宜归因于「加权已生效」

## 后续可选（尚未跑）

1. `FRAME_AUX_WEIGHT=0.5`
2. `FRAME_AUX_TYPE=ce`
3. 先看直方图再决定是否上 T0 Full

## 相关已提交代码

| Commit | 内容 |
|--------|------|
| `312f05f` | Ours T0 脚本（无帧损） |
| `65d45c1` | 帧级弱监督实现 + T2 脚本 |
