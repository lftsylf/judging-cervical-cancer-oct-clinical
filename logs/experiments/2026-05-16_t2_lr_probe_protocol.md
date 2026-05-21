# T2 学习率四组探路协议

- **日期**：2026-05-16
- **脚本**：`run_baseline_t2_lr_probe.sh`
- **规格**：**3 折 LOHO**（华西 / 辽宁 / 湘雅）× **1 seed（42）** × **4 组 LR** = **12 次训练**
- **早停**：验证集 ROC-AUC，`patience=10`（`main.py`，未改）
- **默认**：`resnet50` · `POS_WEIGHT=1.3`
- **输出根**：`outputs_baseline_t2_lr_resnet50/<g1..g4>/<hospital>/seed_42/`

## 四组互斥配置

| 组 | 目录名 | 含义 |
|---|---|---|
| 1 | `g1_uniform_5e5` | 全网 `LR=5e-5` |
| 2 | `g2_disc_bb1e5_head1e4` | backbone `1e-5`，head `1e-4` |
| 3 | `g3_disc_bb5e6_head1e4` | backbone `5e-6`，head `1e-4` |
| 4 | `g4_freeze5_disc_bb1e5_head1e4` | 前 5 epoch 冻 backbone；解冻后 bb `1e-5` + head `1e-4` |

## 如何选 winner

每组对 **3 折外部 ROC-AUC 取均值**（或看最差的折是否拖后腿），再与 `g1` 基线比；优先外部高、开发−外部差距小。

## 启动

```bash
cd /ssd_data/tsy_study_venv/OptiGenesis_Lancet
./run_baseline_t2_lr_probe.sh

RUN_ONLY=3 ./run_baseline_t2_lr_probe.sh
HOSPITALS=(huaxi liaoning) ./run_baseline_t2_lr_probe.sh

# 续跑：已写入「训练完成！」的折自动跳过（如已完成的华西 g1）
./run_baseline_t2_lr_probe.sh
FORCE_RERUN=1 ./run_baseline_t2_lr_probe.sh   # 强制全部重跑

./run_experiment.sh --detach /path/to/tsy_loho ./run_baseline_t2_lr_probe.sh
```
