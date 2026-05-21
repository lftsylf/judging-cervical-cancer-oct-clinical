# ResNet18 T1 Baseline 进度

- **记录日期**：2026-05-16
- **状态**：已完成（9 / 9）
- **协议**：LOHO 三折 × 3 seeds（42, 123, 2024）；`run_baseline_t1.sh` + `OPTIGENESIS_BACKBONE=resnet18`
- **输出根**：`outputs_baseline_t1_v2_resnet18/`

## 3-seed 外部 ROC-AUC 均值

| 折 | 均值 |
|---|---:|
| 华西 | 0.5561 |
| 辽宁 | 0.5475 |
| 湘雅 | 0.5836 |
| 三折平均 | 0.5624 |

对比参考：`logs/experiments/2026-05-16_resnet50_t1_vs_v1_swin_baseline.md`（ResNet50 T1 三折平均 0.5983；v1 Swin 0.5922）
