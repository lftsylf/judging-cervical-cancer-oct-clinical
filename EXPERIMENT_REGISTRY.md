# 实验总表（Registry）

> 用法：每完成一次实验，在本表新增一行，并链接到对应记录文件与 commit。

## 字段说明

- `ID`：自定义编号（例如 `EXP-001`）
- `Date`：实验日期
- `Type`：`baseline / 消融 / 对比 / 调参 / 复现`
- `Tag`：实验短名
- `Branch`：实验所在分支
- `Commit`：关键提交哈希（短哈希即可）
- `Script`：运行脚本
- `Output`：输出目录
- `Summary`：一句话结论
- `Record`：对应详细记录 markdown 路径

## 实验清单

| ID | Date | Type | Tag | Branch | Commit | Script | Output | Summary | Record |
|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-03-27 | baseline | baseline_outputs | main | e49bc7c | run_loho_3centers.sh | baseline_outputs | 第一版 baseline，XiangYa 最优，LiaoNing 存在阈值失配 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-002 | 2026-03-27 | 消融 | ema_huaxi_15 | main | e49bc7c | (单折运行) | ema_huaxi_15_outputs | 相对 baseline_huaxi_15 有提升，但 PR-AUC 略低 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-003 | 2026-03-27 | 消融 | aux_huaxi_15 | main | e49bc7c | (单折运行) | aux_huaxi_outputs | 未观察到提升 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-004 | 2026-03-28 | baseline | baseline_recheck_50 | main | e49bc7c | run_loho_3centers.sh | baseline_recheck_50_outputs | 新选模口径下辽宁改善，华西/湘雅下降 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-005 | 2026-03-28 | 消融 | aux_50 | main | e49bc7c | run_loho_aux_50.sh | aux_50_outputs | 三折完整，整体偏反面 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-006 | 2026-03-28 | 消融 | ema_50 | main | e49bc7c | (全三折 EMA 脚本) | ema_outputs | 三折完整，整体偏反面 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-007 | 2026-03-28 | 消融 | uda_50 | main | e49bc7c | run_loho_uda_50.sh | uda_50_outputs | 三折完整，整体偏反面 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-008 | 2026-03-28 | 对比 | vit_small_50 | main | e49bc7c | run_loho_vit_small.sh | vit_loho_outputs/vit_small_patch16_224 | 宏平均优于 baseline | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-009 | 2026-03-28 | 对比 | vit_base_50 | main | e49bc7c | run_loho_vit_base.sh | vit_loho_outputs/vit_base_patch16_224 | 高敏感低特异，泛化不稳 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-010 | 2026-04-08 | 复现 | baseline_repro_newckpt | main | 00250fe | run_loho_baseline_repro_newckpt.sh | baseline_repro_newckpt_outputs | 锁定配置+防环境残留+dry-run 模式 | logs/experiments/2026-04-08_migrated_from_obsidian.md |
| EXP-011 | 2026-04-09 | baseline | baseline_outputs_mix_v1 | main | - | (目录组合) | baseline_outputs | 组合基线：辽宁/华西取 baseline_recheck_50，湘雅取 baseline_repro_newckpt；三折均值 ROC-AUC=0.6525 | logs/experiments/2026-04-08_migrated_from_obsidian.md |

## 使用建议

- 每次实验至少保留 1 个“代码 commit + 记录文件”配对。
- 对论文会引用的实验，补齐三折明细与 Delta 指标。
- 如果同一实验多次重跑，建议追加 `-r1/-r2` 后缀区分。

