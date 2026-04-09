# 实验记录模板（单次实验）

> 用法：每次实验复制本模板一份，文件名建议：`logs/experiments/YYYY-MM-DD_<tag>.md`

## 1) 基本信息

- 实验名称：
- 日期：
- 负责人：
- 关联分支：
- 关联 commit：
- 实验类型：`baseline / 消融 / 对比 / 调参 / 复现`

## 2) 目标与假设

- 目标：
- 假设：
- 对照组：
- 主要观察指标：`AUC-ROC / AUC-PR / F1 / MCC / BACC / Sens / Spec`

## 3) 代码与配置变更

- 运行脚本：
- 关键环境变量：
  - `OPTIGENESIS_BACKBONE=`
  - `OPTIGENESIS_ENABLE_AUX=`
  - `OPTIGENESIS_ENABLE_EMA=`
  - `OPTIGENESIS_ENABLE_CORAL=`
  - `OPTIGENESIS_BATCH_SIZE=`
  - `OPTIGENESIS_OUTPUT_DIR=`
- 关键超参（实际生效）：
  - `LR=`
  - `EPOCHS=`
  - `POS_WEIGHT=`
  - `IMG_SIZE=`
- checkpoint 选模规则：

## 4) 数据与划分

- 数据版本/路径：
- 划分策略：`LOHO`
- 折顺序：
- 数据清洗/过滤说明：

## 5) 运行结果（按 fold 填写）

| Fold | Best Epoch | AUC-ROC | AUC-PR | F1(+) | MCC | BACC | Sens | Spec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| xiangya |  |  |  |  |  |  |  |  |
| huaxi |  |  |  |  |  |  |  |  |
| liaoning |  |  |  |  |  |  |  |  |
| Macro Avg | - |  |  |  |  |  |  |  |

## 6) 与基线对比（Delta）

- 对比基线版本：
- 指标差值（提升为正）：
  - `ΔAUC-ROC=`
  - `ΔF1(+)=`
  - `ΔMCC=`
- 是否达到预期：

## 7) 结论与下一步

- 结论（1-3 条）：
- 风险/异常：
- 下一步动作：

## 8) 附件与产物

- 输出目录：
- 关键日志文件：
- 关键图表：
- 备注：

