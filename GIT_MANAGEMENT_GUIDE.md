# Git 管理说明（OptiGenesis_Lancet）

本文档记录了当前仓库的 Git 配置状态与后续常用操作，方便你之后稳定管理代码版本。

## 1. 当前已完成的关键配置

- 远程仓库已切换为你自己的仓库：
  - `origin = git@github.com:lftsylf/judging-cervical-cancer-oct-clinical.git`
- 已与原 `Primary_care` 远程断开（不再关联学姐仓库）。
- 本地主分支已统一为 `main`，并跟踪远端 `origin/main`。
- 已成功推送过当前代码到远端 `main`。

## 2. 已设置的忽略规则（避免误传数据/产物）

仓库根目录新增了 `.gitignore`，已忽略以下内容：

- 数据软链接与本地数据目录：
  - `dataset`
  - `tsy_loho`
- 实验输出与大文件目录：
  - `all_outputs/`
  - `outputs/`
  - `baseline_repro_newckpt_outputs/`
  - `configs/outputs/`
- 缓存与 IDE 文件：
  - `__pycache__/`
  - `.vscode/`
  - `.specstory/`

## 3. 日常 Git 使用流程（推荐）

### 3.1 查看状态

```bash
git status
```

### 3.2 拉取远端最新（开始工作前）

```bash
git pull
```

### 3.3 提交本地修改

```bash
git add -A
git commit -m "你的提交说明"
git push
```

### 3.4 Agent / AI 协作：改完即提交（强制）

> **登记日期**：2026-07-20  
> **适用范围**：Cursor / 其他 Agent 在本仓库代改代码或调整文件时。

**规则（请 Agent 默认遵守）：**

1. **每一次**完成一轮有意义的代码修改、配置调整或文档更新后，**立刻在本机创建一次 git commit**，不要攒到多轮改动再一起提交。
2. **提交说明必须用中文**，写清「改了什么 + 为什么改」（1–3 句即可）。
3. **只 stage 本轮相关文件**；不要把无关删除、大结果目录、数据软链一并提交（`outputs/` 等仍按 `.gitignore` 不入库）。
4. **不要默认 `git push`**；推送仅在用户明确要求时执行。
5. 若本机 commit hook / trailer 报错，可用：`/usr/bin/git commit --no-verify`（仅限 hook 干扰，非跳过代码检查的借口）。

**推荐信息风格示例：**

```text
feat: 实现帧级 EDL 不确定加权聚合，替换等权均值池化作为 v4 主方法

docs: 在 Git 指南中约定 Agent 改完即用中文提交
```

## 4. 创建实验里程碑提交（建议）

每次完成一个实验阶段时建议提交一次，例如：

- `baseline_repro`
- `aux_ablation`
- `uda_experiment`
- `vit_comparison`

提交信息尽量包含：

- 做了什么改动
- 为什么改
- 对应输出目录/脚本名

## 5. 常见检查命令

### 查看当前分支和跟踪关系

```bash
git branch -vv
```

### 查看远程地址

```bash
git remote -v
```

### 查看提交历史

```bash
git log --oneline -n 20
```

## 6. SSH 相关（当前可用）

你已在 Linux 端配置好可用 SSH，并验证通过（`ssh -T git@github.com` 成功）。

如果后续 SSH 异常，可优先检查：

```bash
cat ~/.ssh/config
ssh -T git@github.com
```

## 7. 备注

- 当前默认开发分支：`main`
- 默认推送远程：`origin`
- 建议持续保持“代码入库、数据不入库”的策略，确保仓库轻量且可复现。

## 8. 多轮实验与 v1 / v2 并行（分支 + 标签）

当 **main 上已是旧实验代码**（例如 Swin 正式稿），而你要在同一仓库里做 **v2 大改**（如默认骨干改为 ResNet、删多模态等）时，建议：

1. **在当前 main 的提交上打标签**（锁住“旧论文可复现”的代码快照，不依赖分支是否被删）  
   - 例：`git tag -a paper-v1-swin-baseline -m "投稿前 Swin+临床+WMA/EMA/AUX 代码基线"`  
   - `git push origin paper-v1-swin-baseline`

2. **从该提交拉 v2 开发分支**（在分支上改默认骨干、跑新消融，避免把未验证完的改动直接堆在 main）  
   - 例：`git checkout -b feature/v2-resnet-baseline`  
   - v2 跑通、表定稿后再合并回 `main`，或保留分支名并在论文里写「v2 对应分支 `feature/v2-resnet-baseline` + commit xxx」。

3. **正文/笔记里固定“结果 ↔ 代码”**  
   - 每个正式表注明：`tag` 或 `分支名` + `git rev-parse --short HEAD` + 关键 `export`（如 `OPTIGENESIS_BACKBONE`）。  
   - 输出目录与旧实验分开（例如 `outputs_baseline_v2_resnet50/`，见 `run_baseline_t0.sh`）。

已 push 到远端的 history **不会**因你新建分支而丢失；标签指向具体 commit，最适合当“论文快照”。若你希望 **main 永远代表最新开发线**，也可在打完 v1 标签后把 v2 合并进 main，由标签承担“旧稿复现”职责。

### 8.1 本仓库已落盘的标签与分支（2026-05-21）

| 名称 | 类型 | 指向 | 说明（中文） |
|------|------|------|----------------|
| `paper-v1-swin-baseline` | annotated tag | `6c48a1d` | v1：Swin-Tiny + 多模态，投稿前代码快照 |
| `feature/v2-resnet-baseline` | 分支 | tip 见 `git rev-parse --short HEAD`；核心功能提交 `c9d30fe` | v2 开发线：ResNet50 + 临床消融 + OCT-only 定稿 |
| `paper-v2-resnet50-oct-baseline` | annotated tag | 与分支 tip 一致（`git rev-parse paper-v2-resnet50-oct-baseline`） | v2 定稿 baseline：ResNet50、OCT-only、LR=5e-5、POS_WEIGHT=1.25 |

**v2 定稿训练协议（环境变量）**

```bash
export OPTIGENESIS_BACKBONE=resnet50
export OPTIGENESIS_USE_CLINICAL=0
export OPTIGENESIS_LR=5e-5
# POS_WEIGHT 默认 1.25（lancet_config）；可不 export
export OPTIGENESIS_USE_WMA=0
export OPTIGENESIS_ENABLE_AUX=0
export OPTIGENESIS_ENABLE_EMA=0
export OPTIGENESIS_ENABLE_CORAL=0
```

**脚本与文档**

- T0 定稿：`./run_baseline_t0_oct_only.sh` → `outputs_baseline_t0_v2_resnet50_oct_only/`
- 迭代记录：`outputs/baseline v2/BASELINE_V2_迭代与定稿.md`（本地归档；`outputs/**/*.log` 不入库）

**复现命令**

```bash
git fetch origin
git checkout paper-v2-resnet50-oct-baseline   # 或 feature/v2-resnet-baseline
git rev-parse --short HEAD
./run_experiment.sh --detach /path/to/tsy_loho ./run_baseline_t0_oct_only.sh
```

**推送标签与分支（需写权限时在本机执行）**

```bash
git push origin paper-v1-swin-baseline
git push origin feature/v2-resnet-baseline
git push origin paper-v2-resnet50-oct-baseline
```

### 8.2 Paper v3 冻结与 v4 开发线（2026-07-18）

> **登记日期**：2026-07-18  
> **动作**：将「整稿实验（baseline + 消融 + 对比）」冻结为第三版，并开第四版开发分支。

| 名称 | 类型 | 说明（中文） |
|------|------|----------------|
| `paper-v1-swin-baseline` | tag | v1：Swin-Tiny + 多模态 |
| `paper-v2-resnet50-oct-baseline` | tag | v2：ResNet50 OCT-only **协议**锁定 |
| `paper-v3-muse-full` | tag | **v3 整稿冻结**（含消融+对比代码与文档登记日） |
| `feature/v2-resnet-baseline` | 分支 | v2/v3 历史开发线（只读回顾） |
| `feature/paper-v4` | 分支 | **第四版开发中**（帧级不确定聚合 + 评估协议改造） |

**本地归档（大文件不入库，仅本机）**

| 路径 | 说明 |
|------|------|
| `outputs/第三版/` | 第三版实验结果只读归档（见其中 `README_冻结说明_2026-07-18.md`） |
| `data/snapshots/paper_v3_tsy_loho/` | 第三版 LOHO CSV 划分快照（见 `README_2026-07-18.md`） |
| `data/snapshots/paper_v4_tsy_loho/` | **第四版** train/val/external 划分快照（见 `README_2026-07-20.md`） |
| `outputs/paper_v4/` | 第四版新产物目录（见 `README_2026-07-18.md`） |

**为何仍快照 CSV（即使 v4 会改划分）**

审稿意见要求内部再划 val、外部只终评；划分会变。快照用于：**复现第三版表、审稿对照、防止原地改坏旧名单**。v4 新划分另存 `data/snapshots/paper_v4_*`，勿覆盖 v3 快照。

**v4 划分生成（2026-07-20）**

```bash
python data/prepare_paper_v4_splits.py --write --split-seed 20260720 --val-ratio 0.2
```

- 从 `development_*.csv` 按「中心 × 阴阳」分层约 8:2 → `train_*.csv` / `val_*.csv`
- `external_*.csv` 不改，仅终评；`main.py` 早停看内部 val

**常用命令**

```bash
# 回到第三版代码
git checkout paper-v3-muse-full

# 在第四版开发
git checkout feature/paper-v4

# 推送冻结（需远端写权限）
git push origin paper-v3-muse-full
git push -u origin feature/paper-v4
```

### 8.3 根目录脚本归档（2026-07-19）

> **登记日期**：2026-07-19  
> **动作**：清理仓库根目录大量 `run_*.sh`，迁入 `scripts/paper_v3_run/`。

| 路径 | 说明 |
|------|------|
| `run_experiment.sh` | **仍留在仓库根**（detach / 绑数据） |
| `scripts/paper_v3_run/` | 第三版及更早的 baseline / 消融 / 对比启动脚本（只读复现） |
| `README_运行脚本_2026-07-19.md` | 根目录入口说明 |
| `scripts/paper_v3_run/README_2026-07-19.md` | 归档目录说明与调用示例 |

第四版新脚本建议放在 `scripts/paper_v4_run/`（待建），产物进 `outputs/paper_v4/`。

### 8.4 Paper v4 跑法登记（2026-07-20）

| 路径 | 说明 |
|------|------|
| `scripts/paper_v4_run/run_b1_mean_edl_t0_oct_only.sh` | B1 T0：mean + 患者级 EDL；内部 val 早停；external 终评 |
| `scripts/paper_v4_run/README_2026-07-20.md` | v4 脚本说明 |
| `outputs/paper_v4/baseline/b1_mean_edl_t0/` | B1 T0 产物（本地，通常不入库） |

```bash
./run_experiment.sh --detach ./tsy_loho ./scripts/paper_v4_run/run_b1_mean_edl_t0_oct_only.sh
```
