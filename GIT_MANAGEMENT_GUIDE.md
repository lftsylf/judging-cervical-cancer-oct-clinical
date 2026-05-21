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
| `feature/v2-resnet-baseline` | 分支 | `2075111`（当前定稿） | v2 开发线：ResNet50 + 临床消融 + OCT-only 定稿 |
| `paper-v2-resnet50-oct-baseline` | annotated tag | `2075111` | v2 定稿 baseline：ResNet50、OCT-only、LR=5e-5、POS_WEIGHT=1.25 |

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
