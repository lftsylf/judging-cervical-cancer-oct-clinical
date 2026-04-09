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
