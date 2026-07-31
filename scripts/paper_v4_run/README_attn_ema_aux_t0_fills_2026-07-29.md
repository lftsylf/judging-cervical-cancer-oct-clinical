# EMA+Aux T0 + WMA/Aux 补 seed（2026-07-29）

## 为何这样配
- 探路：EMA alone 很差；Aux / WMA 与 Attn 基本持平
- 你需要的是「模块一起有提升」；常见搭配先试 **EMA∥Aux**（不加 WMA，避免再叠全家桶）
- 同时把 **WMA、Aux 单独** 补满 5-seed，便于后来说清单模块定稿

## 参数（已降强度；可用环境变量再调）
| 项 | 值 |
|----|-----|
| EMA decay | **0.99**（`OPTIGENESIS_EMA_DECAY`） |
| Aux vision | **0.1**（`OPTIGENESIS_AUX_W_VISION`） |
| WMA C | **0.1**，帧辅跟随（`OPTIGENESIS_WMA_C`） |

## 四卡
| GPU | 任务 |
|----:|------|
| 0–1 | EMA+Aux · 满 5 seeds（T0） |
| 2 | WMA alone 补 2024/3407/114514 |
| 3 | Aux alone 补 2024/3407/114514 |

## 启动
```bash
./run_experiment.sh --detach ./tsy_loho \
  ./scripts/paper_v4_run/run_overnight_quad_gpu_attn_ema_aux_t0_and_fills.sh
```
