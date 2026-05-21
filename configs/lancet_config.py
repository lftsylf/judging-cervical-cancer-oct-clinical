import os

# 环境变量工具函数，用于获取布尔类型的环境变量
def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

# 环境变量工具函数，用于获取浮点数类型的环境变量
def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


# 环境变量工具函数，用于获取整数类型的环境变量
def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


class Config:
    # --- 1. 路径配置 ---
    # 项目根目录
    PROJECT_ROOT = "/ssd_data/tsy_study_venv/OptiGenesis_Lancet"
    # 数据存放目录
    DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
    # 默认输出根目录（无环境变量时）。常用覆盖方式：
    # - baseline 三中心复核：run_loho_3centers.sh → 建议 export OPTIGENESIS_OUTPUT_DIR=baseline_recheck_50_outputs
    # - ViT 对比：自行设置 OPTIGENESIS_BACKBONE=vit_small_patch16_224 等
    # - v2 默认骨干为 ResNet50；复现旧稿 Swin：OPTIGENESIS_BACKBONE=swin_tiny_patch4_window7_224
    # - 同一医院数据多次实验不同输出子目录：export OPTIGENESIS_OUTPUT_RUN_NAME=huaxi3（仍配合 HOSPITAL_NAME 选 CSV）
    OUTPUT_DIR = os.getenv(
        "OPTIGENESIS_OUTPUT_DIR",
        os.path.join(PROJECT_ROOT, "outputs"),
    )
    # --- 2. 多中心列表（仅作文档备忘；当前 LOHO 代码路径未读取下列名称）---
    # 曾设想的训练中心示例: Hubei_Xiangyang, Hubei_Enshi, Hubei_Wuda, Hubei_Jingzhou, Hubei_Shiyan
    # 曾设想的外部中心示例: Sichuan_WestChina(华西), Hunan_Xiangya(湘雅), Liaoning_Tumor, Henan_Zhengda3

    # --- 3. 数据超参 ---
    IMG_SIZE = 224
    NUM_SLICES = 12 # 新增，固定每个病例的切片数量
    # 默认 4；ViT-Base 多帧输入若 OOM，可 export OPTIGENESIS_BATCH_SIZE=2
    try:
        BATCH_SIZE = int(os.getenv("OPTIGENESIS_BATCH_SIZE", "4"))
    except ValueError:
        BATCH_SIZE = 4
    NUM_WORKERS = 4  # 增加worker数量加快数据加载
    
    # --- 4. 模型超参 ---
    # 代码与仓库内命名统一为 OptiGenesis；论文中的展示名称为 MUSE（勿在 Python 模块名中用 MUSE）。
    MODEL_NAME = "OptiGenesis_v2"
    # 视觉骨干（timm）：v2 默认 ResNet50；可改 resnet18、或复现旧稿用 swin_tiny_patch4_window7_224 等
    BACKBONE = os.getenv("OPTIGENESIS_BACKBONE", "resnet50")
    # 临床多模态融合；批量脚本可 export OPTIGENESIS_USE_CLINICAL=1/0
    USE_CLINICAL = _env_bool("OPTIGENESIS_USE_CLINICAL", True)
    USE_UNCERTAINTY = True  # 开启不确定性估计 (Lancet核心亮点)
    NUM_CLASSES = 2         # 二分类: < CIN2 (阴性) vs >= CIN2 (阳性)
    
    # --- 5. 训练超参 ---
    # 允许通过环境变量覆盖，便于批量脚本循环调用（例如 T0 设为 30）
    EPOCHS = _env_int("OPTIGENESIS_EPOCHS", 30)
    # 全网统一学习率（未设置分层 LR 时使用）
    LR = _env_float("OPTIGENESIS_LR", 5e-5)
    # 分层 LR：同时 export OPTIGENESIS_BACKBONE_LR 与 OPTIGENESIS_HEAD_LR 后生效（互斥于「仅 LR」）
    # 例：backbone 1e-5、head 1e-4 → 第 2 组探路；backbone 5e-6、head 1e-4 → 第 3 组（与第 2 组是两次独立实验，不是同时设两个 backbone）
    _bb_lr = os.getenv("OPTIGENESIS_BACKBONE_LR", "").strip()
    _hd_lr = os.getenv("OPTIGENESIS_HEAD_LR", "").strip()
    BACKBONE_LR = float(_bb_lr) if _bb_lr else None
    HEAD_LR = float(_hd_lr) if _hd_lr else None
    # 前 N 个 epoch 冻结 vision_backbone，只训融合层/头；解冻后按 BACKBONE_LR/HEAD_LR 或 LR 建优化器
    FREEZE_BACKBONE_EPOCHS = _env_int("OPTIGENESIS_FREEZE_BACKBONE_EPOCHS", 0)
    # 冻结阶段仅训练 head 时使用的 LR（默认与 HEAD_LR 相同，否则回退到 LR）
    FREEZE_HEAD_LR = _env_float("OPTIGENESIS_FREEZE_HEAD_LR", HEAD_LR if HEAD_LR is not None else LR)
    WEIGHT_DECAY = _env_float("OPTIGENESIS_WEIGHT_DECAY", 1e-4)
    # 允许通过环境变量覆盖，便于多 seed 复现实验
    SEED = _env_int("OPTIGENESIS_SEED", 42)
    # ⚠️ 注意：数据加载器已启用 WeightedRandomSampler (过采样)，保证了Batch内正负样本约 1:1。
    # 因此这里不需要设置极端的反比权重，否则会导致“双重加权”，模型全猜阳性。
    # 只需微调 (1.2~1.5) 以稍微偏向 Recall 即可。v2 单模态 T1 定稿用 1.25。
    POS_WEIGHT = _env_float("OPTIGENESIS_POS_WEIGHT", 1.25)
        
    # --- 6. 数据集配置 ---
    # 定义当前的医院名称 (小写,与 CSV 文件名对应)
    # 在主程序中，可以通过 Config.HOSPITAL_NAME 动态获取
    HOSPITAL_NAME = os.getenv("HOSPITAL_NAME", "huaxi")
    
    # --- 6. 不确定性 Loss 配置 ---
    # KL散度退火周期：前10个epoch主要学准确率，后面慢慢加不确定性约束
    KL_ANNEALING_EPOCHS = 10

    # WMA Loss（重加权边距调整 + EDL KL）；baseline 脚本常 export OPTIGENESIS_USE_WMA=0
    # 说明（对应 training/losses.py::wma_loss）：
    # - WMA_C → c_margin：边距整体强度，越大对 alpha 的修正越激进（常用约 0.2）
    # - WMA_WARMUP_EPOCHS → warmup_epochs：边距系数 λ 从 0 线性升到 1 的 epoch 数
    # - WMA_TEMPERATURE → τ：TCE-MA 项尺度，默认 1.0 为不额外缩放
    USE_WMA_LOSS = _env_bool("OPTIGENESIS_USE_WMA", True)
    WMA_C = _env_float("OPTIGENESIS_WMA_C", 0.2)
    WMA_WARMUP_EPOCHS = _env_int("OPTIGENESIS_WMA_WARMUP", 10)
    WMA_TEMPERATURE = _env_float("OPTIGENESIS_WMA_TEMP", 1.0)

    # --- 7. 多模态辅助监督aux（低成本创新点，可开关）---
    # 思路：融合分支之外，给视觉分支和临床分支各加一个轻量辅助分类头，
    # 通过辅助监督减少“某一模态偷懒”的现象，提升跨中心鲁棒性。
    # 默认关闭。完整三中心30轮负/正消融，每次和ema一起开启
    # ViT baseline 脚本请保持 OPTIGENESIS_ENABLE_AUX 未设置或显式 0。
    ENABLE_MULTIMODAL_AUX_LOSS = _env_bool("OPTIGENESIS_ENABLE_AUX", False)
    AUX_LOSS_WEIGHT_VISION = 0.2
    AUX_LOSS_WEIGHT_CLINICAL = 0.2

    # --- 8. Model EMA（轻量消融：开启后每个 step 更新 shadow；验证/选模/存盘/导出均用 EMA 权重）---
    # 默认关闭；仅当 OPTIGENESIS_ENABLE_EMA=1 时开启
    ENABLE_MODEL_EMA = _env_bool("OPTIGENESIS_ENABLE_EMA", False)
    EMA_DECAY = 0.999  # 小数据可试 0.99–0.9999；轮数少时可略降 decay 使 shadow 更快跟上

    # --- 9. 无监督域对齐 CORAL（最小 UDA；目标域=当前 fold 的 external CSV，训练时不使用其标签）---
    # 仅 CORAL（协方差对齐），在融合后 256 维特征上计算；λ 带 warmup，默认偏小以降低压制主任务的风险。
    # 历史实验里 λ 过大易伤 AUC；若仍 OOM，可 OPTIGENESIS_BATCH_SIZE=2。
    ENABLE_DOMAIN_CORAL = _env_bool("OPTIGENESIS_ENABLE_CORAL", False)
    CORAL_LAMBDA_MAX = _env_float("OPTIGENESIS_CORAL_LAMBDA", 0.02)
    CORAL_WARMUP_EPOCHS = _env_int("OPTIGENESIS_CORAL_WARMUP", 8)

    @classmethod
    def make_dirs(cls):
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "logs"), exist_ok=True)

