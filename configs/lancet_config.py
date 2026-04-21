import os


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


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
    # 数据存放目录 (AnYang数据集路径)
    DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset")
    # 默认输出根目录（无环境变量时）。常用覆盖方式：
    # - baseline 三中心复核：run_loho_3centers.sh → 建议 export OPTIGENESIS_OUTPUT_DIR=baseline_recheck_50_outputs
    # - 多模态辅助监督 50 轮消融：bash run_loho_aux_50.sh（脚本内 export OPTIGENESIS_ENABLE_AUX=1、关闭 EMA、OUTPUT_DIR=aux_50_outputs）
    # - CORAL 域对齐 50 轮：bash run_loho_uda_50.sh（OPTIGENESIS_ENABLE_CORAL=1，输出 uda_50_outputs）
    # - ViT 对比：run_loho_vit_base.sh 等自行设置 OPTIGENESIS_BACKBONE 与 OPTIGENESIS_OUTPUT_DIR
    # - 同一医院数据多次实验不同输出子目录：export OPTIGENESIS_OUTPUT_RUN_NAME=huaxi3（仍配合 HOSPITAL_NAME 选 CSV）
    OUTPUT_DIR = os.getenv(
        "OPTIGENESIS_OUTPUT_DIR",
        os.path.join(PROJECT_ROOT, "baseline_recheck_50_outputs"),
    )
    
    # --- 2. 多中心设置 (9家医院) 现在不用，只有三家医院---
    # 训练集 (湖北5家)
    TRAIN_CENTERS = [
        "Hubei_Xiangyang", "Hubei_Enshi", "Hubei_Wuda", "Hubei_Jingzhou", "Hubei_Shiyan"
    ]
    # 外部测试集 (全国4家顶级)
    EXTERNAL_TEST_CENTERS = [
        "Sichuan_WestChina",    # 华西
        "Hunan_Xiangya",        # 湘雅
        "Liaoning_Tumor",       # 辽宁肿瘤
        "Henan_Zhengda3"        # 郑大三附院
    ]

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
    MODEL_NAME = "OptiGenesis_v1"
    # 默认 Swin；ViT 对比可设置环境变量 OPTIGENESIS_BACKBONE=vit_small_patch16_224 等
    BACKBONE = os.getenv("OPTIGENESIS_BACKBONE", "swin_tiny_patch4_window7_224")
    # 临床多模态融合；批量脚本可 export OPTIGENESIS_USE_CLINICAL=1/0
    USE_CLINICAL = _env_bool("OPTIGENESIS_USE_CLINICAL", True)
    USE_UNCERTAINTY = True  # 开启不确定性估计 (Lancet核心亮点)
    NUM_CLASSES = 2         # 二分类: < CIN2 (阴性) vs >= CIN2 (阳性)
    
    # --- 5. 训练超参 ---
    # 允许通过环境变量覆盖，便于批量脚本循环调用（例如 T0 设为 30）
    EPOCHS = _env_int("OPTIGENESIS_EPOCHS", 50)
    LR = 5e-5
    WEIGHT_DECAY = 1e-4
    # 允许通过环境变量覆盖，便于多 seed 复现实验
    SEED = _env_int("OPTIGENESIS_SEED", 42)
    # ⚠️ 注意：数据加载器已启用 WeightedRandomSampler (过采样)，保证了Batch内正负样本约 1:1。
    # 因此这里不需要设置极端的反比权重 (如 5.38)，否则会导致“双重加权”，模型全猜阳性。
    # 只需微调 (1.2~1.5) 以稍微偏向 Recall 即可。
    POS_WEIGHT = 1.3
        
    # --- 6. 数据集配置 ---
    # 定义当前的医院名称 (小写,与 CSV 文件名对应)
    # 在主程序中，可以通过 Config.HOSPITAL_NAME 动态获取
    HOSPITAL_NAME = os.getenv("HOSPITAL_NAME", "zhengdasanfu")
    
    # --- 6. 不确定性 Loss 配置 ---
    # KL散度退火周期：前10个epoch主要学准确率，后面慢慢加不确定性约束
    KL_ANNEALING_EPOCHS = 10

    # WMA Loss（重加权边距调整的证据不确定性感知损失）；baseline 脚本请 export OPTIGENESIS_USE_WMA=0
    USE_WMA_LOSS = _env_bool("OPTIGENESIS_USE_WMA", True)
    WMA_C = _env_float("OPTIGENESIS_WMA_C", 0.2)
    WMA_WARMUP_EPOCHS = _env_int("OPTIGENESIS_WMA_WARMUP", 10)
    WMA_TEMPERATURE = _env_float("OPTIGENESIS_WMA_TEMP", 1.0)

    # --- 7. 多模态辅助监督（低成本创新点，可开关）---
    # 思路：融合分支之外，给视觉分支和临床分支各加一个轻量辅助分类头，
    # 通过辅助监督减少“某一模态偷懒”的现象，提升跨中心鲁棒性。
    # 默认关闭。完整三中心 50 轮负/正消融：bash run_loho_aux_50.sh
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

