import os

class Config:
    # --- 1. 路径配置 ---
    # 项目根目录
    PROJECT_ROOT = "/data2/hmy/Primary_care/OptiGenesis_Lancet"
    # 数据存放目录 (请确保此处有 train_5centers.csv, val_5centers.csv 等)
    DATA_ROOT = "/data2/hmy/Primary_care/data"
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    
    # --- 2. 多中心设置 (9家医院) ---
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
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    
    # --- 4. 模型超参 ---
    MODEL_NAME = "OptiGenesis_v1"
    BACKBONE = "swin_tiny_patch4_window7_224" # 既轻量又强
    USE_CLINICAL = True     # 开启多模态融合
    USE_UNCERTAINTY = True  # 开启不确定性估计 (Lancet核心亮点)
    NUM_CLASSES = 2         # 二分类: < CIN2 (阴性) vs >= CIN2 (阳性)
    
    # --- 5. 训练超参 ---
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    SEED = 42
    
    # --- 6. 不确定性 Loss 配置 ---
    # KL散度退火周期：前10个epoch主要学准确率，后面慢慢加不确定性约束
    KL_ANNEALING_EPOCHS = 10 

    @classmethod
    def make_dirs(cls):
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cls.OUTPUT_DIR, "logs"), exist_ok=True)

