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
    # 历史字段：每病例约 12 个点位 TIFF；展开多页后 N≈60（辽宁）或 120（华西/湘雅）
    NUM_SLICES = 12
    # 是否把每个 TIFF 的全部时序页展开为独立帧（病灶常只在少数页可见）
    # 关闭（默认）：与历史实验一致，PIL 只读第一页 → 每患者 N≈12
    # 开启：export OPTIGENESIS_EXPAND_TIFF_PAGES=1 → N≈60/120；建议同时 OPTIGENESIS_BATCH_SIZE=1 或 2
    EXPAND_TIFF_PAGES = _env_bool("OPTIGENESIS_EXPAND_TIFF_PAGES", False)
    # 每个 TIFF 最多取前多少页；0=不截断（辽宁常见 5，华西/湘雅常见 10）
    MAX_PAGES_PER_TIFF = _env_int("OPTIGENESIS_MAX_PAGES_PER_TIFF", 0)
    # --- site-bag 协议（湘雅内部 n 点×全页；val/test 多窗 OR）---
    # 开启后：每次只取 SITEBAG_N 个钟点 TIFF，并强制展开该 TIFF 全部页；
    # 训练按 pos_sites 优先组袋；val/external 按钟点不重叠切窗，validate 内按患者 OR。
    SITEBAG_ENABLE = _env_bool("OPTIGENESIS_SITEBAG", False)
    SITEBAG_N = _env_int("OPTIGENESIS_SITEBAG_N", 2)
    # soft OR：患者分 = max(窗概率)，用于 AUC/早停；硬判定 ≡ max_p>0.5（任一窗>0.5）
    # 兼容旧开关：SITEBAG_EVAL_OR=0 时强制用 mean（弱化）
    SITEBAG_EVAL_OR = _env_bool("OPTIGENESIS_SITEBAG_EVAL_OR", True)
    # 多窗聚合成患者分：max（默认 OR）| mean（弱化）| first（只取第一窗，相当于关多窗 OR）
    SITEBAG_EVAL_AGG = os.getenv("OPTIGENESIS_SITEBAG_EVAL_AGG", "max").strip().lower()
    # 骨干按帧分块前向（+checkpoint），展开 N=120 时防 OOM；0=不分块
    FRAME_ENCODE_CHUNK = _env_int("OPTIGENESIS_FRAME_ENCODE_CHUNK", 16)
    # 默认 4；展开多页后极易 OOM，可 export OPTIGENESIS_BATCH_SIZE=1
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

    # --- 4b. 多帧聚合（v4 主方法：帧级 EDL + 不确定加权）---
    # mean                  : 等权均值池化特征 → 患者级 EDL（≈ v3 / baseline B1）
    # equal                 : 帧级 EDL → 等权平均 α（消融：有帧 u、无加权）
    # uncertainty_weighted  : 帧级 EDL → softmax((1−u)/τ) 加权聚合 α（Ours）
    # attention             : 帧级 EDL → 可学习注意力加权（u 仅导出，不参与加权）
    FRAME_AGG_MODE = os.getenv("OPTIGENESIS_FRAME_AGG", "uncertainty_weighted").strip().lower()
    # 加权温度 τ：越小越「只信最确定的几帧」；越大越接近等权（attention 同样用）
    FRAME_AGG_TEMPERATURE = _env_float("OPTIGENESIS_FRAME_AGG_TEMP", 0.5)
    # attention 的 query 构造（仅 FRAME_AGG=attention）：
    #   mean     : 有效帧融合特征均值（旧默认；易退化成近均值池化）
    #   evidence : 用帧级 EDL 阳性概率 p+ 对融合特征加权得到 query（打破 mean 循环）
    FRAME_ATTN_QUERY = os.getenv("OPTIGENESIS_FRAME_ATTN_QUERY", "mean").strip().lower()
    # 帧权重信号（仅 FRAME_AGG=uncertainty_weighted 时生效）：
    #   edl_u     : softmax((1−u)/τ)，u=K/Σα（默认，历史 UW）
    #   edl_u_amp : softmax(((u_base−u)·scale)/τ)，把挤在 0.2–0.3 的 u 差放大后再加权
    #   maxprob   : softmax(max_k p_k / τ)，p=α/Σα（按帧「最自信类别」打分再软加权）
    #   negent    : softmax((−H − mean(−H))/τ)，H 为帧预测熵
    #   topk_p    : 按阳性 p 取 top-k 帧等权（默认 k=5；训练期硬选）
    #   max_p_pool: 只取阳性 p 最大的 1 帧（探针 max_p 的训练版；非 hard-OR）
    FRAME_WEIGHT_SIGNAL = os.getenv("OPTIGENESIS_FRAME_WEIGHT_SIGNAL", "edl_u").strip().lower()
    # edl_u_amp 专用：score=(u_base−u)*scale；默认 u_base=0.5、scale=10（见 run 脚本注释）
    FRAME_U_SCORE_BASE = _env_float("OPTIGENESIS_FRAME_U_SCORE_BASE", 0.5)
    FRAME_U_SCORE_SCALE = _env_float("OPTIGENESIS_FRAME_U_SCORE_SCALE", 10.0)
    # topk_p 专用
    FRAME_TOPK_K = _env_int("OPTIGENESIS_FRAME_TOPK_K", 5)
    # 每例患者导出不确定度最高的 top-k 帧下标，供人工复核
    FRAME_REVIEW_TOP_K = _env_int("OPTIGENESIS_FRAME_REVIEW_TOP_K", 3)
    
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
    # 患者级 one-hot 标签平滑 ε：y'=(1−ε)y + ε/K；0=关闭。与帧级 EDL 辅损不冲突（只改 bag 主损标签）
    LABEL_SMOOTHING = _env_float("OPTIGENESIS_LABEL_SMOOTHING", 0.0)

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
    AUX_LOSS_WEIGHT_VISION = _env_float("OPTIGENESIS_AUX_W_VISION", 0.2)
    AUX_LOSS_WEIGHT_CLINICAL = _env_float("OPTIGENESIS_AUX_W_CLINICAL", 0.2)

    # --- 7b. 帧级弱监督（单模态；与 multimodal Aux 无关）---
    # 仅在 FRAME_AGG=equal / uncertainty_weighted / attention 时生效：
    # 对帧级 α 加一小权重的 EDL/Focal 或 CE（broadcast 或 mil，见 FRAME_AUX_MODE）。
    # 默认关闭。建议：最终方法 = Ours(uw) + 本项；旧 Ours(无帧损) 作「去掉帧级弱监督」消融。
    ENABLE_FRAME_AUX_LOSS = _env_bool("OPTIGENESIS_ENABLE_FRAME_AUX", False)
    FRAME_AUX_LOSS_WEIGHT = _env_float("OPTIGENESIS_FRAME_AUX_WEIGHT", 0.2)
    # edl：与主损失同族（Focal+EDL 或纯 EDL，随 USE_WMA/focal 开关）；ce：对 p=α/S 做交叉熵
    FRAME_AUX_LOSS_TYPE = os.getenv("OPTIGENESIS_FRAME_AUX_TYPE", "edl").strip().lower()
    # 帧辅损模式：
    #   broadcast — 旧法：患者标签广播到每一帧（噪声标签，易把无病灶点当阳）
    #   mil       — MIL：阴性袋各点压阴；阳性袋若已有点够阳则不再辅损，
    #               否则只对「最阳的那一点」施压（至少一点支持袋标签）
    FRAME_AUX_MODE = os.getenv("OPTIGENESIS_FRAME_AUX_MODE", "broadcast").strip().lower()
    # mil 判定「该点检出阳性」的阈值（帧阳性概率 p_pos = α_pos/Σα）
    FRAME_AUX_MIL_POS_THR = _env_float("OPTIGENESIS_FRAME_AUX_MIL_POS_THR", 0.5)
    # 帧辅损是否跟随患者主损使用 WMA；默认跟随。attn+broadcast 叠 WMA 时建议关（只患者级 WMA）
    FRAME_AUX_USE_WMA = _env_bool("OPTIGENESIS_FRAME_AUX_USE_WMA", True)

    # --- 8. Model EMA（轻量消融：开启后每个 step 更新 shadow；验证/选模/存盘/导出均用 EMA 权重）---
    # 默认关闭；仅当 OPTIGENESIS_ENABLE_EMA=1 时开启
    ENABLE_MODEL_EMA = _env_bool("OPTIGENESIS_ENABLE_EMA", False)
    # 小数据 / 早停轮数少时可降到 0.99，使 shadow 更快跟上
    EMA_DECAY = _env_float("OPTIGENESIS_EMA_DECAY", 0.999)

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

