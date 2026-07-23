import os
import sys

# expandable_segments 需 PyTorch >= 2.1；2.0.x 会在首次 .cuda() 时报 Unrecognized CachingAllocator option
def _configure_cuda_allocator():
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return
    try:
        import torch
        ver = torch.__version__.split("+")[0]
        parts = [int(x) for x in ver.split(".")[:2]]
        if parts[0] > 2 or (parts[0] == 2 and parts[1] >= 1):
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        pass


_configure_cuda_allocator()

import torch
import torch.optim as optim
import pandas as pd
import random

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
# 添加scripts目录到路径（用于工具脚本）
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from configs.lancet_config import Config
from data.dataset_lancet import get_dataloader
from models.optigenesis_model import OptiGenesis
from training.trainer import train_epoch, validate
from training.ema import ModelEMA
import numpy as np


def _set_backbone_trainable(model: OptiGenesis, trainable: bool) -> None:
    for p in model.vision_backbone.parameters():
        p.requires_grad = trainable


def _head_parameters(model: OptiGenesis):
    for name, param in model.named_parameters():
        if not name.startswith("vision_backbone."):
            yield param


def _describe_lr_plan() -> str:
    freeze_n = int(getattr(Config, "FREEZE_BACKBONE_EPOCHS", 0) or 0)
    bb = getattr(Config, "BACKBONE_LR", None)
    hd = getattr(Config, "HEAD_LR", None)
    if freeze_n > 0:
        f_lr = float(getattr(Config, "FREEZE_HEAD_LR", Config.LR))
        if bb is not None and hd is not None:
            after = f"解冻后 backbone={bb}, head={hd}"
        else:
            after = f"解冻后全网 LR={Config.LR}"
        return f"冻结 backbone 前 {freeze_n} 个 epoch（仅训 head, LR={f_lr}）；{after}"
    if bb is not None and hd is not None:
        return f"分层 LR: backbone={bb}, head(融合层+EDL 等)={hd}"
    return f"全网统一 LR={Config.LR}"


def build_optimizer(model: OptiGenesis, epoch: int = 0) -> optim.AdamW:
    freeze_n = int(getattr(Config, "FREEZE_BACKBONE_EPOCHS", 0) or 0)
    wd = float(Config.WEIGHT_DECAY)
    bb_lr = getattr(Config, "BACKBONE_LR", None)
    hd_lr = getattr(Config, "HEAD_LR", None)

    if freeze_n > 0 and epoch < freeze_n:
        _set_backbone_trainable(model, False)
        lr = float(getattr(Config, "FREEZE_HEAD_LR", hd_lr if hd_lr is not None else Config.LR))
        return optim.AdamW(list(_head_parameters(model)), lr=lr, weight_decay=wd)

    _set_backbone_trainable(model, True)
    if bb_lr is not None and hd_lr is not None:
        return optim.AdamW(
            [
                {"params": list(model.vision_backbone.parameters()), "lr": bb_lr},
                {"params": list(_head_parameters(model)), "lr": hd_lr},
            ],
            weight_decay=wd,
        )
    return optim.AdamW(model.parameters(), lr=float(Config.LR), weight_decay=wd)


def build_scheduler(optimizer: optim.AdamW, start_epoch: int = 0):
    remaining = max(1, int(Config.EPOCHS) - int(start_epoch))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)


def seed_everything(seed: int):
    """锁定主要随机源，使训练结果可复现。"""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def export_split_predictions(
    model,
    loader,
    source_csv_path,
    split_name,
    hospital_name,
    logs_dir,
    device,
    use_focal_loss,
    class_weights,
    use_wma=False,
    train_class_counts=None,
    wma_c=0.2,
    wma_warmup_epochs=10,
    wma_temperature=1.0,
    kl_annealing_epochs=10,
):
    metrics, pred_details = validate(
        model,
        loader,
        device,
        verbose=False,
        use_focal=use_focal_loss,
        use_wma=use_wma,
        train_class_counts=train_class_counts,
        wma_c=wma_c,
        wma_warmup_epochs=wma_warmup_epochs,
        wma_temperature=wma_temperature,
        kl_annealing_epochs=kl_annealing_epochs,
        class_weights=class_weights,
        epoch=Config.EPOCHS - 1,
        total_epochs=Config.EPOCHS,
        return_predictions=True,
    )

    df = pd.read_csv(source_csv_path).reset_index(drop=True)
    n = min(len(df), len(pred_details['targets']))
    split_label = {
        "train": "训练集",
        "val": "内部验证集",
        "external": "外部终评集",
        "development": "开发集",
    }.get(split_name, split_name)
    if n == 0:
        print(f"⚠️ 【{split_label}】逐样本结果为空，跳过导出。")
        return None, metrics

    df = df.iloc[:n].copy()
    df['hospital'] = hospital_name
    df['split'] = split_name
    df['y_true'] = pred_details['targets'][:n].astype(int)
    df['prob_positive'] = pred_details['probs'][:n].astype(float)
    df['uncertainty'] = pred_details['uncertainties'][:n].astype(float)
    df['y_pred'] = pred_details['preds'][:n].astype(int)

    # 帧级不确定度 / 权重 / 建议复核帧（高 u）——人工复核提示
    frame_u = pred_details.get('frame_uncertainties')
    frame_w = pred_details.get('frame_weights')
    review_idx = pred_details.get('review_frame_indices')
    if frame_u is not None and len(frame_u) >= n and frame_u.ndim == 2:
        n_frames = frame_u.shape[1]
        for i in range(n_frames):
            df[f'frame_u_{i}'] = frame_u[:n, i].astype(float)
            if frame_w is not None and frame_w.shape == frame_u.shape:
                df[f'frame_w_{i}'] = frame_w[:n, i].astype(float)
        if review_idx is not None and len(review_idx) >= n:
            # 例如 "7;3;11" —— 不确定度从高到低的帧下标
            df['review_frame_indices'] = [
                ";".join(str(int(x)) for x in row)
                for row in review_idx[:n]
            ]

    output_path = os.path.join(logs_dir, f"{split_name}_sample_predictions.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"【{split_label}】逐样本概率已保存: {output_path}")
    if frame_u is not None and len(frame_u) >= n and frame_u.ndim == 2:
        review_path = os.path.join(logs_dir, f"{split_name}_frame_review_hints.csv")
        review_cols = ['hospital', 'split', 'y_true', 'y_pred', 'prob_positive', 'uncertainty', 'review_frame_indices']
        review_cols = [c for c in review_cols if c in df.columns]
        extra_u = [c for c in df.columns if c.startswith('frame_u_')]
        df[review_cols + extra_u].to_csv(review_path, index=False, encoding='utf-8-sig')
        print(f"【{split_label}】高不确定帧复核提示已保存: {review_path}")
    return output_path, metrics


def main():
    seed_everything(Config.SEED)
    print(f"【随机种子】SEED={Config.SEED}（已锁定 Python/Numpy/PyTorch/cudnn）")

    # 1. 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"【项目】OptiGenesis (Lancet)  |  【设备】{device}")
    
    # 2. 数据加载（v4 协议：train / 内部 val / external 终评）
    # 折名仍 = 外部测试医院；选模与早停只用 val_*.csv，external 训练过程不用
    hospital_name = Config.HOSPITAL_NAME.lower()
    run_folder = os.getenv("OPTIGENESIS_OUTPUT_RUN_NAME", "").strip().lower()
    if not run_folder or os.path.sep in run_folder or ".." in run_folder:
        run_folder = hospital_name
    train_csv = os.path.join(Config.DATA_ROOT, f"train_{hospital_name}.csv")
    val_csv = os.path.join(Config.DATA_ROOT, f"val_{hospital_name}.csv")
    external_csv = os.path.join(Config.DATA_ROOT, f"external_{hospital_name}.csv")
    run_output_dir = os.path.join(Config.OUTPUT_DIR, run_folder)
    checkpoints_dir = os.path.join(run_output_dir, "checkpoints")
    logs_dir = os.path.join(run_output_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"【数据目录】{Config.DATA_ROOT}")
    print(f"【输出目录】{run_output_dir}")
    print(
        f"【当前医院折】{Config.HOSPITAL_NAME}  |  "
        f"训练: {train_csv}  |  内部验证: {val_csv}  |  外部终评: {external_csv}"
    )
    
    try:
        missing = [p for p in (train_csv, val_csv, external_csv) if not os.path.exists(p)]
        if missing:
            print("⚠️ 未找到 CSV:", missing)
            print(
                "请先运行: python data/prepare_paper_v4_splits.py --write\n"
                "（需已有 development_*.csv / external_*.csv）"
            )
            
        # DataLoader 使用固定 seed + worker_init_fn，确保多进程增强与采样可复现
        train_loader = get_dataloader(train_csv, mode='train', seed=Config.SEED)
        val_loader = get_dataloader(val_csv, mode='val', seed=Config.SEED + 1000)
        external_loader = get_dataloader(external_csv, mode='val', seed=Config.SEED + 2000)
        train_labels = np.asarray(getattr(train_loader.dataset, "labels", []), dtype=np.int64)
        train_class_counts = np.bincount(train_labels, minlength=2).tolist()
        enable_coral = getattr(Config, "ENABLE_DOMAIN_CORAL", False)
        # v4：禁止用 external 做 CORAL 目标域（避免测试域参与训练）；CORAL 默认应关闭
        if enable_coral:
            print(
                "⚠️ v4 协议下已禁用 CORAL 使用 external；"
                "请关闭 OPTIGENESIS_ENABLE_CORAL，或后续改为仅用内部无标签源。"
            )
        uda_target_loader = None
        
        # -------------------------------------------------------------------------
        # 核心策略调整：针对“零漏诊”需求的手动加权
        # -------------------------------------------------------------------------
        # 使用 Config 中的配置，根据数据集动态调整
        actual_pos_weight = getattr(Config, 'POS_WEIGHT', 1.05)
        print(f"🔥 【策略】阳性类别权重 POS_WEIGHT = {actual_pos_weight}")
        class_weights = torch.tensor([1.0, actual_pos_weight], dtype=torch.float32).to(device)
        print(f" 🎯 设定类别权重: 阴性={class_weights[0]:.2f}, 阳性={class_weights[1]:.2f}")

    except Exception as e:
        print(f" 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        print(
            "请确保已运行 prepare_loho_data.py 与 "
            "python data/prepare_paper_v4_splits.py --write"
        )
        return
    
    # 3. 模型构建
    frame_agg_mode = getattr(Config, "FRAME_AGG_MODE", "uncertainty_weighted")
    frame_agg_temp = float(getattr(Config, "FRAME_AGG_TEMPERATURE", 0.5))
    frame_review_k = int(getattr(Config, "FRAME_REVIEW_TOP_K", 3))
    frame_weight_signal = getattr(Config, "FRAME_WEIGHT_SIGNAL", "edl_u")
    print(
        f"【构建模型】骨干网络: {Config.BACKBONE}  |  临床特征融合: {Config.USE_CLINICAL}  |  "
        f"帧聚合: {frame_agg_mode} (signal={frame_weight_signal}, τ={frame_agg_temp}, "
        f"review_top_k={frame_review_k})"
    )
    print(
        f" TIFF 时序: EXPAND_TIFF_PAGES={bool(getattr(Config, 'EXPAND_TIFF_PAGES', False))} "
        f"| MAX_PAGES_PER_TIFF={getattr(Config, 'MAX_PAGES_PER_TIFF', 0)} "
        f"| BATCH_SIZE={Config.BATCH_SIZE}"
    )
    model = OptiGenesis(
        model_name=Config.BACKBONE,
        use_clinical=Config.USE_CLINICAL,
        num_classes=Config.NUM_CLASSES,
        frame_agg_mode=frame_agg_mode,
        agg_temperature=frame_agg_temp,
        review_top_k=frame_review_k,
        weight_signal=frame_weight_signal,
    ).to(device)
    
    freeze_backbone_epochs = int(getattr(Config, "FREEZE_BACKBONE_EPOCHS", 0) or 0)
    optimizer = build_optimizer(model, epoch=0)
    scheduler = build_scheduler(optimizer, start_epoch=0)
    print(f"【学习率】{_describe_lr_plan()}")

    use_ema = getattr(Config, "ENABLE_MODEL_EMA", False)
    ema_decay = getattr(Config, "EMA_DECAY", 0.999)
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    
    # 4. 训练循环
    best_auc = 0.0
    best_f1 = 0.0
    best_mcc = -1.0
    early_stop_patience = 10
    epochs_without_improve = 0
    metrics_history = []

    use_wma = bool(getattr(Config, "USE_WMA_LOSS", False))
    use_focal_loss = not use_wma
    wma_c = float(getattr(Config, "WMA_C", 0.2))
    wma_warmup = int(getattr(Config, "WMA_WARMUP_EPOCHS", 10))
    wma_temp = float(getattr(Config, "WMA_TEMPERATURE", 1.0))
    kl_ann = int(getattr(Config, "KL_ANNEALING_EPOCHS", 10))

    print("【开始训练】…")
    if use_wma:
        print(
            f" 主损失: WMA Loss（C={wma_c}, warmup={wma_warmup}, τ={wma_temp}）"
            f"  |  N_P=[neg,pos]={train_class_counts}"
        )
    else:
        print(" 主损失: Focal + EDL 组合（含类别权重）；数据侧仍配合过采样")
    print(f" 多模态辅助监督: {getattr(Config, 'ENABLE_MULTIMODAL_AUX_LOSS', False)}")
    enable_frame_aux = bool(getattr(Config, "ENABLE_FRAME_AUX_LOSS", False))
    frame_aux_w = float(getattr(Config, "FRAME_AUX_LOSS_WEIGHT", 0.2))
    frame_aux_type = str(getattr(Config, "FRAME_AUX_LOSS_TYPE", "edl"))
    if enable_frame_aux:
        print(
            f" 帧级弱监督: 开启 (weight={frame_aux_w}, type={frame_aux_type})；"
            f"每帧共用患者标签（单模态，非 clinical Aux）"
        )
    else:
        print(" 帧级弱监督: 关闭")
    if getattr(Config, "ENABLE_DOMAIN_CORAL", False):
        print(" CORAL 域对齐: 配置为开启，但 v4 协议下训练路径已禁用（避免 external 参与训练）")
    else:
        print(" CORAL 域对齐: 关闭")
    print(" 选模与存盘: 仅当【内部验证集 val】ROC-AUC 提升时保存 best_model.pth；external 只终评")
    if use_ema:
        print(f" Model EMA: 开启 (decay={ema_decay})，验证/存盘/导出均使用 EMA 权重")
    else:
        print(" Model EMA: 关闭")

    for epoch in range(Config.EPOCHS):
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs:
            print(
                f"\n🔓 第 {epoch + 1} 轮：解冻 vision_backbone，按新学习率重建优化器 …"
            )
            optimizer = build_optimizer(model, epoch=epoch)
            scheduler = build_scheduler(optimizer, start_epoch=epoch)
            print(f"   {_describe_lr_plan()}")

        # 训练一个 epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            Config.EPOCHS,
            class_weights=class_weights,
            use_focal=use_focal_loss,
            use_wma=use_wma,
            train_class_counts=train_class_counts,
            wma_c=wma_c,
            wma_warmup_epochs=wma_warmup,
            wma_temperature=wma_temp,
            kl_annealing_epochs=kl_ann,
            enable_multimodal_aux=getattr(Config, 'ENABLE_MULTIMODAL_AUX_LOSS', False),
            aux_w_vision=getattr(Config, 'AUX_LOSS_WEIGHT_VISION', 0.2),
            aux_w_clinical=getattr(Config, 'AUX_LOSS_WEIGHT_CLINICAL', 0.2),
            enable_frame_aux=enable_frame_aux,
            frame_aux_weight=frame_aux_w,
            frame_aux_type=frame_aux_type,
            ema=ema,
            uda_target_loader=uda_target_loader,
            lambda_coral_max=(
                getattr(Config, "CORAL_LAMBDA_MAX", 0.0)
                if getattr(Config, "ENABLE_DOMAIN_CORAL", False)
                else 0.0
            ),
            coral_warmup_epochs=getattr(Config, "CORAL_WARMUP_EPOCHS", 8),
        )
        
        # 验证：计算完整指标（开启 EMA 时用 shadow 权重做评估）
        if ema is not None:
            ema.apply_shadow(model)
        try:
            val_metrics = validate(
                model,
                val_loader,
                device,
                verbose=(epoch % 10 == 0 or epoch == Config.EPOCHS - 1),
                use_focal=use_focal_loss,
                use_wma=use_wma,
                train_class_counts=train_class_counts,
                wma_c=wma_c,
                wma_warmup_epochs=wma_warmup,
                wma_temperature=wma_temp,
                kl_annealing_epochs=kl_ann,
                class_weights=class_weights,
                epoch=epoch,
                total_epochs=Config.EPOCHS,
            )
            train_metrics = validate(
                model,
                train_loader,
                device,
                verbose=False,
                use_focal=use_focal_loss,
                use_wma=use_wma,
                train_class_counts=train_class_counts,
                wma_c=wma_c,
                wma_warmup_epochs=wma_warmup,
                wma_temperature=wma_temp,
                kl_annealing_epochs=kl_ann,
                class_weights=class_weights,
                epoch=epoch,
                total_epochs=Config.EPOCHS,
            )
        finally:
            if ema is not None:
                ema.restore(model)
        
        # 保存指标历史
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train': train_metrics,
            'val': val_metrics
        }
        metrics_history.append(epoch_metrics)
        
        # 简洁日志（每epoch）
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | 训练损失: {train_loss:.6f}")
        print(f"{'='*60}")
        print(f"   【验证集关键指标】")
        print(f"      平均损失: {val_metrics['avg_loss']:.6f}")
        print(f"      准确率: {val_metrics['accuracy']:.4f} | 平衡准确率: {val_metrics['balanced_accuracy']:.4f}")
        print(f"      ROC-AUC: {val_metrics['auc_roc']:.4f} | PR-AUC: {val_metrics['auc_pr']:.4f}")
        print(f"      F1(宏平均): {val_metrics['f1_score']:.4f} | MCC: {val_metrics['mcc']:.4f}")
        print(f"      敏感度: {val_metrics['sensitivity']:.4f} | 特异度: {val_metrics['specificity']:.4f}")
        print(f"      PPV: {val_metrics['ppv']:.4f} | NPV: {val_metrics['npv']:.4f}")
        
        scheduler.step()
        
        # 仅当验证集 ROC-AUC 提升时保存 best_model.pth（与论文消融选模口径一致）
        save_path = os.path.join(checkpoints_dir, "best_model.pth")
        if val_metrics['auc_roc'] > best_auc:
            best_auc = val_metrics['auc_roc']
            epochs_without_improve = 0
            if ema is not None:
                ema.apply_shadow(model)
                try:
                    torch.save(model.state_dict(), save_path)
                finally:
                    ema.restore(model)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"    新的最佳ROC-AUC模型! (AUC: {best_auc:.4f})")
            print(f"   💾 模型已保存至: {save_path}")
        else:
            epochs_without_improve += 1
            print(
                f"   ⏳ EarlyStopping 计数: {epochs_without_improve}/{early_stop_patience} "
                f"(当前最佳 AUC={best_auc:.4f})"
            )
        # 以下仅作训练过程记录，不触发保存
        if val_metrics['f1_positive'] > best_f1:
            best_f1 = val_metrics['f1_positive']
        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
        
        # 每10个epoch输出详细指标
        if (epoch + 1) % 10 == 0 or epoch == Config.EPOCHS - 1:
            print(f"\n   📈 【训练集指标】")
            print(f"      平均损失: {train_metrics['avg_loss']:.6f}")
            print(f"      准确率: {train_metrics['accuracy']:.4f} | 平衡准确率: {train_metrics['balanced_accuracy']:.4f}")
            print(f"      ROC-AUC: {train_metrics['auc_roc']:.4f} | PR-AUC: {train_metrics['auc_pr']:.4f}")
            print(f"      F1(宏平均): {train_metrics['f1_score']:.4f} | MCC: {train_metrics['mcc']:.4f}")

        if epochs_without_improve >= early_stop_patience:
            print(
                f"\n🛑 Early Stopping 触发：验证集 ROC-AUC 连续 "
                f"{early_stop_patience} 个 epoch 未提升，提前结束训练。"
            )
            break

    # 加载最佳权重：导出 train / 内部 val / external（external 仅此一次终评）
    save_path = os.path.join(checkpoints_dir, "best_model.pth")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        export_kwargs = dict(
            model=model,
            hospital_name=hospital_name,
            logs_dir=logs_dir,
            device=device,
            use_focal_loss=use_focal_loss,
            class_weights=class_weights,
            use_wma=use_wma,
            train_class_counts=train_class_counts,
            wma_c=wma_c,
            wma_warmup_epochs=wma_warmup,
            wma_temperature=wma_temp,
            kl_annealing_epochs=kl_ann,
        )
        _, train_metrics_best = export_split_predictions(
            loader=train_loader,
            source_csv_path=train_csv,
            split_name='train',
            **export_kwargs,
        )
        _, val_metrics_best = export_split_predictions(
            loader=val_loader,
            source_csv_path=val_csv,
            split_name='val',
            **export_kwargs,
        )
        _, ext_metrics_best = export_split_predictions(
            loader=external_loader,
            source_csv_path=external_csv,
            split_name='external',
            **export_kwargs,
        )

        print(
            f"【最佳权重 · 训练集 train】ROC-AUC={train_metrics_best['auc_roc']:.4f}  "
            f"PR-AUC={train_metrics_best['auc_pr']:.4f}  "
            f"平衡准确率={train_metrics_best['balanced_accuracy']:.4f}"
        )
        print(
            f"【最佳权重 · 内部验证 val】ROC-AUC={val_metrics_best['auc_roc']:.4f}  "
            f"PR-AUC={val_metrics_best['auc_pr']:.4f}  "
            f"平衡准确率={val_metrics_best['balanced_accuracy']:.4f}"
        )
        print(
            f"【最佳权重 · 外部终评 external】ROC-AUC={ext_metrics_best['auc_roc']:.4f}  "
            f"PR-AUC={ext_metrics_best['auc_pr']:.4f}  "
            f"平衡准确率={ext_metrics_best['balanced_accuracy']:.4f}"
        )
    
    # 保存训练历史（保存到logs目录）
    import json
    history_path = os.path.join(logs_dir, "training_history.json")
    
    # 转换为可序列化的格式
    serializable_history = []
    for epoch_data in metrics_history:
        serialized = {
            'epoch': epoch_data['epoch'],
            'train_loss': float(epoch_data['train_loss']),
            'train': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                     for k, v in epoch_data['train'].items() if k != 'confusion_matrix'},
            'val': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                   for k, v in epoch_data['val'].items() if k != 'confusion_matrix'}
        }
        serializable_history.append(serialized)
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    print(f"\n 训练历史已保存至: {history_path}")
    
    # 输出最终总结（best_model.pth 仅与「内部验证集 ROC-AUC 最高」对齐）
    print(f"\n{'='*60}")
    print(f" 训练完成！")
    print(f"   best_model.pth 对应: 内部验证集 val ROC-AUC 最高 (AUC={best_auc:.4f})")
    print(f"   全程最高 阳性F1（参考，非选模依据）: {best_f1:.4f}")
    print(f"   全程最高 MCC（参考，非选模依据）: {best_mcc:.4f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
