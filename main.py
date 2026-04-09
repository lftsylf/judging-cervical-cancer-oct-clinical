import os
import sys
import torch
import torch.optim as optim
import pandas as pd


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
):
    metrics, pred_details = validate(
        model,
        loader,
        device,
        verbose=False,
        use_focal=use_focal_loss,
        class_weights=class_weights,
        epoch=Config.EPOCHS - 1,
        total_epochs=Config.EPOCHS,
        return_predictions=True,
    )

    df = pd.read_csv(source_csv_path).reset_index(drop=True)
    n = min(len(df), len(pred_details['targets']))
    split_label = {"development": "开发集", "external": "外部集"}.get(split_name, split_name)
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

    output_path = os.path.join(logs_dir, f"{split_name}_sample_predictions.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"【{split_label}】逐样本概率已保存: {output_path}")
    return output_path, metrics


def main():
    # 1. 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"【项目】OptiGenesis (Lancet)  |  【设备】{device}")
    
    # 2. 数据加载
    # 动态获取当前医院名称对应的 CSV 文件名
    hospital_name = Config.HOSPITAL_NAME.lower()
    run_folder = os.getenv("OPTIGENESIS_OUTPUT_RUN_NAME", "").strip().lower()
    if not run_folder or os.path.sep in run_folder or ".." in run_folder:
        run_folder = hospital_name
    train_csv = os.path.join(Config.DATA_ROOT, f"development_{hospital_name}.csv")
    val_csv = os.path.join(Config.DATA_ROOT, f"external_{hospital_name}.csv")
    run_output_dir = os.path.join(Config.OUTPUT_DIR, run_folder)
    checkpoints_dir = os.path.join(run_output_dir, "checkpoints")
    logs_dir = os.path.join(run_output_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"【数据目录】{Config.DATA_ROOT}")
    print(f"【输出目录】{run_output_dir}")
    print(f"【当前医院】{Config.HOSPITAL_NAME}  （开发集: {train_csv}  |  外部集: {val_csv}）")
    
    try:
        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            print("⚠️ 未找到对应 CSV，请先运行 data/prepare_loho_data.py 生成 development_*.csv 与 external_*.csv")
            # 为简单起见，这里假设用户已经正确配置了 Config 并且运行了 LOHO 数据预处理
            
        train_loader = get_dataloader(train_csv, mode='train')
        val_loader = get_dataloader(val_csv, mode='val')
        enable_coral = getattr(Config, "ENABLE_DOMAIN_CORAL", False)
        # CORAL 目标域与验证同一 CSV；复用 val_loader，避免再占一份 DataLoader/显存
        uda_target_loader = val_loader if enable_coral else None
        
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
        print("请确保已运行 prepare_loho_data.py 生成 development_*.csv 和 external_*.csv")
        return
    
    # 3. 模型构建
    print(f"【构建模型】骨干网络: {Config.BACKBONE}  |  临床特征融合: {Config.USE_CLINICAL}")
    model = OptiGenesis(
        model_name=Config.BACKBONE, 
        use_clinical=Config.USE_CLINICAL,
        num_classes=Config.NUM_CLASSES
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    use_ema = getattr(Config, "ENABLE_MODEL_EMA", False)
    ema_decay = getattr(Config, "EMA_DECAY", 0.999)
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    
    # 4. 训练循环
    best_auc = 0.0
    best_f1 = 0.0
    best_mcc = -1.0
    metrics_history = []
    
    print("【开始训练】…")
    print(" 使用 Focal Loss + 类别权重 + 过采样策略处理类别不平衡")
    print(f" 多模态辅助监督: {getattr(Config, 'ENABLE_MULTIMODAL_AUX_LOSS', False)}")
    if getattr(Config, "ENABLE_DOMAIN_CORAL", False):
        print(
            f" CORAL 域对齐: 开启 (λ_max={getattr(Config, 'CORAL_LAMBDA_MAX', 0.02)}, "
            f"warmup_epochs={getattr(Config, 'CORAL_WARMUP_EPOCHS', 8)})，目标域=external 折（无标签损失）"
        )
    else:
        print(" CORAL 域对齐: 关闭")
    print(" 选模与存盘: 仅当验证集 ROC-AUC 提升时保存 best_model.pth")
    if use_ema:
        print(f" Model EMA: 开启 (decay={ema_decay})，验证/存盘/导出均使用 EMA 权重")
    else:
        print(" Model EMA: 关闭")
    
    # 使用 Focal Loss 来处理极度不平衡的数据
    use_focal_loss = True
    
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            Config.EPOCHS,
            class_weights=class_weights,
            use_focal=use_focal_loss,
            enable_multimodal_aux=getattr(Config, 'ENABLE_MULTIMODAL_AUX_LOSS', False),
            aux_w_vision=getattr(Config, 'AUX_LOSS_WEIGHT_VISION', 0.2),
            aux_w_clinical=getattr(Config, 'AUX_LOSS_WEIGHT_CLINICAL', 0.2),
            ema=ema,
            uda_target_loader=uda_target_loader,
            lambda_coral_max=(
                getattr(Config, "CORAL_LAMBDA_MAX", 0.0)
                if getattr(Config, "ENABLE_DOMAIN_CORAL", False)
                else 0.0
            ),
            coral_warmup_epochs=getattr(Config, "CORAL_WARMUP_EPOCHS", 8),
        )
        
        # Validation - 获取完整指标（EMA 开启时用 shadow 权重评估）
        if ema is not None:
            ema.apply_shadow(model)
        try:
            val_metrics = validate(
                model,
                val_loader,
                device,
                verbose=(epoch % 10 == 0 or epoch == Config.EPOCHS - 1),
                use_focal=use_focal_loss,
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

    # 加载最佳权重，导出 development / external 逐样本概率（供阈值与曲线分析）
    save_path = os.path.join(checkpoints_dir, "best_model.pth")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        _, dev_metrics_best = export_split_predictions(
            model=model,
            loader=train_loader,
            source_csv_path=train_csv,
            split_name='development',
            hospital_name=hospital_name,
            logs_dir=logs_dir,
            device=device,
            use_focal_loss=use_focal_loss,
            class_weights=class_weights,
        )
        _, val_metrics_best = export_split_predictions(
            model=model,
            loader=val_loader,
            source_csv_path=val_csv,
            split_name='external',
            hospital_name=hospital_name,
            logs_dir=logs_dir,
            device=device,
            use_focal_loss=use_focal_loss,
            class_weights=class_weights,
        )

        print(
            f"【最佳权重 · 开发集】ROC-AUC={dev_metrics_best['auc_roc']:.4f}  "
            f"PR-AUC={dev_metrics_best['auc_pr']:.4f}  "
            f"平衡准确率={dev_metrics_best['balanced_accuracy']:.4f}"
        )
        print(
            f"【最佳权重 · 外部集】ROC-AUC={val_metrics_best['auc_roc']:.4f}  "
            f"PR-AUC={val_metrics_best['auc_pr']:.4f}  "
            f"平衡准确率={val_metrics_best['balanced_accuracy']:.4f}"
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
    
    # 输出最终总结（best_model.pth 仅与「验证集 ROC-AUC 最高」对齐）
    print(f"\n{'='*60}")
    print(f" 训练完成！")
    print(f"   best_model.pth 对应: 验证集 ROC-AUC 最高 (AUC={best_auc:.4f})")
    print(f"   全程最高 阳性F1（参考，非选模依据）: {best_f1:.4f}")
    print(f"   全程最高 MCC（参考，非选模依据）: {best_mcc:.4f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
