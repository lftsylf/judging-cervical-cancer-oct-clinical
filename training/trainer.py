import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score,
    balanced_accuracy_score,  # 平衡准确率
    precision_recall_fscore_support, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,  # PR-AUC
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)
from training.losses import evidence_loss, focal_loss_with_edl, coral_loss, wma_loss
import numpy as np
from collections import defaultdict
from itertools import cycle

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    class_weights=None,
    use_focal=False,
    use_wma=False,
    train_class_counts=None,
    wma_c=0.2,
    wma_warmup_epochs=10,
    wma_temperature=1.0,
    kl_annealing_epochs=10,
    enable_multimodal_aux=False,
    aux_w_vision=0.2,
    aux_w_clinical=0.2,
    ema=None,
    uda_target_loader=None,
    lambda_coral_max=0.0,
    coral_warmup_epochs=8,
):
    model.train()
    running_loss = 0.0
    use_coral = uda_target_loader is not None and float(lambda_coral_max) > 0.0
    target_iter = cycle(uda_target_loader) if use_coral else None
    lambda_eff = float(lambda_coral_max) * min(
        1.0, float(epoch + 1) / float(max(1, int(coral_warmup_epochs)))
    )
    
    pbar = tqdm(loader, desc=f"训练 {epoch+1}/{total_epochs}")
    for imgs, clinical, labels in pbar:
        imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
        
        # One-hot 标签 (EDL Loss 需要)
        y_onehot = F.one_hot(labels, num_classes=2).float()
        
        optimizer.zero_grad()
        
        if use_coral:
            imgs_t, clin_t, _ = next(target_iter)
            imgs_t, clin_t = imgs_t.to(device), clin_t.to(device)

        # Forward (Alpha + 可选 aux + 可选 CORAL 特征)
        if enable_multimodal_aux and use_coral:
            alpha, aux_logits_v, aux_logits_c, feat_s = model(
                imgs, clinical, return_aux=True, return_coral_feat=True
            )
            _, feat_t = model(imgs_t, clin_t, return_coral_feat=True)
        elif enable_multimodal_aux:
            alpha, aux_logits_v, aux_logits_c = model(imgs, clinical, return_aux=True)
            feat_s, feat_t = None, None
        elif use_coral:
            alpha, feat_s = model(imgs, clinical, return_coral_feat=True)
            _, feat_t = model(imgs_t, clin_t, return_coral_feat=True)
            aux_logits_v, aux_logits_c = None, None
        else:
            alpha = model(imgs, clinical)
            aux_logits_v, aux_logits_c = None, None
            feat_s, feat_t = None, None
        
        # Loss - 根据参数选择损失函数
        if use_wma:
            loss = wma_loss(
                alpha,
                y_onehot,
                epoch,
                num_classes=2,
                n_counts=train_class_counts,
                c_margin=wma_c,
                warmup_epochs=wma_warmup_epochs,
                temperature=wma_temperature,
                kl_annealing_epochs=kl_annealing_epochs,
            )
        elif use_focal:
            loss = focal_loss_with_edl(alpha, y_onehot, epoch, total_epochs, class_weights=class_weights)
        else:
            loss = evidence_loss(alpha, y_onehot, epoch, total_epochs, class_weights=class_weights)
        
        # 可选：单模态辅助监督（CE）
        if enable_multimodal_aux:
            aux_loss_v = F.cross_entropy(aux_logits_v, labels)
            aux_loss_c = (
                F.cross_entropy(aux_logits_c, labels)
                if aux_logits_c is not None
                else torch.zeros((), device=device, dtype=loss.dtype)
            )
            loss = loss + float(aux_w_vision) * aux_loss_v + float(aux_w_clinical) * aux_loss_c

        # 可选：CORAL（无标签目标域 = 外部折，仅用特征统计对齐）
        if use_coral and feat_s is not None and feat_t is not None:
            m = min(feat_s.size(0), feat_t.size(0))
            loss = loss + lambda_eff * coral_loss(feat_s[:m], feat_t[:m])
        
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        
        running_loss += loss.item()
        pbar.set_postfix({'损失': f"{loss.item():.4f}"})
    
    return running_loss / len(loader)
def validate(
    model,
    loader,
    device,
    verbose=True,
    use_focal=False,
    use_wma=False,
    train_class_counts=None,
    wma_c=0.2,
    wma_warmup_epochs=10,
    wma_temperature=1.0,
    kl_annealing_epochs=10,
    class_weights=None,
    epoch=0,
    total_epochs=50,
    return_predictions=False,
):
    """
    验证集/任意划分上的指标计算；可选返回逐样本概率等明细。

    返回:
        return_predictions=False 时: metrics 字典
        return_predictions=True 时: (metrics, prediction_details)
    """
    model.eval()
    probs = []
    uncertainties = []
    targets = []
    losses = []
    
    # 收集所有batch的预测结果和损失
    with torch.no_grad():
        for imgs, clinical, labels in loader:
            imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
            
            alpha = model(imgs, clinical)
            
            # 计算损失（用于验证集的平均损失）
            y_onehot = F.one_hot(labels, num_classes=2).float()
            if use_wma:
                loss = wma_loss(
                    alpha,
                    y_onehot,
                    epoch,
                    num_classes=2,
                    n_counts=train_class_counts,
                    c_margin=wma_c,
                    warmup_epochs=wma_warmup_epochs,
                    temperature=wma_temperature,
                    kl_annealing_epochs=kl_annealing_epochs,
                )
            elif use_focal:
                from training.losses import focal_loss_with_edl
                loss = focal_loss_with_edl(alpha, y_onehot, epoch, total_epochs, class_weights=class_weights)
            else:
                from training.losses import evidence_loss
                loss = evidence_loss(alpha, y_onehot, epoch, total_epochs, class_weights=class_weights)
            losses.append(loss.item())
            
            # 1. 计算预测概率: p = alpha / sum(alpha)
            S = torch.sum(alpha, dim=1, keepdim=True)
            p = alpha / S
            
            # 2. 计算不确定性: u = K / sum(alpha)
            u = 2.0 / S
            
            probs.extend(p[:, 1].cpu().numpy())  # 取阳性概率
            uncertainties.extend(u.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy())
    
    probs = np.array(probs)
    targets = np.array(targets)
    uncertainties = np.array(uncertainties)
    
    # 转换为二分类预测（阈值0.5）
    preds = (probs > 0.5).astype(int)
    
    # 初始化指标字典
    metrics = {}
    
    # ========== 0. 平均损失 ==========
    metrics['avg_loss'] = np.mean(losses)
    
    # ========== 1. 基本分类指标 ==========
    try:
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, preds)
        
        # Balanced Accuracy (平衡准确率，适合不平衡数据)
        metrics['balanced_accuracy'] = balanced_accuracy_score(targets, preds)
        
        # Precision, Recall, F1 (per class)
        precision_per_class = precision_score(targets, preds, average=None, zero_division=0, labels=[0, 1])
        recall_per_class = recall_score(targets, preds, average=None, zero_division=0, labels=[0, 1])
        f1_per_class = f1_score(targets, preds, average=None, zero_division=0, labels=[0, 1])
        
        # Weighted平均（考虑类别不平衡，更有意义）
        metrics['precision_weighted'] = precision_score(targets, preds, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(targets, preds, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, preds, average='weighted', zero_division=0)
        
        # F1-Score (宏平均，用于整体评估)
        metrics['f1_score'] = f1_score(targets, preds, average='macro', zero_division=0)
        
        # 阳性类别（类别1）的关键指标（最重要）
        metrics['precision_positive'] = precision_per_class[1] if len(precision_per_class) > 1 else 0.0  # PPV
        metrics['recall_positive'] = recall_per_class[1] if len(recall_per_class) > 1 else 0.0  # Sensitivity/TPR
        metrics['f1_positive'] = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
        
    except Exception as e:
        if verbose:
            print(f"   >> [警告] 基本分类指标计算错误: {e}")
        for key in ['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 
                   'f1_weighted', 'f1_score', 'precision_positive', 'recall_positive', 'f1_positive']:
            metrics[key] = 0.0
    
    # ========== 2. AUC指标 ==========
    try:
        # ROC-AUC
        if len(np.unique(targets)) >= 2:
            metrics['auc_roc'] = roc_auc_score(targets, probs)
        else:
            metrics['auc_roc'] = 0.5
    except:
        metrics['auc_roc'] = 0.5
    
    try:
        # PR-AUC (Precision-Recall AUC, 更适合不平衡数据)
        if len(np.unique(targets)) >= 2:
            metrics['auc_pr'] = average_precision_score(targets, probs)
        else:
            metrics['auc_pr'] = 0.0
    except:
        metrics['auc_pr'] = 0.0
    
    # ========== 3. 混淆矩阵相关指标 ==========
    try:
        cm = confusion_matrix(targets, preds, labels=[0, 1])
        metrics['confusion_matrix'] = cm
        
        if cm.size == 4:  # 2x2矩阵
            tn, fp, fn, tp = cm.ravel()
            
            # Sensitivity (Recall of positive class, 敏感度/召回率)
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity (Recall of negative class, 特异度)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Positive Predictive Value (PPV, 阳性预测值 = Precision)
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Negative Predictive Value (NPV, 阴性预测值)
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # 混淆矩阵元素
            metrics['tp'] = int(tp)
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
        else:
            metrics['sensitivity'] = 0.0
            metrics['specificity'] = 0.0
            metrics['ppv'] = 0.0
            metrics['npv'] = 0.0
            metrics['tp'] = 0
            metrics['tn'] = 0
            metrics['fp'] = 0
            metrics['fn'] = 0
    except Exception as e:
        if verbose:
            print(f"   >> [警告] 混淆矩阵计算错误: {e}")
        for key in ['sensitivity', 'specificity', 'ppv', 'npv', 'tp', 'tn', 'fp', 'fn']:
            metrics[key] = 0.0
    
    # ========== 4. MCC (Matthews Correlation Coefficient) ==========
    try:
        # MCC (综合评估指标，适合不平衡数据)
        metrics['mcc'] = matthews_corrcoef(targets, preds)
    except:
        metrics['mcc'] = 0.0
    
    # ========== 5. Triage分流策略指标 ==========
    correct_list = (preds == targets).astype(int)
    data = list(zip(uncertainties, correct_list))
    data.sort(key=lambda x: x[0])  # 按不确定性从小到大排序
    
    cutoff = int(len(data) * 0.8)
    if cutoff > 0:
        high_conf_data = data[:cutoff]
        metrics['triage_acc'] = sum([x[1] for x in high_conf_data]) / len(high_conf_data)
    else:
        metrics['triage_acc'] = 0.0
    
    metrics['raw_acc'] = metrics['accuracy']  # 别名
    
    # ========== 6. 统计信息 ==========
    metrics['num_samples'] = len(targets)
    metrics['num_positive'] = int(np.sum(targets == 1))
    metrics['num_negative'] = int(np.sum(targets == 0))
    metrics['mean_uncertainty'] = float(np.mean(uncertainties))
    metrics['mean_prob_positive'] = float(np.mean(probs))
    
    # ========== 打印结果（分类任务专用格式）==========
    if verbose:
        print(f"\n   📊 ========== Epoch {epoch+1} 验证统计 ==========")
        print(f"     - 平均损失: {metrics['avg_loss']:.6f}")
        print(f"     - 准确率: {metrics['accuracy']:.4f}")
        print(f"     - 平衡准确率: {metrics['balanced_accuracy']:.4f}")
        print(f"     - ROC-AUC: {metrics['auc_roc']:.4f}")
        print(f"     - PR-AUC: {metrics['auc_pr']:.4f}")
        print(f"     - F1(宏平均): {metrics['f1_score']:.4f}")
        print(f"     - MCC: {metrics['mcc']:.4f}")
        
        print(f"\n     📋 二分类详细指标:")
        print(f"     - 阳性精确率(PPV): {metrics['ppv']:.4f}")
        print(f"     - 敏感度(召回): {metrics['sensitivity']:.4f}")
        print(f"     - 特异度: {metrics['specificity']:.4f}")
        print(f"     - 阴性预测值(NPV): {metrics['npv']:.4f}")
        print(f"     - 精确率(加权): {metrics['precision_weighted']:.4f}")
        print(f"     - 召回率(加权): {metrics['recall_weighted']:.4f}")
        
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            print(f"\n     【混淆矩阵】")
            print(f"      [[TN={cm[0,0]:d}, FP={cm[0,1]:d}]")
            print(f"       [FN={cm[1,0]:d}, TP={cm[1,1]:d}]]")
        
        print(f"   ===========================================\n")
    
    if return_predictions:
        prediction_details = {
            'targets': targets.copy(),
            'probs': probs.copy(),
            'uncertainties': uncertainties.copy(),
            'preds': preds.copy(),
        }
        return metrics, prediction_details
    return metrics
