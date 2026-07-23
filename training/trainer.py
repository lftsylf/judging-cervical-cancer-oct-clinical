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
from data.dataset_lancet import unpack_loader_batch
import numpy as np
from collections import defaultdict
from itertools import cycle


def _pad_stack_2d(arrays, pad_value=0.0):
    """将若干 [B_i, N_i]（或 [N_i]）数组 pad 到相同 N 后沿 axis=0 拼接。"""
    if not arrays:
        return np.zeros((0, 0), dtype=np.float32)
    normed = []
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[None, :]
        normed.append(a)
    max_n = max(int(a.shape[1]) for a in normed)
    out = []
    for a in normed:
        if a.shape[1] < max_n:
            pad = np.full(
                (a.shape[0], max_n - a.shape[1]),
                pad_value,
                dtype=a.dtype,
            )
            a = np.concatenate([a, pad], axis=1)
        out.append(a)
    return np.concatenate(out, axis=0)

def _patient_edl_loss(
    alpha,
    y_onehot,
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
):
    if use_wma:
        return wma_loss(
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
    if use_focal:
        return focal_loss_with_edl(
            alpha, y_onehot, epoch, total_epochs, class_weights=class_weights
        )
    return evidence_loss(
        alpha, y_onehot, epoch, total_epochs, class_weights=class_weights
    )


def _frame_aux_loss(
    frame_alpha,
    labels,
    y_onehot,
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
    loss_type="edl",
    frame_mask=None,
):
    """
    帧级弱监督：每帧共用患者标签。
    frame_alpha: [B, N, K]
    frame_mask: 可选 [B, N]，只对有效帧（非 pad）计损失。
    """
    b, n, k = frame_alpha.shape
    if frame_mask is not None:
        valid = frame_mask > 0.5  # [B, N]
        if not bool(valid.any()):
            return frame_alpha.new_zeros(())
        alpha_flat = frame_alpha[valid]  # [M, K]
        y_flat = y_onehot.unsqueeze(1).expand(-1, n, -1)[valid]
        labels_flat = labels.unsqueeze(1).expand(-1, n)[valid]
    else:
        alpha_flat = frame_alpha.reshape(b * n, k)
        y_flat = y_onehot.unsqueeze(1).expand(-1, n, -1).reshape(b * n, k)
        labels_flat = labels.unsqueeze(1).expand(-1, n).reshape(b * n)

    if str(loss_type).lower() == "ce":
        # p = α / Σα，对 log p 做 CE（弱监督，数值上简单）
        s = alpha_flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_p = torch.log((alpha_flat / s).clamp_min(1e-8))
        return F.nll_loss(log_p, labels_flat)

    return _patient_edl_loss(
        alpha_flat,
        y_flat,
        epoch,
        total_epochs,
        class_weights=class_weights,
        use_focal=use_focal,
        use_wma=use_wma,
        train_class_counts=train_class_counts,
        wma_c=wma_c,
        wma_warmup_epochs=wma_warmup_epochs,
        wma_temperature=wma_temperature,
        kl_annealing_epochs=kl_annealing_epochs,
    )


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
    enable_frame_aux=False,
    frame_aux_weight=0.2,
    frame_aux_type="edl",
    ema=None,
    uda_target_loader=None,
    lambda_coral_max=0.0,
    coral_warmup_epochs=8,
):
    model.train()
    running_loss = 0.0
    use_coral = uda_target_loader is not None and float(lambda_coral_max) > 0.0
    need_frame = bool(enable_frame_aux)
    target_iter = cycle(uda_target_loader) if use_coral else None
    lambda_eff = float(lambda_coral_max) * min(
        1.0, float(epoch + 1) / float(max(1, int(coral_warmup_epochs)))
    )
    
    pbar = tqdm(loader, desc=f"训练 {epoch+1}/{total_epochs}")
    for batch in pbar:
        imgs, clinical, labels, frame_mask = unpack_loader_batch(batch)
        imgs = imgs.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)
        frame_mask = frame_mask.to(device)
        
        # One-hot 标签 (EDL Loss 需要)
        y_onehot = F.one_hot(labels, num_classes=2).float()
        
        optimizer.zero_grad()
        
        if use_coral:
            t_batch = next(target_iter)
            imgs_t, clin_t, _, mask_t = unpack_loader_batch(t_batch)
            imgs_t, clin_t = imgs_t.to(device), clin_t.to(device)
            mask_t = mask_t.to(device)

        # 前向：得到 alpha，以及可选的 aux / 帧明细 / CORAL 用特征
        frame_details = None
        fw = dict(frame_mask=frame_mask)
        if enable_multimodal_aux and use_coral and need_frame:
            alpha, aux_logits_v, aux_logits_c, feat_s, frame_details = model(
                imgs, clinical, return_aux=True, return_coral_feat=True, return_frame_details=True, **fw
            )
            _, feat_t = model(imgs_t, clin_t, return_coral_feat=True, frame_mask=mask_t)
        elif enable_multimodal_aux and need_frame:
            alpha, aux_logits_v, aux_logits_c, frame_details = model(
                imgs, clinical, return_aux=True, return_frame_details=True, **fw
            )
            feat_s, feat_t = None, None
        elif enable_multimodal_aux and use_coral:
            alpha, aux_logits_v, aux_logits_c, feat_s = model(
                imgs, clinical, return_aux=True, return_coral_feat=True, **fw
            )
            _, feat_t = model(imgs_t, clin_t, return_coral_feat=True, frame_mask=mask_t)
        elif enable_multimodal_aux:
            alpha, aux_logits_v, aux_logits_c = model(imgs, clinical, return_aux=True, **fw)
            feat_s, feat_t = None, None
        elif use_coral and need_frame:
            alpha, feat_s, frame_details = model(
                imgs, clinical, return_coral_feat=True, return_frame_details=True, **fw
            )
            _, feat_t = model(imgs_t, clin_t, return_coral_feat=True, frame_mask=mask_t)
            aux_logits_v, aux_logits_c = None, None
        elif use_coral:
            alpha, feat_s = model(imgs, clinical, return_coral_feat=True, **fw)
            _, feat_t = model(imgs_t, clin_t, return_coral_feat=True, frame_mask=mask_t)
            aux_logits_v, aux_logits_c = None, None
        elif need_frame:
            alpha, frame_details = model(imgs, clinical, return_frame_details=True, **fw)
            aux_logits_v, aux_logits_c = None, None
            feat_s, feat_t = None, None
        else:
            alpha = model(imgs, clinical, **fw)
            aux_logits_v, aux_logits_c = None, None
            feat_s, feat_t = None, None
        
        # 患者级主损失
        loss = _patient_edl_loss(
            alpha,
            y_onehot,
            epoch,
            total_epochs,
            class_weights=class_weights,
            use_focal=use_focal,
            use_wma=use_wma,
            train_class_counts=train_class_counts,
            wma_c=wma_c,
            wma_warmup_epochs=wma_warmup_epochs,
            wma_temperature=wma_temperature,
            kl_annealing_epochs=kl_annealing_epochs,
        )
        
        # 可选：单模态辅助监督（CE）
        if enable_multimodal_aux:
            aux_loss_v = F.cross_entropy(aux_logits_v, labels)
            aux_loss_c = (
                F.cross_entropy(aux_logits_c, labels)
                if aux_logits_c is not None
                else torch.zeros((), device=device, dtype=loss.dtype)
            )
            loss = loss + float(aux_w_vision) * aux_loss_v + float(aux_w_clinical) * aux_loss_c

        # 可选：帧级弱监督（每帧共用患者标签；非多模态）
        if need_frame and frame_details is not None:
            frame_alpha = frame_details.get("frame_alpha")
            if frame_alpha is not None:
                floss = _frame_aux_loss(
                    frame_alpha,
                    labels,
                    y_onehot,
                    epoch,
                    total_epochs,
                    class_weights=class_weights,
                    use_focal=use_focal,
                    use_wma=use_wma,
                    train_class_counts=train_class_counts,
                    wma_c=wma_c,
                    wma_warmup_epochs=wma_warmup_epochs,
                    wma_temperature=wma_temperature,
                    kl_annealing_epochs=kl_annealing_epochs,
                    loss_type=frame_aux_type,
                    frame_mask=frame_mask,
                )
                loss = loss + float(frame_aux_weight) * floss
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
    # 帧级明细（uncertainty_weighted / equal 时有意义；mean 模式为占位）
    frame_uncertainties_all = []
    frame_weights_all = []
    review_indices_all = []
    
    # 收集所有batch的预测结果和损失
    with torch.no_grad():
        for batch in loader:
            imgs, clinical, labels, frame_mask = unpack_loader_batch(batch)
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device)
            frame_mask = frame_mask.to(device)
            
            # 需要帧级复核信息时打开 return_frame_details；训练损失仍只用患者级 alpha
            alpha, frame_details = model(
                imgs, clinical, return_frame_details=True, frame_mask=frame_mask
            )            
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
            
            # 2. 患者级不确定性: u = K / sum(alpha)
            u = 2.0 / S
            
            probs.extend(p[:, 1].cpu().numpy())  # 取阳性概率
            uncertainties.extend(u.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy())

            fu = frame_details["frame_uncertainty"].detach().cpu().numpy()
            fw = frame_details["frame_weights"].detach().cpu().numpy()
            ri = frame_details["review_frame_indices"].detach().cpu().numpy()
            frame_uncertainties_all.append(fu)
            frame_weights_all.append(fw)
            review_indices_all.append(ri)
    
    probs = np.array(probs)
    targets = np.array(targets)
    uncertainties = np.array(uncertainties)
    if frame_uncertainties_all:
        # 全时序展开后辽宁 N≈60、华西/湘雅 N≈120；batch_size=1 时各 batch 的 N 不同
        frame_uncertainties_arr = _pad_stack_2d(frame_uncertainties_all, pad_value=0.0)
        frame_weights_arr = _pad_stack_2d(frame_weights_all, pad_value=0.0)
        review_indices_arr = _pad_stack_2d(review_indices_all, pad_value=-1)
    else:
        frame_uncertainties_arr = np.zeros((0, 0), dtype=np.float32)
        frame_weights_arr = np.zeros((0, 0), dtype=np.float32)
        review_indices_arr = np.zeros((0, 0), dtype=np.int64)
    
    # 转换为二分类预测（阈值0.5）
    preds = (probs > 0.5).astype(int)
    
    # 初始化指标字典
    metrics = {}
    
    # ========== 0. 平均损失 ==========
    metrics['avg_loss'] = np.mean(losses)
    
    # ========== 1. 基本分类指标 ==========
    try:
        # 准确率
        metrics['accuracy'] = accuracy_score(targets, preds)
        
        # 平衡准确率（适合类别不平衡）
        metrics['balanced_accuracy'] = balanced_accuracy_score(targets, preds)
        
        # 各类别 precision / recall / F1
        precision_per_class = precision_score(targets, preds, average=None, zero_division=0, labels=[0, 1])
        recall_per_class = recall_score(targets, preds, average=None, zero_division=0, labels=[0, 1])
        f1_per_class = f1_score(targets, preds, average=None, zero_division=0, labels=[0, 1])
        
        # 加权平均（考虑类别不平衡，有时比 macro 更贴近整体）
        metrics['precision_weighted'] = precision_score(targets, preds, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(targets, preds, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, preds, average='weighted', zero_division=0)
        
        # F1-Score (宏平均，用于整体评估)
        metrics['f1_score'] = f1_score(targets, preds, average='macro', zero_division=0)
        
        # 阳性类别（类别1）的关键指标（最重要）
        metrics['precision_positive'] = precision_per_class[1] if len(precision_per_class) > 1 else 0.0  # PPV
        metrics['recall_positive'] = recall_per_class[1] if len(recall_per_class) > 1 else 0.0  # sensitivity / TPR
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
            
            # sensitivity：阳性类召回率
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # specificity：阴性类召回率
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # PPV：阳性预测值（precision 在阳性上）
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # NPV：阴性预测值
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
            # 帧级：供人工复核高不确定 B-scan（每行一个患者）
            'frame_uncertainties': frame_uncertainties_arr.copy(),
            'frame_weights': frame_weights_arr.copy(),
            'review_frame_indices': review_indices_arr.copy(),
        }
        return metrics, prediction_details
    return metrics
