import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from training.losses import evidence_loss

def train_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Training Ep {epoch+1}/{total_epochs}")
    for imgs, clinical, labels in pbar:
        imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
        
        # One-hot 标签 (EDL Loss 需要)
        y_onehot = F.one_hot(labels, num_classes=2).float()
        
        optimizer.zero_grad()
        
        # Forward (输出 Alpha)
        alpha = model(imgs, clinical)
        
        # Loss
        loss = evidence_loss(alpha, y_onehot, epoch, total_epochs)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(loader)

def validate(model, loader, device):
    """
    验证函数：同时输出 AUC 和 分流策略(Triage) 的效果
    """
    model.eval()
    probs = []
    uncertainties = []
    targets = []
    
    with torch.no_grad():
        for imgs, clinical, labels in loader:
            imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
            
            alpha = model(imgs, clinical)
            
            # 1. 计算预测概率: p = alpha / sum(alpha)
            S = torch.sum(alpha, dim=1, keepdim=True)
            p = alpha / S
            
            # 2. 计算不确定性: u = K / sum(alpha)
            # 这里的 K=2
            u = 2.0 / S
            
            probs.extend(p[:, 1].cpu().numpy()) # 取阳性概率
            uncertainties.extend(u.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy())
            
    # --- 指标计算 ---
    # 1. 常规 AUC
    try:
        auc = roc_auc_score(targets, probs)
    except:
        auc = 0.5
        
    # 2. Triage 分流模拟 (The Lancet 核心图表)
    # 假设我们剔除不确定性最高的 20% 病人，看剩下病人的准确率
    preds = [1 if x > 0.5 else 0 for x in probs]
    correct_list = [1 if p == t else 0 for p, t in zip(preds, targets)]
    
    # 组合数据: (不确定性, 是否预测正确)
    data = list(zip(uncertainties, correct_list))
    # 按不确定性从小到大排序 (最确定的在前面)
    data.sort(key=lambda x: x[0])
    
    # 取前 80% (High Confidence 样本)
    cutoff = int(len(data) * 0.8)
    if cutoff > 0:
        high_conf_data = data[:cutoff]
        # 计算这部分样本的准确率
        triage_acc = sum([x[1] for x in high_conf_data]) / len(high_conf_data)
    else:
        triage_acc = 0.0
        
    raw_acc = sum(correct_list) / len(correct_list)
        
    return auc, triage_acc, raw_acc
