import torch
import torch.nn.functional as F

def evidence_loss(alpha, y_onehot, epoch, total_epochs, num_classes=2):
    """
    EDL Loss 计算:
    Loss = 准确性损失 (MSE) + KL散度正则化
    
    Args:
        alpha: 模型输出的 Dirichlet 参数 [Batch, K]
        y_onehot: 真实标签的 One-hot 编码 [Batch, K]
        epoch: 当前轮次 (用于退火)
    """
    # 1. 计算预测概率 p 和 总证据 S
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S 
    
    # 2. 准确性损失 (Sum of Squares Error, Type II ML)
    # A = (y - p)^2
    # B = p * (1-p) / (S+1)
    err = torch.sum((y_onehot - p) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loss_acc = torch.sum(err + var, dim=1)
    
    # 3. KL 散度正则化 (惩罚"不知道还乱猜")
    # 目标: 当样本是OOD或难以分类时，Alpha应该趋向于全1 (Uniform Distribution)
    # 构造目标 Alpha_tilde
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    
    # KL 散度计算 (Dirichlet KL divergence)
    # KL(Dir(alpha_tilde) || Dir([1,1,...,1]))
    num_classes_tensor = torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype)
    kl = torch.lgamma(S_tilde) - torch.lgamma(num_classes_tensor) \
         - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
         + torch.sum((alpha_tilde - 1) * (torch.digamma(S_tilde) - torch.digamma(num_classes_tensor)), dim=1, keepdim=True)
         
    # 4. 退火系数 (Annealing)
    # 前期不加 KL，让模型先学特征；后期慢慢加
    annealing_coef = min(1.0, epoch / 10.0) 
    
    final_loss = torch.mean(loss_acc + annealing_coef * torch.mean(kl))
    return final_loss
