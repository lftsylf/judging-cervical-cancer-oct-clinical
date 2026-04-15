import torch
import torch.nn.functional as F


def _to_float_tensor(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.float().to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=torch.float32, device=device).to(dtype=dtype)


def wma_loss(
    alpha,
    y_onehot,
    epoch,
    num_classes=2,
    n_counts=None,
    c_margin=0.2,
    warmup_epochs=10,
    temperature=1.0,
    eps=1e-8,
    kl_annealing_epochs=10,
):
    """
    WMA Loss：重加权边距调整 + UNCE-MA / TCE-MA + 原始 EDL KL 正则。

    Args:
        alpha: Dirichlet 浓度参数 [B, K]，应已保证 > 0（证据层常用 softplus+1）。
        y_onehot: one-hot [B, K]
        epoch: 当前 epoch（0-based，与 train_epoch 一致）
        n_counts: 训练集各类样本数 [N_0, N_1, ...]；None 时退化为均匀权重、均匀边距尺度
        c_margin: 边距超参 C（默认 0.2）
        warmup_epochs: λ(t) 线性增长周期        temperature: TCE-MA 温度 τ
        kl_annealing_epochs: KL 退火周期（与 evidence_loss 口径一致：min(1, epoch / kl_annealing_epochs)）
    """
    device = alpha.device
    dtype = alpha.dtype
    B, K = alpha.shape[0], alpha.shape[1]
    if K != int(num_classes):
        num_classes = K

    # ---------- 先验计数 -> 类别权重 w_k与边距 m_k ----------
    if n_counts is None:
        inv = torch.ones(K, device=device, dtype=torch.float32)
    else:
        n_float = _to_float_tensor(n_counts, device, torch.float32).view(-1)
        if n_float.numel() != K:
            raise ValueError(f"n_counts length {n_float.numel()} != num_classes {K}")
        n_float = torch.clamp(n_float, min=1.0)
        inv = 1.0 / n_float

    w_k = (float(K) * inv) / torch.clamp(inv.sum(), min=eps)
    m_k = c_margin * (inv.pow(0.25)) / torch.clamp(inv.pow(0.25).max(), min=eps)

    # λ(t)：随 epoch 线性升至 1（1-based 与常见 warmup 一致）
    wu = max(1, int(warmup_epochs))
    lam = min(1.0, float(epoch + 1) / float(wu))

    S = alpha.sum(dim=1, keepdim=True)
    s_bar = S.detach()

    m_row = m_k.to(device=device, dtype=dtype).view(1, K)
    Delta = m_row * lam * s_bar

    alpha_prime = F.relu(alpha - y_onehot * Delta - 1.0) + 1.0
    delta_true = (y_onehot * Delta).sum(dim=1, keepdim=True)
    S_prime = S - delta_true

    # 数值稳定：浓度与总量下限（K 类 Dirichlet 在 alpha_k>=1 时 S>=K）
    alpha_prime = torch.clamp(alpha_prime, min=1.0 + eps)
    S_prime = torch.clamp(S_prime, min=float(K))

    # ---------- 组件 A: UNCE-MA ----------
    psi_S = torch.digamma(S_prime.squeeze(-1))
    alpha_true = torch.sum(alpha_prime * y_onehot, dim=1)
    psi_alpha_t = torch.digamma(alpha_true)
    loss_a = psi_S - psi_alpha_t

    # ---------- 组件 B: TCE-MA ----------
    den = torch.sum((alpha_prime - 1.0 + eps) * y_onehot, dim=1)
    den = torch.clamp(den, min=eps)
    num = temperature * S_prime.squeeze(-1)
    num = torch.clamp(num, min=eps)
    loss_b = torch.log(num / den)

    # ---------- 样本级类别权重（真实类） ----------
    true_class = torch.argmax(y_onehot, dim=1)
    w_sample = w_k.to(device=device, dtype=dtype)[true_class]
    main = w_sample * (loss_a + loss_b)

    # ---------- 原始 EDL KL（在未调整 alpha 上，与 evidence_loss 同形） ----------
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    num_classes_tensor = torch.tensor(float(num_classes), device=device, dtype=dtype)
    kl = (
        torch.lgamma(S_tilde)
        - torch.lgamma(num_classes_tensor)
        - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True)
        + torch.sum(
            (alpha_tilde - 1.0)
            * (torch.digamma(S_tilde) - torch.digamma(num_classes_tensor)),
            dim=1,
            keepdim=True,
        )
    ).squeeze(-1)

    ka = max(1, int(kl_annealing_epochs))
    annealing_coef = min(1.0, float(epoch) / float(ka))

    per_sample = main + annealing_coef * kl
    return torch.mean(per_sample)

def evidence_loss(alpha, y_onehot, epoch, total_epochs, num_classes=2, class_weights=None):
    """
    EDL Loss 计算（带类别权重）:
    Loss = 准确性损失 (MSE) + KL散度正则化
    
    Args:
        alpha: 模型输出的 Dirichlet 参数 [Batch, K]
        y_onehot: 真实标签的 One-hot 编码 [Batch, K]
        epoch: 当前轮次 (用于退火)
        class_weights: 类别权重 [K]，用于处理类别不平衡
    """
    # 1. 计算预测概率 p 和 总证据 S
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S 
    
    # 2. 准确性损失 (Sum of Squares Error, Type II ML)
    err = torch.sum((y_onehot - p) ** 2, dim=1, keepdim=True)
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loss_acc = torch.sum(err + var, dim=1)
    
    # 3. 应用类别权重（如果提供）
    if class_weights is not None:
        # 获取每个样本的真实类别
        true_class = torch.argmax(y_onehot, dim=1)  # [Batch]
        sample_weights = class_weights[true_class]  # [Batch]
        loss_acc = loss_acc * sample_weights
    
    # 4. KL 散度正则化 (惩罚"不知道还乱猜")
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
    
    # KL 散度计算
    num_classes_tensor = torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype)
    kl = torch.lgamma(S_tilde) - torch.lgamma(num_classes_tensor) \
         - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) \
         + torch.sum((alpha_tilde - 1) * (torch.digamma(S_tilde) - torch.digamma(num_classes_tensor)), dim=1, keepdim=True)
         
    # 5. 退火系数 (Annealing)
    annealing_coef = min(1.0, epoch / 10.0) 
    
    final_loss = torch.mean(loss_acc + annealing_coef * torch.mean(kl))
    return final_loss

def focal_loss_with_edl(alpha, y_onehot, epoch, total_epochs, num_classes=2, alpha_focal=0.25, gamma=2.0, class_weights=None):
    """
    结合 Focal Loss 和 EDL Loss 的混合损失函数
    用于处理极度不平衡的数据集
    
    Args:
        alpha: Dirichlet 参数 [Batch, K]
        y_onehot: One-hot 标签 [Batch, K]
        alpha_focal: Focal Loss 的 alpha 参数
        gamma: Focal Loss 的 gamma 参数（聚焦参数）
        class_weights: 类别权重
    """
    # 计算预测概率
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / S  # [Batch, K]
    
    # Focal Loss 部分
    # 获取真实类别对应的概率
    pt = torch.sum(p * y_onehot, dim=1)  # [Batch]
    # 计算 focal weight
    focal_weight = (1 - pt) ** gamma
    # Focal Loss
    ce_loss = -torch.log(pt + 1e-8)
    focal_loss = alpha_focal * focal_weight * ce_loss
    
    # 应用类别权重
    if class_weights is not None:
        true_class = torch.argmax(y_onehot, dim=1)
        sample_weights = class_weights[true_class]
        focal_loss = focal_loss * sample_weights
    
    # EDL 正则化部分（简化版，只保留方差项）
    var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1)
    
    # 组合损失
    final_loss = torch.mean(focal_loss + 0.1 * var)  # 0.1是正则化系数
    return final_loss


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CORAL：对齐源/目标特征的二阶统计（协方差），用于无监督域对齐。
    source, target: [B, D]，要求 B>=2 时协方差更稳定；B=1 时退化为 0。
    """
    d = source.size(1)
    if source.size(0) < 2 or target.size(0) < 2:
        return source.sum() * 0.0

    def _cov(x):
        xm = x - x.mean(dim=0, keepdim=True)
        return (xm.t() @ xm) / (xm.size(0) - 1 + 1e-5)

    cs = _cov(source)
    ct = _cov(target)
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)
