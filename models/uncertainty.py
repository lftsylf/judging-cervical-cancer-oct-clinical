import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyHead(nn.Module):
    """
    Evidence-based Uncertainty Head (EDL).
    输出不是概率，而是'证据'(Evidence)。
    Reference: Evidential Deep Learning to Quantify Classification Uncertainty (NeurIPS 2018)
    """
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # 1. 计算 Logits
        logits = self.linear(x)
        
        # 2. Evidence 必须非负 (使用 Softplus 激活)
        evidence = F.softplus(logits)
        
        # 3. Alpha 参数 (Dirichlet分布参数 = evidence + 1)
        alpha = evidence + 1
        
        return alpha
