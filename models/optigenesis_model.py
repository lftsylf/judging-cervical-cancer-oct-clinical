import torch
import torch.nn as nn
import timm
from .uncertainty import UncertaintyHead

class OptiGenesis(nn.Module):
    """
    OptiGenesis Model: Lancet Edition
    架构: Swin Transformer (Vision) + MLP (Clinical) -> Fusion -> EDL Head
    """
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=2, use_clinical=True):
        super().__init__()
        self.use_clinical = use_clinical
        
        # 1. 视觉基座 (Swin Transformer)
        print(f"🔍 Loading Vision Backbone: {model_name}")
        # num_classes=0 表示移除原本的分类头，只提取特征
        self.vision_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.vision_dim = self.vision_backbone.num_features # Swin-Tiny通常是768
        
        # 2. 临床数据编码器 (MLP)
        if self.use_clinical:
            self.clinical_mlp = nn.Sequential(
                nn.Linear(3, 32), # 输入: Age, HPV, TCT
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
            fusion_input_dim = self.vision_dim + 64
        else:
            fusion_input_dim = self.vision_dim
            
        # 3. 多模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2) # 防止过拟合
        )
        
        # 4. 不确定性输出头
        self.uncertainty_head = UncertaintyHead(in_features=256, num_classes=num_classes)

    def forward(self, img, clinical):
        # A. 提取视觉特征
        v_feat = self.vision_backbone(img) # [B, 768]
        
        # B. 提取/融合临床特征
        if self.use_clinical:
            c_feat = self.clinical_mlp(clinical) # [B, 64]
            feat = torch.cat([v_feat, c_feat], dim=1) # [B, 832]
        else:
            feat = v_feat
            
        # C. 特征融合
        feat = self.fusion_layer(feat) # [B, 256]
        
        # D. 输出 Dirichlet 参数 Alpha
        alpha = self.uncertainty_head(feat)
        
        return alpha
